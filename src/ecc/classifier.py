"""The fine-tuned classifier in PyTorch, shared by train and export.

Margin = log P(" ban") - log P(" ok") at the token after a prompt's last line (the target). Base models load from the
local Hugging Face cache or a model folder.
torch is imported where it is used, so `uv run status` does not load it.
"""
import os
from functools import cache

os.environ.setdefault("HF_HUB_OFFLINE", "1")
# Scoring pads each batch to a new length, so tensor sizes grow batch after batch. Without expandable segments the
# allocator keeps every outgrown block cached: on the full val set its reservation passed 28 GiB while 8.3 GiB was in use.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

ANSWERS = (" ban", " ok")
CUDA_HEADROOM = 2**30   # left free for the CUDA context and libraries, which allocate outside PyTorch's cap


@cache
def device():
    """"cuda" when there is a GPU, else "cpu". On the GPU, PyTorch may use what the card has free now, less
    CUDA_HEADROOM. Past the card, the Windows driver spills into shared system memory and runs many times slower; with
    the cap, the allocator first releases its cache and then fails with an out-of-memory error."""
    import torch
    if not torch.cuda.is_available():
        return "cpu"
    free, total = torch.cuda.mem_get_info()
    torch.cuda.set_per_process_memory_fraction((free - CUDA_HEADROOM) / total)
    return "cuda"


def base_model(cfg, base):
    """A [train.models.*] base: a model folder (a configured path, e.g. under {data}) or a Hugging Face id in the local
    cache."""
    folder = cfg.path(base)
    return str(folder) if folder.is_dir() else base


def token_count(tok):
    return lambda text: len(tok.encode(text, add_special_tokens=False))


def encode(tok, prompts, chunk=4096):
    """Token ids of each prompt as int32 views into one flat array. A list of Python ints costs about 8 times the
    memory, which is gigabytes for a training subset."""
    import numpy as np
    parts, lengths = [], []
    for k in range(0, len(prompts), chunk):
        ids = tok(prompts[k:k + chunk], add_special_tokens=False).input_ids
        parts.append(np.fromiter((t for s in ids for t in s), dtype=np.int32))
        lengths += [len(s) for s in ids]
    flat = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int32)
    ends = np.cumsum(lengths).tolist()
    return [flat[e - n:e] for e, n in zip(ends, lengths)]


def answer_ids(tok):
    ids = [tok.encode(a, add_special_tokens=False) for a in ANSWERS]
    assert all(len(i) == 1 for i in ids), ids
    return ids[0][0], ids[1][0]


def hidden_at(causal, input_ids, attention_mask, rows, cols):
    """Final-norm hidden states at the tokens (rows[i], cols[i]) of a right-padded batch."""
    h = causal.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
    return h[rows, cols]


def pad_batch(seqs, pad_id, dev):
    import torch
    n = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), n), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), n), dtype=torch.long)
    for r, s in enumerate(seqs):
        ids[r, :len(s)] = torch.from_numpy(s)
        mask[r, :len(s)] = 1
    lengths = torch.tensor([len(s) for s in seqs])
    return ids.to(dev), mask.to(dev), lengths.to(dev)


def torch_margins(causal, tok, prompts, advance=None, token_budget=24576, max_rows=128):
    """Margin per prompt. While each prompt's tokens extend the previous prompt's (a conversation's prompts in order,
    see chat.WINDOW_MIN), they form one window: one forward pass over its last prompt, with each prompt's margin read
    at its own last token, which causal attention lets see only that prompt. Windows are batched by length up to
    token_budget padded tokens. advance(n) reports progress in prompts."""
    import numpy as np
    import torch
    ban, ok = answer_ids(tok)
    dev = causal.lm_head.weight.device
    enc = encode(tok, prompts)
    windows = []   # [tokens of the last prompt, indices of its prompts]
    for i, e in enumerate(enc):
        prev = windows[-1][0] if windows else None
        if prev is not None and len(e) > len(prev) and np.array_equal(e[:len(prev)], prev):
            windows[-1][0] = e
            windows[-1][1].append(i)
        else:
            windows.append([e, [i]])
    order = sorted(range(len(windows)), key=lambda w: len(windows[w][0]))
    margins = [0.0] * len(enc)
    with torch.inference_mode():
        # The softmax normalizer cancels in the margin, so only the two answer rows of the output layer are needed.
        # They run in fp32: bf16 logits are spaced 0.125 apart near |logit| 20, which would quantize the margin.
        head = causal.lm_head.weight[[ban, ok]].float()
        pos = 0
        while pos < len(order):
            n = 1
            while (pos + n < len(order) and n < max_rows
                   and (n + 1) * len(windows[order[pos + n]][0]) <= token_budget):
                n += 1
            batch = [windows[w] for w in order[pos:pos + n]]
            ids, mask, _ = pad_batch([w[0] for w in batch], tok.pad_token_id, dev)
            which = [i for _, idx in batch for i in idx]
            rows = torch.tensor([r for r, (_, idx) in enumerate(batch) for _ in idx], device=dev)
            cols = torch.tensor([len(enc[i]) - 1 for i in which], device=dev)
            with torch.autocast(dev.type, dtype=torch.bfloat16):
                h = hidden_at(causal, ids, mask, rows, cols)
            logits = h.float() @ head.T
            for i, m in zip(which, (logits[:, 0] - logits[:, 1]).tolist()):
                margins[i] = m
            pos += n
            if advance:
                advance(len(which))
    return margins
