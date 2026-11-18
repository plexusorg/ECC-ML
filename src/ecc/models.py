"""The two v4 models in torch, and scoring them in fp32.

- SH, the student: every line encoded alone by multilingual e5-small (mean pooling, LayerNorm) into one 384-float
  vector (`LineEncoder`); a small transformer (`Context`) reads the vectors of the line's context and the line, each
  plus learned embeddings of its speaker slot, addressee slot, channel, time gap and position, and gives the eight
  logits at the line's position. A model folder holds model.safetensors, meta.json (`layers`) and e5-small's
  config.json and tokenizer.json.
- C: XLM-RoBERTa-large fine-tuned as a sequence classifier with eight outputs, a HuggingFace model folder.

SH has one definition: training builds it with `new_sh` and writes it with `save_sh`, export and the bundle load it with
`load_sh`. Inputs come from `ecc.inputs`.
"""
import functools
import json
import math
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np

from .inputs import ADDR_NONE, CHANNELS, GAPS, OUTS, PAD, SEQ

D = 384     # e5-small's width, SH's vector


@functools.cache
def torch_modules():
    """The torch modules, defined once per process: SH's `LineEncoder`, `Block`, `Context` and `SH` (the two as
    `encoder` and `context`), and `Logits` for C."""
    import torch
    import torch.nn.functional as F
    from torch import nn

    def pool(h, mask):
        m = mask.unsqueeze(-1).to(h.dtype)
        return (h * m).sum(1) / m.sum(1).clamp(min=1)

    class LineEncoder(nn.Module):
        """One line's tokens [N, T] -> its vector [N, 384]."""

        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone
            self.norm = nn.LayerNorm(D)

        def forward(self, input_ids, attention_mask):
            # a ready 4D mask: with a 2D one transformers asks the card whether every token is used (mask.all()),
            # which makes the host wait for the card twice a step
            h = self.backbone(input_ids=input_ids,
                              attention_mask=attention_mask[:, None, None, :].bool()).last_hidden_state
            return self.norm(pool(h, attention_mask))

    class Block(nn.Module):
        def __init__(self, d=D, h=6, ff=1536, p=0.1):
            super().__init__()
            self.h = h
            self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
            self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
            self.f1, self.f2 = nn.Linear(d, ff), nn.Linear(ff, d)
            self.drop = nn.Dropout(p)

        def forward(self, x, bias):
            B, L, Dm = x.shape
            q, k, v = self.qkv(self.n1(x)).view(B, L, 3, self.h, Dm // self.h).permute(2, 0, 3, 1, 4)
            a = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(Dm // self.h) + bias, -1)
            x = x + self.drop(self.o((self.drop(a) @ v).transpose(1, 2).reshape(B, L, Dm)))
            return x + self.drop(self.f2(self.drop(F.gelu(self.f1(self.n2(x))))))

    class Context(nn.Module):
        """The transformer over the line vectors; the line scored is the last used position."""

        def __init__(self, layers):
            super().__init__()
            self.slot = nn.Embedding(SEQ + 1, D)
            self.addr = nn.Embedding(ADDR_NONE + 1, D)
            self.chan = nn.Embedding(len(CHANNELS), D)
            self.gap = nn.Embedding(GAPS, D)
            self.pos = nn.Embedding(SEQ, D)
            for e in (self.slot, self.addr, self.chan, self.gap, self.pos):
                nn.init.normal_(e.weight, std=0.02)
            self.blocks = nn.ModuleList([Block() for _ in range(layers)])
            self.norm = nn.LayerNorm(D)
            self.drop = nn.Dropout(0.1)
            self.head = nn.Linear(D, len(OUTS))

        def forward(self, vec, slot, addr, chan, gap, pos, mask):
            x = vec + self.slot(slot) + self.addr(addr) + self.chan(chan) + self.gap(gap) + self.pos(pos)
            last = mask.sum(1).long() - 1
            bias = ((1.0 - mask.to(x.dtype)) * -1e4)[:, None, None, :]
            for b in self.blocks:
                x = b(x, bias)
            t = x[torch.arange(x.shape[0], device=x.device), last]
            return self.head(self.drop(self.norm(t)))

    class SH(nn.Module):
        def __init__(self, backbone, layers):
            super().__init__()
            self.encoder = LineEncoder(backbone)
            self.context = Context(layers)

    class Logits(nn.Module):
        """C's logits alone, the graph's single output."""

        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask):
            return self.m(input_ids=input_ids, attention_mask=attention_mask).logits

    return SimpleNamespace(LineEncoder=LineEncoder, Block=Block, Context=Context, SH=SH, Logits=Logits)


def e5_backbone(path, pretrained):
    """e5-small without its pooling layer: its weights from `path` (pretrained), or only its configuration."""
    import torch
    from transformers import AutoConfig, AutoModel
    if pretrained:
        return AutoModel.from_pretrained(path, dtype=torch.float32, attn_implementation="sdpa", add_pooling_layer=False)
    return AutoModel.from_config(AutoConfig.from_pretrained(path), attn_implementation="sdpa", add_pooling_layer=False)


def new_sh(base, layers, prior):
    """An SH to train, in train mode: e5-small's weights from the base model folder `base`; the encoder's norm, the
    embeddings, the context blocks and the head new, drawn from torch's generator in that order; the head's bias at
    `prior`, the eight outputs' log-odds."""
    import torch
    if len(prior) != len(OUTS):
        raise SystemExit(f"{len(prior)} prior log-odds for {len(OUTS)} outputs")
    m = torch_modules().SH(e5_backbone(base, pretrained=True), layers)
    with torch.no_grad():
        m.context.head.bias.copy_(torch.tensor(prior, dtype=torch.float32))
    return m.train()


def sh_groups(sh, lr, new_lr, head_lr):
    """SH's three optimiser groups: e5-small at `lr`; the encoder's norm, the embeddings and the context blocks at
    `new_lr`; the head and the final norm at `head_lr`."""
    head = list(sh.context.head.parameters()) + list(sh.context.norm.parameters())
    hid = {id(p) for p in head}
    new = [p for n, p in sh.encoder.named_parameters() if not n.startswith("backbone.")] + \
        [p for p in sh.context.parameters() if id(p) not in hid]
    return [{"params": list(sh.encoder.backbone.parameters()), "lr": lr},
            {"params": new, "lr": new_lr}, {"params": head, "lr": head_lr}]


def save_sh(sh, folder, base, **meta):
    """SH's folder as `load_sh` reads it: model.safetensors, e5-small's config.json and tokenizer.json from the base
    model folder `base`, and meta.json with the layers, the outputs and `meta`."""
    from safetensors.torch import save_file
    folder.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in sh.state_dict().items()}, str(folder / "model.safetensors"))
    for f in ("config.json", "tokenizer.json"):
        shutil.copy(base / f, folder / f)
    (folder / "meta.json").write_text(json.dumps({"layers": len(sh.context.blocks), "outputs": OUTS, **meta}),
                                      encoding="utf-8")


def load_sh(folder, device="cpu"):
    """A trained SH from its folder, in eval mode."""
    from safetensors.torch import load_file
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    if meta["outputs"] != OUTS:
        raise SystemExit(f"{folder}: outputs {meta['outputs']} are not {OUTS}")
    m = torch_modules().SH(e5_backbone(folder, pretrained=False), meta["layers"])
    m.load_state_dict(load_file(str(folder / "model.safetensors")), strict=True)
    return m.to(device).eval()


def load_c(folder, device="cpu"):
    """A trained C from its folder, in eval mode, fp32."""
    import torch
    from transformers import XLMRobertaForSequenceClassification
    c = XLMRobertaForSequenceClassification.from_pretrained(folder, dtype=torch.float32, attn_implementation="sdpa")
    if c.config.num_labels != len(OUTS):
        raise SystemExit(f"{folder}: {c.config.num_labels} outputs, not {len(OUTS)}")
    return c.to(device).eval()


def c_logits(c):
    """C wrapped to return its logits tensor alone."""
    return torch_modules().Logits(c).eval()


def fp32():
    """Full fp32 matmuls on the GPU (no TF32), as the exported graphs compute."""
    import torch
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.fp32_precision = "ieee"


def pad(seqs):
    """Return ids and a length-based mask. Text can contain the reserved PAD token."""
    n = max(len(s) for s in seqs)
    ids = np.full((len(seqs), n), PAD, np.int64)
    mask = np.zeros_like(ids)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = s
        mask[i, :len(s)] = 1
    return ids, mask


def score_sh(sh, line_tokens, feats, device):
    """SH's logits [rows, 8] for the rows of sh_features `feats`: every line of `line_tokens` encoded once, the context
    step over the kept vectors."""
    import torch
    vecs = np.zeros((len(line_tokens), D), np.float32)
    order = np.argsort([len(x) for x in line_tokens], kind="stable")
    R = len(feats["n"])
    zs = np.zeros((R, len(OUTS)), np.float32)
    with torch.inference_mode():
        for i in range(0, len(order), 512):
            idx = order[i:i + 512]
            ids, mask = (torch.from_numpy(x).to(device) for x in pad([line_tokens[k] for k in idx]))
            vecs[idx] = sh.encoder(ids, mask).float().cpu().numpy()
        V = torch.from_numpy(vecs).to(device)
        for i in range(0, R, 512):
            P = np.arange(i, min(R, i + 512))
            L = int(feats["n"][P].max())
            lines = feats["line"][P, :L]
            used = lines >= 0
            g = torch.from_numpy(np.where(used, lines, 0)).to(device)
            mask = torch.from_numpy(used).to(device)
            f = {k: torch.from_numpy(feats[k][P, :L]).to(device) for k in ("slot", "addr", "chan", "gap", "pos")}
            vec = V[g] * mask[..., None].float()
            zs[P] = sh.context(vec, f["slot"], f["addr"], f["chan"], f["gap"], f["pos"], mask).float().cpu().numpy()
    return zs


def score_c(c, ids_list, device, tokens_per_batch=32768):
    """C's logits [rows, 8] for each input id array, in length-sorted padded batches."""
    import torch
    z = np.zeros((len(ids_list), len(OUTS)), np.float32)
    order = np.argsort([len(x) for x in ids_list], kind="stable")
    with torch.inference_mode():
        i = 0
        while i < len(order):
            n = max(1, min(256, tokens_per_batch // len(ids_list[order[min(len(order) - 1, i + 255)]])))
            idx = order[i:i + n]
            ids, mask = (torch.from_numpy(x).to(device) for x in pad([ids_list[k] for k in idx]))
            z[idx] = c(input_ids=ids, attention_mask=mask).logits.float().cpu().numpy()
            i += n
    return z


def gpu_free_mib():
    """(used, total) MiB of the first GPU, from nvidia-smi."""
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout.split("\n")[0]
    used, total = (float(x) for x in out.split(","))
    return used, total


def limit_gpu_memory(log):
    """Cap this process at what the card has free now, less 600 MiB, before anything is allocated on it, so it never
    spills into shared memory."""
    import torch
    used, total = gpu_free_mib()
    frac = (total - used - 600) / total
    if frac < 0.2:
        raise SystemExit(f"the GPU has {total - used:.0f} of {total:.0f} MiB free; wait for the job using it")
    torch.cuda.set_per_process_memory_fraction(frac)
    log.info("GPU cap %.1f GB (%.1f of %.1f GB in use at start)", frac * total / 1024, used / 1024, total / 1024)
