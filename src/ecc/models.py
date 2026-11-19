"""The classifier in torch, and scoring it in fp32.

The classifier is XLM-RoBERTa-large fine-tuned as a sequence classifier with eight outputs, a HuggingFace model folder.
Inputs come from `ecc.inputs`.
"""
import functools
import subprocess
from types import SimpleNamespace

import numpy as np

from .inputs import OUTS, PAD


@functools.cache
def torch_modules():
    """The torch modules, defined once per process: `Logits`."""
    from torch import nn

    class Logits(nn.Module):
        """The classifier's logits alone, the graph's single output."""

        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask):
            return self.m(input_ids=input_ids, attention_mask=attention_mask).logits

    return SimpleNamespace(Logits=Logits)


def load_classifier(folder, device="cpu"):
    """A trained classifier from its folder, in eval mode, fp32."""
    import torch
    from transformers import XLMRobertaForSequenceClassification
    model = XLMRobertaForSequenceClassification.from_pretrained(folder, dtype=torch.float32,
                                                                attn_implementation="sdpa")
    if model.config.num_labels != len(OUTS):
        raise SystemExit(f"{folder}: {model.config.num_labels} outputs, not {len(OUTS)}")
    return model.to(device).eval()


def classifier_logits(model):
    """The classifier wrapped to return its logits tensor alone."""
    return torch_modules().Logits(model).eval()


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


def score_classifier(model, ids_list, device, tokens_per_batch=32768):
    """The classifier's logits [rows, 8] for each input id array, in length-sorted padded batches."""
    import torch
    z = np.zeros((len(ids_list), len(OUTS)), np.float32)
    order = np.argsort([len(x) for x in ids_list], kind="stable")
    with torch.inference_mode():
        i = 0
        while i < len(order):
            n = max(1, min(256, tokens_per_batch // len(ids_list[order[min(len(order) - 1, i + 255)]])))
            idx = order[i:i + n]
            ids, mask = (torch.from_numpy(x).to(device) for x in pad([ids_list[k] for k in idx]))
            z[idx] = model(input_ids=ids, attention_mask=mask).logits.float().cpu().numpy()
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
