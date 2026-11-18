"""Train a research-only kind detector. Do not use its untrained hold output for decisions.

python -m ecc.baseline --out PATH [--tokens 256] [--smoke] [--resume]
The final test period is never scored. Each epoch has a verified model and optimizer checkpoint.
"""
import os

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import uuid  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import inputs as I, trainset as T  # noqa: E402
from .base import path as base_path, pinned, sha256  # noqa: E402
from .core import is_current, log, run, source_hash, write_text  # noqa: E402
from .train import fit, frame, score, settings  # noqa: E402


def prepare(cfg, tokens, smoke):
    inp = cfg.work / "notes/takeover-audit/rebuilt-inputs"
    if not is_current(inp / "inputs.json", T.made(cfg)):
        raise SystemExit(f"{inp} does not match current input builders")
    lines = pd.read_parquet(inp / "lines.parquet")
    if lines.duplicated(["dossier", "id"]).any():
        raise SystemExit("duplicate input line keys")
    y = lines[I.KINDS].to_numpy(np.float32)
    known = y >= 0
    eligible = (lines.split.to_numpy() == "train") & ~lines.kmask.to_numpy() & known.any(1)
    zero = (y == 0).all(1)
    draw = np.random.default_rng(20261116).random(len(lines)) < 0.10
    rows = np.flatnonzero(eligible & (~zero | draw))
    val = np.flatnonzero((lines.split.to_numpy() == "val") & (lines.paper_row.to_numpy() >= 0))
    if smoke:
        pos = rows[(y[rows] == 1).any(1)]
        neg = rows[zero[rows]]
        rng = np.random.default_rng(20261116)
        rows = np.sort(np.concatenate([rng.choice(pos, 32, replace=False), rng.choice(neg, 32, replace=False)]))
        val = val[:128]
    ids, off = T.ragged(inp / f"c-{tokens}.npy")
    if len(off) != len(lines) + 1 or off[-1] != len(ids) or (np.diff(off) < 1).any():
        raise SystemExit("input token offsets do not cover the input rows")
    if (np.diff(off) > tokens).any():
        raise SystemExit("input token length exceeds the selected context")

    def take(idx):
        return [np.array(ids[off[k]:off[k + 1]]) for k in idx]

    Y = np.zeros((len(rows), len(I.OUTS)), np.float32)
    W = np.zeros_like(Y)
    for j, kind in enumerate(I.KINDS):
        o = I.OUTS.index(kind)
        Y[:, o] = np.maximum(y[rows, j], 0)
        W[:, o] = known[rows, j]
    if not (W.sum(1) > 0).all() or W[:, I.HOLD].any():
        raise SystemExit("invalid kind-only loss weights")
    counts = {"train": len(rows), "validation": len(val), "train_positive": int((y[rows] == 1).any(1).sum()),
              "train_zero": int(zero[rows].sum()), "train_partial_unknown": int((~known[rows]).any(1).sum()),
              "kind_positive": dict(zip(I.KINDS, (y[rows] == 1).sum(0).tolist())),
              "validation_kind_positive": dict(zip(I.KINDS, ((y[val] == 1) & ~lines.kmask.to_numpy()[val, None]).sum(0).tolist()))}
    request = {"inputs": json.loads((inp / "inputs.json").read_text(encoding="utf-8")),
               "row_sha256": hashlib.sha256(rows.astype("<i8").tobytes()).hexdigest(),
               "tokens": tokens, "smoke": smoke, "selection_seed": 20261116, "zero_share": 0.10,
               "selection": "all nonzero or partially unknown rows; seeded 10% of all-zero rows",
               "unknown": "mask each output", "hold": "no loss; never use this output", "counts": counts}
    return take(rows), Y, W, take(val), lines, val, request


def measurements(lines, val, z):
    y = lines[I.KINDS].to_numpy()[val]
    unmasked = ~lines.kmask.to_numpy()[val]
    kinds = z[:, [I.OUTS.index(k) for k in I.KINDS]]
    union = kinds.max(1)
    result = {"rows": len(val), "scope": "hindsight kind labels from prefix inputs; not policy or ban recall", "shares": {}}
    for share in (0.005, 0.01, 0.02, 0.05, 0.10):
        threshold = float(np.sort(union)[::-1][max(1, int(np.ceil(share * len(val)))) - 1])
        flagged = union >= threshold
        metrics = {}
        for j, kind in enumerate(I.KINDS):
            positive = (y[:, j] == 1) & unmasked
            metrics[kind] = {"caught": int((positive & flagged).sum()), "positive": int(positive.sum())}
        positive = (y == 1).any(1) & unmasked
        metrics["any"] = {"caught": int((positive & flagged).sum()), "positive": int(positive.sum())}
        zero = (y == 0).all(1) & unmasked
        result["shares"][str(share)] = {"threshold": threshold, "flagged": int(flagged.sum()),
                                         "known_zero_flagged": int((zero & flagged).sum()), "recall": metrics}
    return result


def body(cfg, args):
    import torch
    out = Path(args.out).resolve()
    if not out.is_relative_to(cfg.work.resolve()):
        raise SystemExit("keep research models in the configured work folder")
    if shutil.disk_usage(cfg.work).free < 30 * 2**30:
        raise SystemExit("keep at least 30 GiB free for three models, optimizer states and save buffers")
    base = base_path(cfg, "c")
    X, Y, W, Xv, lines, val, request = prepare(cfg, args.tokens, args.smoke)
    s = settings(cfg)
    s["epochs"] = 2 if args.smoke else 3
    request.update({"settings": s, "base": pinned(cfg, "c")[:2],
                    "code": source_hash("baseline", "train", "models", "inputs")})
    request = json.loads(json.dumps(request))
    if out.exists():
        if not args.resume:
            raise SystemExit(f"{out} exists; use --resume to verify and continue it")
        if json.loads((out / "request.json").read_text(encoding="utf-8")) != request:
            raise SystemExit("research run belongs to other inputs, settings or source")
    else:
        out.mkdir(parents=True)
        write_text(out / "request.json", json.dumps(request, indent=2))
    log.info("preflight passed: %s", json.dumps(request["counts"]))
    slots = sorted(p.parent for p in out.glob("epoch-*/complete.json"))
    for slot in slots:
        record = json.loads((slot / "complete.json").read_text())
        if record["request"] != request:
            raise SystemExit(f"{slot}: checkpoint belongs to another request")
        for name, digest in record["files"].items():
            if sha256(slot / name) != digest:
                raise SystemExit(f"{slot / name}: checkpoint changed")

    def checkpoint(model, rec, state):
        final = out / f"epoch-{state['next_epoch']}"
        if final.exists():
            raise SystemExit(f"preserve existing completed checkpoint {final}")
        slot = out / f".epoch-{state['next_epoch']}-{uuid.uuid4().hex}.part"
        slot.mkdir()
        model.save_pretrained(slot)
        shutil.copy(base / "tokenizer.json", slot / "tokenizer.json")
        torch.save(state, slot / "optimizer.pt")
        write_text(slot / "input.json", json.dumps({"model": "C", "run": f"research kinds baseline epoch {state['next_epoch']}",
                   "max_tokens": args.tokens, "research": True, "smoke": True, "outputs": I.OUTS,
                   "hold": "untrained; never use", "request": request}, indent=2))
        files = {name: sha256(slot / name) for name in ("config.json", "model.safetensors", "tokenizer.json", "optimizer.pt", "input.json")}
        write_text(slot / "complete.json", json.dumps({"request": request, "files": files}, indent=2))
        os.replace(slot, final)
        log.info("saved epoch %d model, optimizer and random state", state["next_epoch"])

    model, rec = fit(X, Y, W, base, s, checkpoint=checkpoint, resume=slots[-1] if slots else None)
    del model, X, Y, W
    torch.cuda.empty_cache()
    from .models import load_c
    results = {}
    for slot in sorted(p.parent for p in out.glob("epoch-*/complete.json")):
        model = load_c(slot, torch.device("cuda"))
        z = score(model, Xv)
        frame(lines, val, z).to_parquet(slot / "validation.parquet", index=False)
        results[slot.name] = measurements(lines, val, z)
        log.info("%s validation: %s", slot.name, json.dumps(results[slot.name]))
        del model
        torch.cuda.empty_cache()
    write_text(out / "results.json", json.dumps({"research": True, "smoke": args.smoke, "epochs": results, "train": rec}, indent=2))
    log.info("baseline complete: %s; test period untouched", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, choices=(128, 256), default=256)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    run("baseline", body, ap)


if __name__ == "__main__":
    main()
