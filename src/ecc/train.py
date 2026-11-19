"""uv run train classifier --tokens N: the classifier, trained on the inputs `uv run trainset` writes.

The classifier is XLM-RoBERTa-large ([train.classifier] base, the pinned copy in data/) with a fresh head of eight
sigmoid outputs: the seven categories and hold. A category at -1 is out of that output's loss; a line with
category_mask set is out of the category loss; hold is 0 or 1, and a line at hold -1 is out of the hold loss. The loss
is the binary cross-entropy summed over the outputs each line takes part in, divided by their number. AdamW (lr,
weight decay on every parameter, fused), a linear warm-up over the [train.classifier] warmup share of the steps then a
linear decay to 0, gradients clipped, bf16 autocast over fp32 weights with TF32 matmuls, a fixed number of epochs and
one seed ([train.classifier]). Batches: the rows shuffled, cut in chunks of 64 batches, each chunk sorted by length,
the batch order shuffled.

The training lines are the rows of lines.parquet that <[train] out>/training-lines.parquet names (transcript, id), in
training row order: the owner's lines with their hold label (each [train] owner batch's hold.parquet, which
`uv run labels` writes), then the categories-only lines (Creayun) at hold -1. The classifier reads each line's ids cut
to N tokens (classifier-<N>.npy); N must be the coverage length the inputs were built at. Rejected targets train no
loss.

After the last epoch the classifier scores, in fp32, the owner's validation and test lines and the categories-only
test_creayun lines. The run records GPU use and warns when a step window slows. GPU use alone does not establish
memory spill. Non-finite losses or gradients stop training before another optimizer update.

Writes <[train] out>/classifier-<N>[-smoke]/: the model (config.json, model.safetensors, tokenizer.json),
scores.parquet (transcript, id, period, coverage, classifier_<out>), creayun.parquet (transcript, id, coverage,
classifier_<out>), train.json (the steps, each epoch's loss, time and memory, GPU readings, the validation hold metric)
and last input.json (what export and the bundle read: run, max_tokens, the base revision, the settings and what the
run was made from).

The trainer saves verified model, optimizer, scheduler and random state after every epoch. The same command resumes
an interrupted run only when its inputs, settings and source match. Incomplete checkpoint files remain beside the
completed epochs. A finished run whose scoring stopped resumes from trained.json without training again.
"""
import os

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from . import inputs as I  # noqa: E402
from .core import fingerprint, is_current, log, run, write_text  # noqa: E402

SMOKE_SCORED = 2000     # a smoke run scores the first this many lines of each scored set
WATCH_BASE = (1000, 2500)   # the steps whose 500-step windows set the base step time
WATCH_SLOW = 1.4
HOLDS = (0.005, 0.01, 0.02)


def batches(lengths, batch, rng):
    """The epoch's batches: rows shuffled, chunks of 64 batches sorted by length, the batch order shuffled."""
    idx = rng.permutation(len(lengths))
    out = []
    for c in range(0, len(idx), batch * 64):
        chunk = sorted(idx[c:c + batch * 64], key=lambda i: lengths[i])
        out += [chunk[k:k + batch] for k in range(0, len(chunk), batch)]
    return [out[i] for i in rng.permutation(len(out))]


def hold_metric(score, hold):
    """Mean recall of hold = 1 lines at the scores that hold 0.5, 1 and 2% of the lines given."""
    s = np.sort(np.asarray(score, np.float64))[::-1]
    pos = hold == 1
    return float(np.mean([(score[pos] >= s[max(1, int(np.ceil(h * len(s)))) - 1]).mean() for h in HOLDS]))


def gpu_reading():
    """(used MiB, total MiB, utilisation %) from nvidia-smi."""
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout
    return tuple(int(float(x)) for x in out.split("\n")[0].split(","))


def memory_report():
    import torch
    used, total, util = gpu_reading()
    return (f"peak reserved {torch.cuda.max_memory_reserved() / 2**30:.1f} GB; nvidia-smi {used / 1024:.1f} of "
            f"{total / 1024:.1f} GB, util {util}%"), util


def pad(seqs, device):
    """A padded batch on the device and its attention mask (bool), through pinned memory."""
    import torch
    from .models import pad as padded
    return tuple(torch.from_numpy(x).pin_memory().to(device, non_blocking=True) for x in padded(seqs))


def fit(X, Y, W, base, s, checkpoint=None, resume=None):
    """The classifier trained on the id arrays X with targets Y and loss weights W ([rows, 8] float32 each) from the base model folder
    `base`, with the settings s (seed, epochs, lr, batch, warmup, clip, weight_decay). Returns the model, in train mode
    on the GPU, and the training record."""
    import torch
    import torch.nn.functional as F
    from transformers import XLMRobertaForSequenceClassification

    from .models import limit_gpu_memory
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(s["seed"])
    rng = np.random.default_rng(s["seed"])
    torch.backends.cuda.matmul.fp32_precision = "tf32"
    dev = torch.device("cuda")
    limit_gpu_memory(log)
    Y, W = torch.as_tensor(Y), torch.as_tensor(W)
    if not (W.sum(1) > 0).all():
        raise SystemExit("a training line takes part in no loss")
    model = XLMRobertaForSequenceClassification.from_pretrained(
        resume if resume is not None else base,
        num_labels=len(I.OUTS), id2label=dict(enumerate(I.OUTS)), label2id={k: i for i, k in enumerate(I.OUTS)},
        problem_type="multi_label_classification", ignore_mismatched_sizes=True, dtype=torch.float32,
        attn_implementation="sdpa")
    if resume is None:
        for lin in (model.classifier.dense, model.classifier.out_proj):
            torch.nn.init.normal_(lin.weight, std=model.config.initializer_range)
            torch.nn.init.zeros_(lin.bias)
    model.to(dev).train()
    opt = torch.optim.AdamW(model.parameters(), lr=s["lr"], weight_decay=s["weight_decay"], fused=True)
    steps = s["epochs"] * math.ceil(len(X) / s["batch"])
    warm = int(s["warmup"] * steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda k: k / max(1, warm) if k < warm else max(0.0, (steps - k) / max(1, steps - warm)))
    rec = {"steps_planned": steps, "warmup_steps": warm, "epochs": [], "gpu": []}
    lengths = [len(x) for x in X]
    step, first_epoch, base_ms = 0, 0, []
    if resume is not None:
        saved = torch.load(resume / "optimizer.pt", map_location="cpu", weights_only=True)
        if saved["steps_planned"] != steps:
            raise SystemExit("saved optimizer belongs to another training schedule")
        opt.load_state_dict(saved["optimizer"])
        sched.load_state_dict(saved["scheduler"])
        rng.bit_generator.state = saved["numpy_rng"]
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
        step, first_epoch, rec = saved["step"], saved["next_epoch"], saved["record"]
        del saved
    torch.cuda.reset_peak_memory_stats()
    t_start = time.perf_counter()
    for ep in range(first_epoch, s["epochs"]):
        t0, losses = time.perf_counter(), []
        t_win, s_win = t0, step
        for idx in batches(lengths, s["batch"], rng):
            ids, mask = pad([X[j] for j in idx], dev)
            y = Y[idx].to(dev, non_blocking=True)
            w = W[idx].to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(input_ids=ids, attention_mask=mask).logits
            loss = (F.binary_cross_entropy_with_logits(lg.float(), y, reduction="none") * w).sum() / w.sum().clamp_min(1)
            if not torch.isfinite(loss).item():
                raise SystemExit(f"non-finite training loss at step {step + 1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), s["clip"], error_if_nonfinite=True)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            losses.append(loss.detach())
            step += 1
            if step % 500 == 0 or step == 50:
                now = time.perf_counter()
                ms = (now - t0) / len(losses) * 1000
                ms_win = (now - t_win) / (step - s_win) * 1000
                t_win, s_win = now, step
                rep, util = memory_report()
                recent = torch.stack(losses[-500:]).mean().item()
                rec["gpu"].append({"step": step, "loss": recent, "lr": sched.get_last_lr()[0],
                                   "ms_per_step": round(ms), "ms_window": round(ms_win), "report": rep})
                log.info("epoch %d step %d/%d loss %.4f lr %.2e, %.0f ms/step (last window %.0f); %s", ep + 1, step,
                         steps, recent, sched.get_last_lr()[0], ms, ms_win, rep)
                if WATCH_BASE[0] <= step <= WATCH_BASE[1] and step % 500 == 0:
                    base_ms.append(ms_win)
                slow = len(base_ms) == 4 and step > WATCH_BASE[1] and ms_win > WATCH_SLOW * float(np.median(base_ms))
                if slow:
                    log.warning("slow training window at step %d: %.0f ms/step, baseline %.0f; GPU util %d%%",
                                step, ms_win, float(np.median(base_ms)), util)
        train_s = time.perf_counter() - t0
        rep, _ = memory_report()
        e = {"epoch": ep + 1, "train_loss": float(torch.stack(losses).mean().item()), "train_s": round(train_s),
             "ms_per_step": round(train_s / len(losses) * 1000), "lr_end": sched.get_last_lr()[0],
             "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 1), "report": rep}
        log.info("epoch %d: %s", ep + 1, json.dumps(e))
        rec["epochs"].append(e)
        if checkpoint is not None:
            checkpoint(model, rec, {"optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                       "numpy_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                       "cuda_rng": torch.cuda.get_rng_state_all(), "step": step, "next_epoch": ep + 1,
                       "record": rec, "steps_planned": steps})
    rec["train_s"] = sum(e["train_s"] for e in rec["epochs"])
    return model, rec


# ---------------------------------------------------------------- the run's data

def settings(cfg):
    t = cfg["train"]["classifier"]
    return {k: t[k] for k in ("seed", "epochs", "lr", "batch", "warmup", "clip", "weight_decay")}


def picked(cfg, lines):
    """Whether each line of lines.parquet is a training line: named in training-lines.parquet, which must be current with the
    inputs, the ranking and the hold labels. Refuses training lines that name a line twice, a line outside the
    inputs or the train split, or a rejected line."""
    import pandas as pd

    from . import trainset
    p = trainset.training_lines_path(cfg)
    if not is_current(p, trainset.made_training_lines(cfg)):
        raise SystemExit(f"{p} is missing or was made from other inputs, rankings or hold labels; run "
                         f"`uv run trainset`")
    chosen = pd.read_parquet(p, columns=["transcript", "id"])
    if chosen.duplicated().any():
        raise SystemExit(f"{p} names a line twice")
    m = lines[["transcript", "id", "split"]].reset_index().merge(chosen, on=["transcript", "id"], validate="1:1")
    if len(m) != len(chosen):
        raise SystemExit(f"{len(chosen) - len(m):,} lines of {p} are not in the inputs; run `uv run trainset`")
    if (m.split != "train").any():
        raise SystemExit(f"{p} names {int((m.split != 'train').sum()):,} lines outside the train split")
    mask = np.zeros(len(lines), bool)
    mask[m["index"].to_numpy()] = True
    if (mask & (lines.coverage.to_numpy() == "rejected")).any():
        raise SystemExit(f"{p} selects a rejected target; run `uv run trainset --force`")
    return mask, fingerprint(p)


def data(cfg, tokens, limit):
    """Everything the run reads: training rows, targets, weights, the scored rows and their ids, what it is made from."""
    import pandas as pd

    from . import trainset
    d = trainset.folder(cfg)
    made = trainset.made(cfg)
    if not is_current(d / "inputs.json", made):
        raise SystemExit(f"{d} is not current with the tables and settings; run `uv run trainset`")
    made = json.loads((d / "inputs.json").read_text(encoding="utf-8"))
    if tokens != made.get("coverage_tokens"):
        raise SystemExit("train the classifier at the input set's coverage_tokens; rebuild the inputs to select "
                         "another serving length")
    if tokens not in made["tokens"]:
        raise SystemExit(f"the inputs hold the classifier's ids at {made['tokens']} tokens, not {tokens}")
    lines = pd.read_parquet(d / "lines.parquet")
    if "coverage" not in lines or not lines.coverage.isin(["scored", "rejected"]).all():
        raise SystemExit("training inputs need valid coverage routes")
    if lines.duplicated(["transcript", "id"]).any():
        raise SystemExit(f"{d / 'lines.parquet'} names a line twice")
    logged = lines.server_log_row.to_numpy() >= 0
    train, chosen_fp = picked(cfg, lines)
    hold = trainset.hold_labels(cfg, lines)
    split = lines.split.to_numpy()
    rows, crows = np.flatnonzero(train & logged), np.flatnonzero(train & ~logged)
    va, te = np.flatnonzero(logged & (split == "val")), np.flatnonzero(logged & (split == "test"))
    ct = np.flatnonzero(~logged & (split == "test_creayun"))
    if limit:
        r0 = np.random.default_rng(0)
        count = min(limit, len(rows) + len(crows))
        nc = min(len(crows), count // 2)
        nr = min(len(rows), count - nc)
        nc = min(len(crows), count - nr)
        rows = np.sort(r0.choice(rows, nr, replace=False))
        crows = np.sort(r0.choice(crows, nc, replace=False))
        va, te, ct = va[:SMOKE_SCORED], te[:SMOKE_SCORED], ct[:SMOKE_SCORED]
    rows = np.concatenate([rows, crows])
    yk = lines[I.CATEGORIES].to_numpy(np.float32)[rows]
    unknown = (yk < 0).sum(0)
    wk = (yk >= 0) & ~lines.category_mask.to_numpy()[rows, None]
    yk = np.maximum(yk, 0)
    yh = hold[rows]
    Y = np.concatenate([yk, (yh == 1)[:, None]], 1).astype(np.float32)
    W = np.concatenate([wk, (yh != -1)[:, None]], 1).astype(np.float32)
    ids, off = trainset.ragged(d / f"classifier-{tokens}.npy")
    if len(off) != len(lines) + 1:
        raise SystemExit(f"classifier-{tokens}.npy holds {len(off) - 1:,} lines, lines.parquet {len(lines):,}")

    def take(r):
        return [np.array(ids[off[k]:off[k + 1]]) for k in r]

    counts = {"train": len(rows), "train_owner": len(rows) - len(crows), "train_categories_only": len(crows),
              "train_hold1": int((yh == 1).sum()), "val": len(va), "val_hold1": int((hold[va] == 1).sum()),
              "test": len(te), "test_creayun": len(ct),
              "category_unknown_masked": dict(zip(I.CATEGORIES, unknown.tolist())),
              "category_weighted": dict(zip(I.CATEGORIES, wk.sum(0).tolist()))}
    made_from = {"inputs": made, "hold": {b: fingerprint(trainset.hold_path(cfg, b))
                                          for b in cfg["train"]["owner"]}, "training_lines": chosen_fp}
    return {"X": take(rows), "Y": Y, "W": W, "lines": lines, "va": va, "te": te, "ct": ct,
            "Xva": take(va), "Xte": take(te), "Xct": take(ct), "vhold": hold[va], "counts": counts,
            "made_from": made_from}


def frame(lines, rows, z, period=True):
    import pandas as pd
    r = pd.DataFrame({"transcript": lines.transcript.to_numpy()[rows], "id": lines.id.to_numpy()[rows]})
    if period:
        r["period"] = lines.split.to_numpy()[rows]
    r["coverage"] = lines.coverage.to_numpy()[rows]
    for k, o in enumerate(I.OUTS):
        r[f"classifier_{o}"] = z[:, k]
    return r


def score(model, seqs):
    """The classifier's fp32 logits of the id arrays (no TF32, no autocast), as export scores."""
    import torch

    from .models import fp32, score_classifier
    fp32()
    model.eval()
    z = score_classifier(model, seqs, torch.device("cuda"))
    if not np.isfinite(z).all():
        raise SystemExit("the classifier gave a non-finite logit")
    return z


def score_covered(model, seqs, lines, rows):
    """Keep every row key. Rejected targets have no model score and are blocked by the coverage rule."""
    valid = lines.coverage.to_numpy()[rows] != "rejected"
    z = np.zeros((len(rows), len(I.OUTS)), np.float32)
    if valid.any():
        z[valid] = score(model, [x for x, keep in zip(seqs, valid) if keep])
    return z


def run_name(tokens, smoke):
    """A training run's folder name: classifier-<N>[-smoke]."""
    return f"classifier-{tokens}" + ("-smoke" if smoke else "")


def checkpoints(out, request):
    """Verified completed epochs, including optimizer and random state, in epoch order."""
    from .base import sha256
    slots = sorted((p.parent for p in out.glob("epoch-*/complete.json")),
                   key=lambda p: int(p.name.split("-")[1]))
    for expected_epoch, slot in enumerate(slots, 1):
        record = json.loads((slot / "complete.json").read_text(encoding="utf-8"))
        if slot.name != f"epoch-{expected_epoch}" or record.get("request") != request:
            raise SystemExit(f"{slot}: checkpoint epochs are incomplete or belong to another request")
        expected = {"config.json", "model.safetensors", "tokenizer.json", "optimizer.pt"}
        if set(record.get("files", {})) != expected:
            raise SystemExit(f"{slot}: checkpoint file list is incomplete")
        for name, digest in record["files"].items():
            if not (slot / name).is_file() or sha256(slot / name) != digest:
                raise SystemExit(f"{slot / name}: checkpoint changed")
    return slots


def train_classifier(cfg, args):
    from .base import path as base_path
    from .base import pinned
    tokens, limit = args.tokens, args.limit
    s = settings(cfg)
    if args.epochs:
        s["epochs"] = args.epochs
    name = run_name(tokens, bool(limit or args.epochs))
    out = cfg.path(cfg["train"]["out"]) / name
    if (out / "input.json").exists() and not args.force:
        raise SystemExit(f"{out} holds a finished run; --force trains it again")
    base = base_path(cfg)
    t0 = time.perf_counter()
    D = data(cfg, tokens, limit)
    c = D["counts"]
    log.info("%s: train %s lines = owner's %s (hold = 1 %s) + categories-only %s; val %s (hold = 1 %s), test %s, "
             "test_creayun %s; %d tokens (%.0f s)", name, f"{c['train']:,}", f"{c['train_owner']:,}",
             f"{c['train_hold1']:,}", f"{c['train_categories_only']:,}", f"{c['val']:,}", f"{c['val_hold1']:,}",
             f"{c['test']:,}", f"{c['test_creayun']:,}", tokens, time.perf_counter() - t0)
    import torch

    from .base import sha256
    from .core import source_hash
    from .models import limit_gpu_memory, load_classifier
    repo, rev, _ = pinned(cfg)
    request = json.loads(json.dumps({
        "tokens": tokens, "limit": limit, "smoke": bool(limit or args.epochs), "settings": s,
        "base": {"repo": repo, "revision": rev}, "made_from": D["made_from"],
        "code": source_hash("train", "models", "inputs", "core"),
    }))
    mark = out / "trained.json"
    files = ("config.json", "model.safetensors", "tokenizer.json")
    if mark.exists() and not args.force:
        kept = json.loads(mark.read_text(encoding="utf-8"))
        if kept.get("request") != request:
            raise SystemExit(f"{out}: saved training uses other inputs, settings or code; use another output or --force")
        for file_name in files:
            p = out / file_name
            if not p.is_file() or kept.get("files", {}).get(file_name) != {"size": p.stat().st_size, "sha256": sha256(p)}:
                raise SystemExit(f"{p}: saved training file is missing or changed; preserve it or use --force")
        log.info("%s: finish scoring from verified saved training; do not train again", out)
        for field in ("X", "Y", "W"):
            del D[field]
        limit_gpu_memory(log)
        model, rec = load_classifier(out, torch.device("cuda")), kept["record"]
    else:
        if out.exists() and args.force:
            # --force explicitly permits replacing this run's output folder.
            if out.resolve().parent != cfg.path(cfg["train"]["out"]).resolve():
                raise SystemExit(f"refuse to replace a run outside [train] out: {out}")
            shutil.rmtree(out)
        request_path = out / "request.json"
        if out.exists():
            if not request_path.is_file() or json.loads(request_path.read_text(encoding="utf-8")) != request:
                raise SystemExit(f"{out}: unfinished run has other or unrecorded inputs; preserve it or use --force")
        else:
            out.mkdir(parents=True)
            write_text(request_path, json.dumps(request, indent=1) + "\n")
        slots = checkpoints(out, request)

        def checkpoint(model, record, state):
            slot = out / f"epoch-{state['next_epoch']}"
            if slot.exists():
                if slot.resolve().parent != out.resolve() or (slot / "complete.json").exists():
                    raise SystemExit(f"refuse to replace completed or redirected checkpoint {slot}")
                preserved = slot.with_name(slot.name + f"-incomplete-{time.time_ns()}")
                slot.rename(preserved)
                log.warning("preserved interrupted checkpoint in %s", preserved)
            slot.mkdir()
            model.save_pretrained(slot)
            shutil.copy(base / "tokenizer.json", slot / "tokenizer.json")
            torch.save(state, slot / "optimizer.pt")
            hashes = {n: sha256(slot / n) for n in (*files, "optimizer.pt")}
            write_text(slot / "complete.json", json.dumps({"request": request, "files": hashes}, indent=1) + "\n")
            log.info("saved verified epoch %d with optimizer and random state", state["next_epoch"])

        model, rec = fit(D.pop("X"), D.pop("Y"), D.pop("W"), base, s,
                         checkpoint=checkpoint, resume=slots[-1] if slots else None)
        rec["peak_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 2**30, 1)
        model.save_pretrained(out)
        shutil.copy(base / "tokenizer.json", out / "tokenizer.json")
        saved = {name: {"size": (out / name).stat().st_size, "sha256": sha256(out / name)} for name in files}
        write_text(mark, json.dumps({"request": request, "files": saved, "record": rec}, indent=1) + "\n")
    t1 = time.perf_counter()
    lines = D["lines"]
    zv, zt, zc = (score_covered(model, D["X" + key], lines, D[key]) for key in ("va", "te", "ct"))
    rec["score_s"] = round(time.perf_counter() - t1)
    import pandas as pd
    pd.concat([frame(lines, D["va"], zv), frame(lines, D["te"], zt)]).to_parquet(out / "scores.parquet", index=False)
    frame(lines, D["ct"], zc, period=False).to_parquet(out / "creayun.parquet", index=False)
    covered = lines.coverage.to_numpy()[D["va"]] != "rejected"
    rec["val_metric"] = hold_metric(zv[covered, I.HOLD], D["vhold"][covered]) if (D["vhold"][covered] == 1).any() else None
    log.info("scored %s validation, %s test and %s test_creayun lines in fp32 (%d s); validation hold metric %s",
             f"{len(zv):,}", f"{len(zt):,}", f"{len(zc):,}", rec["score_s"], rec["val_metric"])
    write_text(out / "train.json", json.dumps({"settings": s, "counts": c, **rec}, indent=1) + "\n")
    what = f"smoke run on {limit:,} lines, " if limit else ""
    write_text(out / "input.json", json.dumps({
        "model": "classifier",
        "run": f"{name}: the classifier trained at {tokens} tokens, {what}seed {s['seed']}, epoch {s['epochs']} of {s['epochs']}",
        "base": {"repo": repo, "revision": rev}, "max_tokens": tokens, "trained_tokens": tokens,
        "coverage_tokens": D["made_from"]["inputs"]["coverage_tokens"],
        "ctx_lines": I.CTX_LINES, "ctx_seconds": I.CTX_SECONDS, "seed": s["seed"], "epoch": s["epochs"],
        "outputs": I.OUTS, "limit": limit or None, "smoke": bool(limit or args.epochs), "settings": s, "counts": c,
        "labels": "the owner's lines: hold.parquet of each [train] owner batch; categories-only lines teach the categories only "
                  "(hold -1)",
        "made_from": D["made_from"]}, indent=1) + "\n")
    log.info("%s written (%.1f min)", out, (time.perf_counter() - t0) / 60)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="model", required=True)
    c = sub.add_parser("classifier", help="train the classifier")
    c.add_argument("--tokens", type=int, required=True, help="the classifier's length: its input cut to this many "
                                                             "tokens")
    c.add_argument("--epochs", type=int, default=0, help="a smoke run of this many epochs")
    c.add_argument("--limit", type=int, default=0,
                   help=f"a smoke run on this many training lines (half owner's, half categories-only, drawn with seed 0), "
                        f"scoring the first {SMOKE_SCORED} of each scored set")
    c.add_argument("--force", action="store_true", help="train again over a finished run of the same name")

    def body(cfg, args):
        train_classifier(cfg, args)

    run("train", body, ap)


if __name__ == "__main__":
    main()
