"""uv run train c --tokens N [--fold K] and uv run train sh --c RUN: C, the large model, and SH, the
student, trained on the inputs `uv run trainset` writes.

C is XLM-RoBERTa-large ([train.c] base, the pinned copy in data/) with a fresh head of eight sigmoid outputs: the seven
kinds and hold. A kind at -1 is out of that output's loss; a line with kmask set is out of the kind loss; hold is 0 or 1, and a
line at hold -1 is out of the hold loss. The loss is the binary cross-entropy summed over the outputs each line takes
part in, divided by their number. AdamW (lr, weight decay on every parameter, fused), a linear warm-up over the
[train.c] warmup share of the steps then a linear decay to 0, gradients clipped, bf16 autocast over fp32 weights with
TF32 matmuls, a fixed number of epochs and one seed ([train.c]). Batches: the rows shuffled, cut in chunks of 64
batches, each chunk sorted by length, the batch order shuffled.

The training lines are the rows of lines.parquet that <[train] out>/pick.parquet names (dossier, id), in training row
order: the owner's lines with their hold label (each [train] owner batch's hold.parquet, which `uv run labels` writes),
then the kinds-only lines (Creayun) at hold -1. C reads each line's ids cut to N tokens (c-<N>.npy); N must be one of
the lengths the inputs were built at. With --fold K the run is teacher fold K of [train.c] folds: the Paper lines are
split into folds by run (`line_folds`), and fold K's model trains on the owner's training lines of the other folds and
every selected kinds-only line. After training it scores, as SH's teacher, only cascade-route train-split Paper lines
of fold K (bf16, as the ranker scores), and refuses when one is of a run it trained on. Teacher folds do not score
validation or test lines. C trains at the coverage length recorded by its inputs. Rejected targets train no loss.

After the last epoch C scores, in fp32, the owner's validation and test lines and the kinds-only test_creayun lines.
The run records GPU use and warns when a step window slows. GPU use alone does not establish memory spill.
Non-finite losses or gradients stop training before another optimizer update.

Writes <[train] out>/c-<N>[-fold<K>][-smoke]/: the model (config.json, model.safetensors, tokenizer.json), scores.parquet
(dossier, id, period, c_<out>), creayun.parquet (dossier, id, c_<out>), train.json (the steps, each epoch's loss, time
and memory, GPU readings, the validation hold metric) and last input.json (what export and the bundle read: run,
max_tokens, the base revision, the settings and what the run was made from). A teacher fold also writes teacher.parquet
(dossier, id, run, c_<out> of its fold's train lines) and trained-runs.json (the Paper runs it trained on).

SH ([train.sh], the pinned e5-small copy; models.new_sh) trains on cascade-route train-split judged Paper lines, with the
trainset's SH line ids and features. Each line's teacher target is the logits of the fold model of C run RUN that did
not train on its run (c-<N>-fold<K>/teacher.parquet). The loss is the soft BCE of SH's logits / T
against sigmoid(teacher / T), times T^2, on all eight outputs, plus the hard BCE on the outputs a line takes part in
(as C's), summed and divided by the number of terms. Batches are chunks of consecutive lines of one log file, shuffled.
AdamW in three groups (e5-small, the new layers, the head and final norm), warm-up then linear decay over [train.sh]
epochs, bf16 autocast. After every epoch SH scores the validation lines in fp32; the epoch kept is the one where SH
passes, at the bundle's pass share (data/v4/bundle.toml [thresholds] pass_share), the largest share of the validation
lines RUN holds at the hold share [train.sh] epoch_hold names (C's own top share of the lines RUN scored); training stops
[train.sh] patience epochs after the best. Writes <[train] out>/sh-RUN/: SH's model folder (as `export`
loads it), scores.parquet (dossier, id, period, sh_<out> of the kept epoch, fp32), train.json and input.json.

Both trainers save verified model, optimizer, scheduler and random state after every epoch. The same command resumes
an interrupted run only when its inputs, settings and source match. Incomplete checkpoint files remain beside the
completed epochs. A finished C whose scoring stopped resumes from trained.json without training again.
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


def line_folds(lines, folds, seed):
    """The teacher fold (1..folds) of each line of lines.parquet, 0 for a kinds-only line: the runs of the Paper lines
    sorted, shuffled with `seed` and dealt round the folds, so a run is in one fold whichever batches hold it."""
    import pandas as pd
    paper = lines.paper_row.to_numpy() >= 0
    names = np.array(sorted(set(lines.run.to_numpy()[paper])), object)
    of = np.empty(len(names), np.int8)
    of[np.random.default_rng(seed).permutation(len(names))] = np.arange(len(names)) % folds + 1
    out = np.zeros(len(lines), np.int8)
    out[paper] = of[pd.Index(names).get_indexer(lines.run.to_numpy()[paper])]
    return out


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
    """C trained on the id arrays X with targets Y and loss weights W ([rows, 8] float32 each) from the base model folder
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
    t = cfg["train"]["c"]
    return {k: t[k] for k in ("seed", "epochs", "lr", "batch", "warmup", "clip", "weight_decay")}


def picked(cfg, lines):
    """Whether each line of lines.parquet is a training line: named in pick.parquet, which must be current with the
    inputs, the ranking and the hold labels. Refuses a pick naming no line or a line outside the train split."""
    import pandas as pd

    from . import trainset
    p = trainset.pick_path(cfg)
    if not is_current(p, trainset.made_pick(cfg)):
        raise SystemExit(f"{p} is missing or was made from other inputs, rankings or hold labels; run "
                         f"`uv run trainset`")
    pick = pd.read_parquet(p, columns=["dossier", "id"])
    if pick.duplicated().any():
        raise SystemExit(f"{p} names a line twice")
    m = lines[["dossier", "id", "split"]].reset_index().merge(pick, on=["dossier", "id"], validate="1:1")
    if len(m) != len(pick):
        raise SystemExit(f"{len(pick) - len(m):,} lines of {p} are not in the inputs; run `uv run trainset`")
    if (m.split != "train").any():
        raise SystemExit(f"{p} names {int((m.split != 'train').sum()):,} lines outside the train split")
    mask = np.zeros(len(lines), bool)
    mask[m["index"].to_numpy()] = True
    if (mask & (lines.coverage.to_numpy() == "reject")).any():
        raise SystemExit(f"{p} selects a rejected target; rebuild the pick")
    return mask, fingerprint(p)


def data(cfg, tokens, fold, limit):
    """Everything the run reads: training rows, targets, weights, the scored rows and their ids, what it is made from."""
    import pandas as pd

    from . import trainset
    d = trainset.folder(cfg)
    made = trainset.made(cfg)
    if not is_current(d / "inputs.json", made):
        raise SystemExit(f"{d} is not current with the tables and settings; run `uv run trainset`")
    made = json.loads((d / "inputs.json").read_text(encoding="utf-8"))
    if tokens != made.get("coverage_tokens"):
        raise SystemExit("train C at the input set's coverage_tokens; rebuild inputs to select another serving length")
    if tokens not in made["c_tokens"]:
        raise SystemExit(f"the inputs hold C's ids at {made['c_tokens']} tokens, not {tokens}")
    lines = pd.read_parquet(d / "lines.parquet")
    if "coverage" not in lines or not lines.coverage.isin(["cascade", "c", "reject"]).all():
        raise SystemExit("training inputs need valid coverage routes")
    if lines.duplicated(["dossier", "id"]).any():
        raise SystemExit(f"{d / 'lines.parquet'} names a line twice")
    paper = lines.paper_row.to_numpy() >= 0
    train, pick_fp = picked(cfg, lines)
    hold = trainset.hold_labels(cfg, lines)
    t = cfg["train"]["c"]
    split = lines.split.to_numpy()
    teach = None
    if fold:
        if not 1 <= fold <= t["folds"]:
            raise SystemExit(f"--fold {fold}: [train.c] folds is {t['folds']}")
        folds = line_folds(lines, t["folds"], t["fold_seed"])
        train &= folds != fold
        teach = np.flatnonzero(paper & (split == "train") & (folds == fold) &
                              (lines.coverage.to_numpy() == "cascade"))
    rows, crows = np.flatnonzero(train & paper), np.flatnonzero(train & ~paper)
    va, te = np.flatnonzero(paper & (split == "val")), np.flatnonzero(paper & (split == "test"))
    ct = np.flatnonzero(~paper & (split == "test_creayun"))
    if fold:
        va = te = ct = np.empty(0, np.int64)
    if limit:
        r0 = np.random.default_rng(0)
        count = min(limit, len(rows) + len(crows))
        nc = min(len(crows), count // 2)
        nr = min(len(rows), count - nc)
        nc = min(len(crows), count - nr)
        rows = np.sort(r0.choice(rows, nr, replace=False))
        crows = np.sort(r0.choice(crows, nc, replace=False))
        va, te, ct = va[:SMOKE_SCORED], te[:SMOKE_SCORED], ct[:SMOKE_SCORED]
    trained_runs = sorted(set(lines.run.to_numpy()[rows]))
    if fold:
        fold_overlap(trained_runs, lines.run.to_numpy()[teach], f"teacher fold {fold}")
    rows = np.concatenate([rows, crows])
    yk = lines[I.KINDS].to_numpy(np.float32)[rows]
    unknown = (yk < 0).sum(0)
    wk = (yk >= 0) & ~lines.kmask.to_numpy()[rows, None]
    yk = np.maximum(yk, 0)
    yh = hold[rows]
    Y = np.concatenate([yk, (yh == 1)[:, None]], 1).astype(np.float32)
    W = np.concatenate([wk, (yh != -1)[:, None]], 1).astype(np.float32)
    ids, off = trainset.ragged(d / f"c-{tokens}.npy")
    if len(off) != len(lines) + 1:
        raise SystemExit(f"c-{tokens}.npy holds {len(off) - 1:,} lines, lines.parquet {len(lines):,}")

    def take(r):
        return [np.array(ids[off[k]:off[k + 1]]) for k in r]

    counts = {"train": len(rows), "train_owner": len(rows) - len(crows), "train_kinds_only": len(crows),
              "train_hold1": int((yh == 1).sum()), "val": len(va), "val_hold1": int((hold[va] == 1).sum()),
              "test": len(te), "test_creayun": len(ct),
              "kind_unknown_masked": dict(zip(I.KINDS, unknown.tolist())),
              "kind_weighted": dict(zip(I.KINDS, wk.sum(0).tolist()))}
    made_from = {"inputs": made, "hold": {b: fingerprint(trainset.hold_path(cfg, b))
                                          for b in cfg["train"]["owner"]}, "pick": pick_fp}
    if fold:
        counts["teacher"] = len(teach)
    return {"X": take(rows), "Y": Y, "W": W, "lines": lines, "va": va, "te": te, "ct": ct,
            "Xva": take(va), "Xte": take(te), "Xct": take(ct), "vhold": hold[va], "counts": counts,
            "made_from": made_from, "teach": teach, "trained_runs": trained_runs, "ids": ids, "off": off}


def fold_overlap(trained_runs, scored_runs, what):
    """The teacher folds' leakage check: refuses when a run is both among the Paper runs a fold model trained on and
    among the runs of the lines it scores."""
    both = sorted(set(trained_runs) & set(scored_runs))
    if both:
        raise SystemExit(f"{what}: {len(both):,} runs are both in the model's training lines and in the lines it "
                         f"scores as a teacher: {', '.join(both[:5])}{' ...' if len(both) > 5 else ''}")


def teacher_logits(model, ids, off, rows, what):
    """A teacher's logits of the lines `rows` of lines.parquet from the ragged ids (ids, off): fp32 weights under bf16
    autocast, as the ranker scores, in length-sorted chunks."""
    import torch

    from .trainset import RANK_CHUNK, ranker_logits
    dev = torch.device("cuda")
    model.eval()
    z = np.zeros((len(rows), len(I.OUTS)), np.float32)
    order = np.argsort(np.diff(off)[rows], kind="stable")
    t0 = time.perf_counter()
    for i in range(0, len(order), RANK_CHUNK):
        idx = order[i:i + RANK_CHUNK]
        z[idx] = ranker_logits(model, [ids[off[r]:off[r + 1]] for r in rows[idx]], dev)
        done = min(len(order), i + RANK_CHUNK)
        log.info("%s: %s of %s lines (%.0f s, %.0f a second)", what, f"{done:,}", f"{len(rows):,}",
                 time.perf_counter() - t0, done / (time.perf_counter() - t0))
    if not np.isfinite(z).all():
        raise SystemExit(f"{what}: a logit that is not finite")
    return z


def frame(lines, rows, z, period=True, model="c"):
    import pandas as pd
    r = pd.DataFrame({"dossier": lines.dossier.to_numpy()[rows], "id": lines.id.to_numpy()[rows]})
    if period:
        r["period"] = lines.split.to_numpy()[rows]
    r["coverage"] = lines.coverage.to_numpy()[rows]
    for k, o in enumerate(I.OUTS):
        r[f"{model}_{o}"] = z[:, k]
    return r


def score(model, seqs):
    """C's fp32 logits of the id arrays (no TF32, no autocast), as export scores."""
    import torch

    from .models import fp32, score_c
    fp32()
    model.eval()
    z = score_c(model, seqs, torch.device("cuda"))
    if not np.isfinite(z).all():
        raise SystemExit("C gave a non-finite logit")
    return z


def score_covered(model, seqs, lines, rows):
    """Keep every row key. Rejected targets have no model score and are blocked by the coverage rule."""
    valid = lines.coverage.to_numpy()[rows] != "reject"
    z = np.zeros((len(rows), len(I.OUTS)), np.float32)
    if valid.any():
        z[valid] = score(model, [x for x, keep in zip(seqs, valid) if keep])
    return z


def c_name(tokens, fold, smoke):
    """A C run's folder name: c-<N>[-fold<K>][-smoke]."""
    return f"c-{tokens}" + (f"-fold{fold}" if fold else "") + ("-smoke" if smoke else "")


def fold_name(run_name, k):
    """The name of teacher fold k of the C run `run_name`."""
    smoke = run_name.endswith("-smoke")
    return run_name.removesuffix("-smoke") + f"-fold{k}" + ("-smoke" if smoke else "")


def checkpoints(out, request, extra=()):
    """Verified completed epochs, including optimizer and random state, in epoch order."""
    from .base import sha256
    slots = sorted((p.parent for p in out.glob("epoch-*/complete.json")),
                   key=lambda p: int(p.name.split("-")[1]))
    for expected_epoch, slot in enumerate(slots, 1):
        record = json.loads((slot / "complete.json").read_text(encoding="utf-8"))
        if slot.name != f"epoch-{expected_epoch}" or record.get("request") != request:
            raise SystemExit(f"{slot}: checkpoint epochs are incomplete or belong to another request")
        expected = {"config.json", "model.safetensors", "tokenizer.json", "optimizer.pt", *extra}
        if set(record.get("files", {})) != expected:
            raise SystemExit(f"{slot}: checkpoint file list is incomplete")
        for name, digest in record["files"].items():
            if not (slot / name).is_file() or sha256(slot / name) != digest:
                raise SystemExit(f"{slot / name}: checkpoint changed")
    return slots


def train_c(cfg, args):
    from .base import path as base_path
    from .base import pinned
    tokens, fold, limit = args.tokens, args.fold, args.limit
    s = settings(cfg)
    if args.epochs:
        s["epochs"] = args.epochs
    name = c_name(tokens, fold, bool(limit or args.epochs))
    out = cfg.path(cfg["train"]["out"]) / name
    if (out / "input.json").exists() and not args.force:
        raise SystemExit(f"{out} holds a finished run; --force trains it again")
    base = base_path(cfg, "c")
    t0 = time.perf_counter()
    D = data(cfg, tokens, fold, limit)
    c = D["counts"]
    log.info("%s: train %s lines = owner's %s (hold = 1 %s) + kinds-only %s; val %s (hold = 1 %s), test %s, "
             "test_creayun %s; %d tokens (%.0f s)", name, f"{c['train']:,}", f"{c['train_owner']:,}",
             f"{c['train_hold1']:,}", f"{c['train_kinds_only']:,}", f"{c['val']:,}", f"{c['val_hold1']:,}",
             f"{c['test']:,}", f"{c['test_creayun']:,}", tokens, time.perf_counter() - t0)
    import torch

    from .base import sha256
    from .core import source_hash
    from .models import limit_gpu_memory, load_c
    repo, rev, _ = pinned(cfg, "c")
    t = cfg["train"]["c"]
    request = json.loads(json.dumps({
        "tokens": tokens, "fold": {"k": fold, "of": t["folds"], "seed": t["fold_seed"]},
        "limit": limit, "smoke": bool(limit or args.epochs), "settings": s,
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
        model, rec = load_c(out, torch.device("cuda")), kept["record"]
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
    if not fold:
        zv, zt, zc = (score_covered(model, D["X" + key], lines, D[key]) for key in ("va", "te", "ct"))
        rec["score_s"] = round(time.perf_counter() - t1)
        import pandas as pd
        pd.concat([frame(lines, D["va"], zv), frame(lines, D["te"], zt)]).to_parquet(out / "scores.parquet", index=False)
        frame(lines, D["ct"], zc, period=False).to_parquet(out / "creayun.parquet", index=False)
        covered = lines.coverage.to_numpy()[D["va"]] != "reject"
        rec["val_metric"] = hold_metric(zv[covered, I.HOLD], D["vhold"][covered]) if (D["vhold"][covered] == 1).any() else None
        log.info("scored %s validation, %s test and %s test_creayun lines in fp32 (%d s); validation hold metric %s",
                 f"{len(zv):,}", f"{len(zt):,}", f"{len(zc):,}", rec["score_s"], rec["val_metric"])
    if fold:
        t2 = time.perf_counter()
        teach = D["teach"]
        z = teacher_logits(model, D["ids"], D["off"], teach, f"{name} teacher scores")
        tf = frame(lines, teach, z, period=False)
        tf.insert(2, "run", lines.run.to_numpy()[teach])
        tf.to_parquet(out / "teacher.parquet", index=False)
        write_text(out / "trained-runs.json", json.dumps(D["trained_runs"]) + "\n")
        rec["teacher_s"] = round(time.perf_counter() - t2)
        log.info("teacher scores of fold %d's %s train lines (%s runs, none of the %s runs trained on) in %d s", fold,
                 f"{len(teach):,}", f"{tf.run.nunique():,}", f"{len(D['trained_runs']):,}", rec["teacher_s"])
    write_text(out / "train.json", json.dumps({"settings": s, "counts": c, **rec}, indent=1) + "\n")
    repo, rev, _ = pinned(cfg, "c")
    t = cfg["train"]["c"]
    what = (f"teacher fold {fold} of {t['folds']} (fold seed {t['fold_seed']}), " if fold else "") + \
        (f"smoke run on {limit:,} lines, " if limit else "")
    write_text(out / "input.json", json.dumps({
        "model": "C",
        "run": f"{name}: C trained at {tokens} tokens, {what}seed {s['seed']}, epoch {s['epochs']} of {s['epochs']}",
        "base": {"repo": repo, "revision": rev}, "max_tokens": tokens, "trained_tokens": tokens,
        "coverage_tokens": D["made_from"]["inputs"]["coverage_tokens"],
        "ctx_lines": I.CTX_LINES, "ctx_seconds": I.CTX_SECONDS, "seed": s["seed"], "epoch": s["epochs"],
        "outputs": I.OUTS, "fold": {"k": fold, "of": t["folds"], "seed": t["fold_seed"]} if fold else None,
        "limit": limit or None, "smoke": bool(limit or args.epochs), "settings": s, "counts": c,
        "labels": "the owner's lines: hold.parquet of each [train] owner batch; kinds-only lines teach the kinds only "
                  "(hold -1)",
        "made_from": D["made_from"]}, indent=1) + "\n")
    log.info("%s written (%.1f min)", out, (time.perf_counter() - t0) / 60)


# ---------------------------------------------------------------- SH

def sh_settings(cfg):
    t = cfg["train"]["sh"]
    return {k: t[k] for k in ("seed", "layers", "epochs", "patience", "batch", "chunk", "lr", "new_lr", "head_lr",
                              "weight_decay", "warmup", "clip", "temperature")}


def prior_logodds(Y, W):
    """Per output, the log-odds of 1 among the training lines that carry a 0 or 1 label for it (a kind's 0.5 targets
    and the lines out of its loss do not count), clipped to [1e-4, 0.5]: SH's head bias at the start."""
    lab = (W > 0) & ((Y == 0) | (Y == 1))
    p = ((Y == 1) & lab).sum(0) / np.maximum(1, lab.sum(0))
    p = np.clip(p.astype(np.float64), 1e-4, 0.5)
    return np.log(p / (1 - p))


def sh_chunks(keys, chunk):
    """The training rows cut into chunks: runs of at most `chunk` consecutive rows with the same key (log file)."""
    out, start = [], 0
    for k in range(1, len(keys) + 1):
        if k == len(keys) or keys[k] != keys[start] or k - start == chunk:
            out.append(np.arange(start, k))
            start = k
    return out


def sh_batches(chunks, per, rng):
    """An epoch's batches: the chunks shuffled, `per` chunks a batch."""
    cs = [chunks[i] for i in rng.permutation(len(chunks))]
    return [np.concatenate(cs[i:i + per]) for i in range(0, len(cs), per)]


def fit_sh(T, F, Y, W, S, keys, base, s, evaluate, checkpoint=None, resume=None):
    """SH trained by distillation. T: the line ids of every line F names (indexable by F["line"]); F: the SH features
    of the training rows (line, slot, addr, chan, gap, pos, n); Y, W: hard targets and loss weights, S: the teacher's
    logits ([rows, 8] float32 each); keys: each training row's log file, whose consecutive rows make the chunks; base:
    e5-small's folder; s: the settings (sh_settings, plus max_steps for a smoke run). After each epoch
    evaluate(model) gives the validation metric; each epoch is checkpointed and the record identifies the best epoch;
    training stops `patience` epochs after the best (a smoke run after its second epoch). The loss is the soft BCE of
    z / T against sigmoid(teacher / T), times T^2, on all eight outputs of every line, plus the hard BCE on the outputs
    each line takes part in, summed and divided by the number of terms. Returns the model of the last epoch (in train
    mode on the GPU) and the training record."""
    import torch
    import torch.nn.functional as F_

    from .models import load_sh, new_sh, pad, sh_groups
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.fp32_precision = "tf32"
    dev = torch.device("cuda")
    rng = np.random.default_rng(s["seed"])
    torch.manual_seed(s["seed"])
    model = load_sh(resume, dev).train() if resume is not None else new_sh(base, s["layers"], prior_logodds(Y, W)).to(dev)
    opt = torch.optim.AdamW(sh_groups(model, s["lr"], s["new_lr"], s["head_lr"]), weight_decay=s["weight_decay"],
                            fused=True)
    chunks = sh_chunks(keys, s["chunk"])
    per = max(1, s["batch"] // s["chunk"])
    per_epoch = math.ceil(len(chunks) / per)
    Temp, max_steps = s["temperature"], s.get("max_steps") or 0
    epochs = min(s["epochs"], 2) if max_steps else s["epochs"]
    per_epoch = min(per_epoch, max_steps) if max_steps else per_epoch
    steps = epochs * per_epoch
    warm = int(s["warmup"] * steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda k: k / max(1, warm) if k < warm else max(0.0, (steps - k) / max(1, steps - warm)))
    Yt, Wt, St = torch.as_tensor(Y), torch.as_tensor(W), torch.as_tensor(S)

    def gpu(x):
        """A copy from pinned memory that does not wait for the card."""
        x = torch.from_numpy(np.ascontiguousarray(x)) if isinstance(x, np.ndarray) else x
        return x.pin_memory().to(dev, non_blocking=True)

    def encode(toks):
        """The line encoder in two length buckets (up to 16 tokens, longer)."""
        lens = np.array([len(t) for t in toks])
        parts, order = [], []
        for ix in (np.flatnonzero(lens <= 16), np.flatnonzero(lens > 16)):
            if len(ix):
                ids, mask = (gpu(x) for x in pad([toks[i] for i in ix]))
                parts.append(model.encoder(ids, mask))
                order.append(ix)
        return torch.cat(parts)[gpu(np.argsort(np.concatenate(order)))]

    def forward(P):
        """The logits of training rows P: each line their sequences use encoded once."""
        L = int(F["n"][P].max())
        lines = F["line"][P, :L]
        used = lines >= 0
        uniq, inv = np.unique(lines[used], return_inverse=True)
        vecs = encode([T[u] for u in uniq])
        g = torch.zeros(lines.shape, dtype=torch.long)
        g[torch.from_numpy(used)] = torch.from_numpy(inv.reshape(-1).astype(np.int64))
        g, mask = gpu(g), gpu(used)
        f = {k: gpu(F[k][P, :L].astype(np.int64)) for k in ("slot", "addr", "chan", "gap", "pos")}
        vec = vecs[g] * mask[..., None].to(vecs.dtype)
        return model.context(vec, f["slot"], f["addr"], f["chan"], f["gap"], f["pos"], mask)

    rec = {"steps_planned": steps, "steps_per_epoch": per_epoch, "warmup_steps": warm, "epochs": [], "gpu": []}
    best, step, first_epoch = None, 0, 0
    if resume is not None:
        saved = torch.load(resume / "optimizer.pt", map_location="cpu", weights_only=True)
        if saved["steps_planned"] != steps:
            raise SystemExit("saved SH optimizer belongs to another training schedule")
        opt.load_state_dict(saved["optimizer"])
        sched.load_state_dict(saved["scheduler"])
        rng.bit_generator.state = saved["numpy_rng"]
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
        step, first_epoch, rec, best = saved["step"], saved["next_epoch"], saved["record"], saved["best"]
        del saved
    torch.cuda.reset_peak_memory_stats()
    t_start = time.perf_counter()
    for ep in range(first_epoch, epochs):
        if best is not None and ep - best[1] >= s["patience"]:
            break
        t0, losses = time.perf_counter(), []
        for idx in sh_batches(chunks, per, rng):
            y, w, st = gpu(Yt[idx]), gpu(Wt[idx]), gpu(St[idx])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = forward(idx)
            z = z.float()
            lh = F_.binary_cross_entropy_with_logits(z, y, reduction="none") * w
            ls = F_.binary_cross_entropy_with_logits(z / Temp, torch.sigmoid(st / Temp), reduction="none") * Temp * Temp
            loss = (lh.sum() + ls.sum()) / (w.sum() + ls.numel())
            if not torch.isfinite(loss).item():
                raise SystemExit(f"non-finite SH loss at step {step + 1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), s["clip"], error_if_nonfinite=True)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            losses.append(loss.detach())
            step += 1
            if step == 50 or step % 1000 == 0 or len(losses) == max_steps:
                ms = (time.perf_counter() - t0) / len(losses) * 1000
                rep, _ = memory_report()
                recent = torch.stack(losses[-1000:]).mean().item()
                rec["gpu"].append({"step": step, "loss": recent, "lr": sched.get_last_lr()[0], "ms_per_step": round(ms),
                                   "report": rep})
                log.info("epoch %d step %d/%d loss %.8f lr %.2e, %.0f ms/step; %s", ep + 1, step, steps, recent,
                         sched.get_last_lr()[0], ms, rep)
            if max_steps and len(losses) >= max_steps:
                break
        train_s = time.perf_counter() - t0
        rep, _ = memory_report()
        t1 = time.perf_counter()
        metric = evaluate(model)
        model.train()
        e = {"epoch": ep + 1, "train_loss": float(torch.stack(losses).mean().item()), "val_metric": metric,
             "train_s": round(train_s), "score_s": round(time.perf_counter() - t1),
             "ms_per_step": round(train_s / len(losses) * 1000), "lr_end": sched.get_last_lr()[0],
             "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 1), "report": rep}
        if best is None or metric > best[0]:
            best = (metric, ep + 1)
            e["saved"] = True
        log.info("epoch %d: %s", ep + 1, json.dumps(e))
        rec["epochs"].append(e)
        if checkpoint is not None:
            checkpoint(model, rec, {"optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                       "numpy_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                       "cuda_rng": torch.cuda.get_rng_state_all(), "step": step, "next_epoch": ep + 1,
                       "record": rec, "best": best, "steps_planned": steps})
        if ep + 1 - best[1] >= s["patience"] or (max_steps and ep + 1 >= 2):
            break
    rec["best_epoch"], rec["best_val_metric"] = best[1], best[0]
    rec["stopped_after"] = len(rec["epochs"])
    rec["train_s"] = sum(e["train_s"] for e in rec["epochs"])
    rec["peak_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 2**30, 1)
    return model, rec


def sh_logits(sh, T, feats, rows):
    """SH's fp32 logits (no TF32, no autocast; as export scores) of the lines `rows` of the features `feats`: the
    lines their sequences use encoded once (models.score_sh)."""
    import torch

    from .models import score_sh
    prec = (torch.backends.cuda.matmul.fp32_precision, torch.backends.cudnn.fp32_precision)
    torch.backends.cuda.matmul.fp32_precision = torch.backends.cudnn.fp32_precision = "ieee"
    sh.eval()
    try:
        line = feats["line"][rows].astype(np.int64)
        used = line >= 0
        uniq, inv = np.unique(line[used], return_inverse=True)
        f = {k: feats[k][rows].astype(np.int64) for k in ("slot", "addr", "chan", "gap", "pos", "n")}
        f["line"] = np.full(line.shape, -1, np.int64)
        f["line"][used] = inv
        z = score_sh(sh, [T[u] for u in uniq], f, torch.device("cuda"))
    finally:
        torch.backends.cuda.matmul.fp32_precision, torch.backends.cudnn.fp32_precision = prec
    if not np.isfinite(z).all():
        raise SystemExit("SH gave a non-finite logit")
    return z


def sh_covered_logits(sh, tokens, feats, lines, rows):
    """Keep full score keys, without running SH on rejected targets."""
    covered = lines.coverage.to_numpy()[rows] != "reject"
    z = np.zeros((len(rows), len(I.OUTS)), np.float32)
    if covered.any():
        z[covered] = sh_logits(sh, tokens, feats, rows[covered])
    return z


def top(scores, share):
    """The score of the ceil(share * n)-th highest of n scores: scores >= it are that share (more on a tie)."""
    s = np.sort(np.asarray(scores, np.float64))[::-1]
    return s[max(1, int(np.ceil(share * len(s)))) - 1]


def pass_metric(sh_hold, c_hold, pass_share, hold_share):
    """SH's epoch rule: the share of the lines C holds (its hold logit at the top hold_share of the lines) that SH
    passes (its hold logit at the top pass_share of the same lines)."""
    held = c_hold >= top(c_hold, hold_share)
    return float((sh_hold[held] >= top(sh_hold, pass_share)).mean())


def sh_data(cfg, limit):
    """What SH trains and is scored on: every train-split judged Paper line (with --limit, `limit` of them drawn with
    seed 0), the validation and test lines, their hard targets, the SH features and line ids."""
    import pandas as pd

    from . import trainset
    d = trainset.folder(cfg)
    if not is_current(d / "inputs.json", trainset.made(cfg)):
        raise SystemExit(f"{d} is not current with the tables and settings; run `uv run trainset`")
    made = json.loads((d / "inputs.json").read_text(encoding="utf-8"))
    lines = pd.read_parquet(d / "lines.parquet")
    paper = lines.paper_row.to_numpy() >= 0
    n = int(paper.sum())
    feats = dict(np.load(d / "sh-features.npz"))
    if not paper[:n].all() or len(feats["n"]) != n:
        raise SystemExit(f"{d}: sh-features.npz holds {len(feats['n']):,} lines, lines.parquet {n:,} judged Paper "
                         "lines first")
    ids, off = trainset.ragged(d / "sh-tokens.npy")
    ids = np.asarray(ids)
    T = [ids[off[k]:off[k + 1]] for k in range(len(off) - 1)]
    if int(feats["line"].max()) >= len(T):
        raise SystemExit(f"{d}: sh-features.npz names a Paper row past sh-tokens.npy's {len(T):,}")
    hold = trainset.hold_labels(cfg, lines)
    split = lines.split.to_numpy()
    tr = np.flatnonzero(paper & (split == "train") & (lines.coverage.to_numpy() == "cascade"))
    all_train = len(tr)
    if limit:
        tr = np.sort(np.random.default_rng(0).choice(tr, min(len(tr), limit), replace=False))
    va, te = np.flatnonzero(paper & (split == "val")), np.flatnonzero(paper & (split == "test"))
    yk = lines[I.KINDS].to_numpy(np.float32)[tr]
    unknown = (yk < 0).sum(0)
    wk = (yk >= 0) & ~lines.kmask.to_numpy()[tr, None]
    yk = np.maximum(yk, 0)
    yh = hold[tr]
    Y = np.concatenate([yk * wk, (yh == 1)[:, None]], 1).astype(np.float32)
    W = np.concatenate([wk, (yh != -1)[:, None]], 1).astype(np.float32)
    counts = {"train": len(tr), "train_all": all_train, "train_hold1": int((yh == 1).sum()),
              "train_hold_out": int((yh == -1).sum()), "train_kind_masked": int((~wk.any(1)).sum()), "val": len(va),
              "val_hold1": int((hold[va] == 1).sum()), "test": len(te),
              "kind_unknown_masked": dict(zip(I.KINDS, unknown.tolist())),
              "kind_weighted": dict(zip(I.KINDS, wk.sum(0).tolist()))}
    return {"lines": lines, "feats": feats, "T": T, "tr": tr, "va": va, "te": te, "Y": Y, "W": W, "hold": hold,
            "counts": counts, "made": made}


def c_run_record(folder):
    p = folder / "input.json"
    if not p.exists():
        raise SystemExit(f"{folder} holds no finished C run (no input.json)")
    info = json.loads(p.read_text(encoding="utf-8"))
    if info.get("coverage_tokens") != info.get("max_tokens"):
        raise SystemExit(f"{p}: C's coverage length must match its trained and served length")
    return info


def check_c_scores(lines, scores, limit):
    """Require the exact C scoring scope. Only an explicitly limited C run uses its first smoke-scored rows."""
    import pandas as pd
    if scores.duplicated(["dossier", "id"]).any() or not np.isfinite(scores.c_hold.to_numpy()).all():
        raise SystemExit("C scores contain a duplicate key or non-finite logit")
    key = pd.MultiIndex.from_arrays([lines.dossier, lines.id])
    pos = key.get_indexer(pd.MultiIndex.from_arrays([scores.dossier, scores.id]))
    paper = lines.paper_row.to_numpy() >= 0
    expected = []
    for period in ("val", "test"):
        rows = np.flatnonzero(paper & (lines.split.to_numpy() == period))
        expected.append(rows[:SMOKE_SCORED] if limit else rows)
    if (pos < 0).any() or not np.array_equal(np.sort(pos), np.sort(np.concatenate(expected))):
        raise SystemExit("C must score every key of its recorded validation and test scope exactly once")
    if (not np.array_equal(scores.period.to_numpy(), lines.split.to_numpy()[pos]) or
            not np.array_equal(scores.coverage.to_numpy(), lines.coverage.to_numpy()[pos])):
        raise SystemExit("C score periods or coverage routes differ from the input table")
    return pos


def fold_teacher(cfg, root, run_name, crun, D):
    """The teacher scores of SH's training lines from the teacher folds of the C run `run_name`: each line scored by
    the fold model that did not train on its run. Refuses a fold made from other data or settings than the run, a
    fold that scored a run it trained on, and a training line scored by no fold or by two."""
    import pandas as pd
    t = cfg["train"]["c"]
    parts, used = [], {}
    for k in range(1, t["folds"] + 1):
        name = fold_name(run_name, k)
        f = root / name
        info = c_run_record(f)
        if info.get("fold") != {"k": k, "of": t["folds"], "seed": t["fold_seed"]}:
            raise SystemExit(f"{f}: fold {info.get('fold')}, not fold {k} of {t['folds']} with fold seed "
                             f"{t['fold_seed']}")
        if info["max_tokens"] != crun["max_tokens"] or info["made_from"] != crun["made_from"]:
            raise SystemExit(f"{f} was trained at another length or on other inputs, labels or lines than "
                             f"{root / run_name}")
        if (info.get("coverage_tokens") != crun.get("coverage_tokens") or
                info.get("settings") != crun.get("settings") or info.get("limit") != crun.get("limit")):
            raise SystemExit(f"{f} has other coverage, training settings or smoke limits than {root / run_name}")
        tp = f / "teacher.parquet"
        if not tp.exists():
            raise SystemExit(f"{f}: no teacher.parquet; train the fold again with `uv run train c --tokens "
                             f"{info['max_tokens']} --fold {k}`")
        tz = pd.read_parquet(tp)
        if tz.duplicated(["dossier", "id"]).any() or not tz.coverage.eq("cascade").all():
            raise SystemExit(f"{tp} must score unique cascade targets only")
        trained = json.loads((f / "trained-runs.json").read_text(encoding="utf-8"))
        fold_overlap(trained, tz.run.to_numpy(), name)
        parts.append(tz)
        used[name] = {"teacher": fingerprint(tp), "input": fingerprint(f / "input.json")}
    tz = pd.concat(parts, ignore_index=True)
    lines, tr = D["lines"], D["tr"]
    key = pd.MultiIndex.from_arrays([lines.dossier.to_numpy(), lines.id.to_numpy()])
    pos = key.get_indexer(pd.MultiIndex.from_arrays([tz.dossier.to_numpy(), tz.id.to_numpy()]))
    if (pos < 0).any():
        raise SystemExit(f"the teacher folds score {int((pos < 0).sum()):,} lines that are not in the inputs")
    want = np.zeros(len(lines), np.int64)
    np.add.at(want, pos, 1)
    paper_train = ((lines.paper_row.to_numpy() >= 0) & (lines.split.to_numpy() == "train") &
                   (lines.coverage.to_numpy() == "cascade"))
    if (want[paper_train] != 1).any() or want[~paper_train].any():
        raise SystemExit(f"the teacher folds score {int((want[paper_train] == 0).sum()):,} train lines not at all, "
                         f"{int((want[paper_train] > 1).sum()):,} twice and {int(want[~paper_train].sum()):,} lines "
                         "outside the train split")
    z = np.zeros((len(lines), len(I.OUTS)), np.float32)
    logits = tz[[f"c_{o}" for o in I.OUTS]].to_numpy(np.float32)
    if not np.isfinite(logits).all():
        raise SystemExit("a teacher fold has a non-finite logit")
    if not (tz.run.to_numpy() == lines.run.to_numpy()[pos]).all():
        raise SystemExit("a teacher fold records a run that differs from the input line's run")
    actual_fold = line_folds(lines, t["folds"], t["fold_seed"])
    for name, part in zip(used, parts):
        k = c_run_record(root / name)["fold"]["k"]
        at = key.get_indexer(pd.MultiIndex.from_arrays([part.dossier, part.id]))
        if not (actual_fold[at] == k).all():
            raise SystemExit(f"{name} scores a target assigned to another teacher fold")
    z[pos] = logits
    desc = f"{run_name}'s teacher folds ({', '.join(used)}): each line scored by the fold model that did not train " \
           f"on its run, bf16"
    return z[tr], desc, used


def train_sh(cfg, args):
    import shutil
    import tomllib

    import pandas as pd
    import torch

    from .base import path as base_path
    from .base import pinned
    from .models import limit_gpu_memory, load_sh, save_sh
    s = sh_settings(cfg)
    s["max_steps"] = args.max_steps
    root = cfg.path(cfg["train"]["out"])
    crun = c_run_record(root / args.c)
    if crun.get("fold"):
        raise SystemExit(f"--c {args.c} is a teacher fold; name the C run whose folds teach")
    if crun.get("smoke") and not (args.limit or args.max_steps):
        raise SystemExit("a smoke C can teach only a smoke SH; set --limit or --max-steps")
    name = f"sh-{args.c}"
    if (args.limit or args.max_steps) and not name.endswith("-smoke"):
        name += "-smoke"
    out = root / name
    if (out / "input.json").exists() and not args.force:
        raise SystemExit(f"{out} holds a finished run; --force trains it again")
    conf = tomllib.loads((cfg.path(cfg["bundle"]["private"]) / "bundle.toml").read_text(encoding="utf-8"))
    pass_share = conf["thresholds"]["pass_share"]
    hold_share = conf["thresholds"]["hold_shares"][cfg["train"]["sh"]["epoch_hold"]]
    base = base_path(cfg, "sh")
    t0 = time.perf_counter()
    D = sh_data(cfg, args.limit)
    from . import trainset
    hold_now = {b: fingerprint(trainset.hold_path(cfg, b)) for b in cfg["train"]["owner"]}
    if crun["made_from"]["inputs"] != D["made"] or crun["made_from"]["hold"] != hold_now:
        raise SystemExit(f"{root / args.c} was trained on other inputs or hold labels than the current ones")
    _, pick_now = picked(cfg, D["lines"])
    if crun["made_from"].get("pick") != pick_now or crun["coverage_tokens"] != D["made"]["coverage_tokens"]:
        raise SystemExit(f"{root / args.c} was trained on another pick or coverage length")
    lines, va, te = D["lines"], D["va"], D["te"]
    key = pd.MultiIndex.from_arrays([lines.dossier.to_numpy(), lines.id.to_numpy()])

    # the epoch rule's lines: the C run's scored validation lines, with C's hold logit
    cs = pd.read_parquet(root / args.c / "scores.parquet", columns=["dossier", "id", "period", "coverage", "c_hold"])
    check_c_scores(lines, cs, crun.get("limit"))
    cs = cs[cs.period == "val"]
    cs = cs[cs.coverage == "cascade"]
    pos = key.get_indexer(pd.MultiIndex.from_arrays([cs.dossier.to_numpy(), cs.id.to_numpy()]))
    at = pd.Index(va).get_indexer(pos)
    if (pos < 0).any() or (at < 0).any():
        raise SystemExit(f"{root / args.c}: {int((at < 0).sum()):,} scored validation lines are not validation lines "
                         "of the inputs")
    if not len(at):
        raise SystemExit("SH epoch selection needs cascade validation lines")
    c_hold = cs.c_hold.to_numpy(np.float64)
    vhold = D["hold"][va]

    limit_gpu_memory(log)
    S, teacher, used = fold_teacher(cfg, root, args.c, crun, D)
    c = D["counts"]
    log.info("%s: train %s lines (hold = 1 %s, hold out %s, kind-masked %s), val %s (hold = 1 %s), test %s; teacher "
             "%s; epoch rule: the share of %s's top %g%% of validation lines by C's hold logit that SH passes at %g%% "
             "(%s lines) (%.0f s)", name, f"{c['train']:,}", f"{c['train_hold1']:,}", f"{c['train_hold_out']:,}",
             f"{c['train_kind_masked']:,}", f"{c['val']:,}", f"{c['val_hold1']:,}", f"{c['test']:,}", teacher, args.c,
             100 * hold_share, 100 * pass_share, f"{len(cs):,}", time.perf_counter() - t0)
    feats, T, tr = D["feats"], D["T"], D["tr"]
    Ftr = {k: feats[k][tr] for k in feats}
    meta = {"ctx_lines": I.CTX_LINES, "ctx_seconds": I.CTX_SECONDS, "line_tokens": I.SH_LINE_TOKENS,
            "coverage_tokens": D["made"]["coverage_tokens"]}
    from .base import sha256
    from .core import source_hash
    request = json.loads(json.dumps({"model": "SH", "settings": s, "limit": args.limit, "teacher": used,
                                    "inputs": D["made"], "holds": hold_now, "c": fingerprint(root / args.c / "scores.parquet"),
                                    "code": source_hash("train", "models", "inputs", "core")}))
    if out.exists() and args.force:
        if out.resolve().parent != root.resolve():
            raise SystemExit(f"refuse to replace a run outside [train] out: {out}")
        shutil.rmtree(out)
    if out.exists():
        rp = out / "request.json"
        if not rp.is_file() or json.loads(rp.read_text(encoding="utf-8")) != request:
            raise SystemExit(f"{out}: unfinished SH uses other inputs or settings; preserve it or use --force")
    else:
        out.mkdir(parents=True)
        write_text(out / "request.json", json.dumps(request, indent=1) + "\n")
    slots = checkpoints(out, request, extra=("meta.json",))

    def evaluate(model):
        z = sh_covered_logits(model, T, feats, lines, va)
        m = pass_metric(z[at, I.HOLD], c_hold, pass_share, hold_share)
        log.info("validation: SH passes %.4f of the lines %s holds; SH alone hold metric %s", m, args.c,
                 hold_metric(z[at, I.HOLD], vhold[at]) if (vhold[at] == 1).any() else None)
        return m

    def checkpoint(model, record, state):
        slot = out / f"epoch-{state['next_epoch']}"
        if slot.exists():
            if slot.resolve().parent != out.resolve() or (slot / "complete.json").exists():
                raise SystemExit(f"refuse to replace completed or redirected checkpoint {slot}")
            slot.rename(slot.with_name(slot.name + f"-incomplete-{time.time_ns()}"))
        slot.mkdir()
        save_sh(model, slot, base, epoch=state["next_epoch"], **meta)
        torch.save(state, slot / "optimizer.pt")
        names = ("config.json", "model.safetensors", "tokenizer.json", "meta.json", "optimizer.pt")
        write_text(slot / "complete.json", json.dumps({"request": request,
                   "files": {n: sha256(slot / n) for n in names}}, indent=1) + "\n")
        log.info("saved verified SH epoch %d with optimizer and random state", state["next_epoch"])

    last, rec = fit_sh(T, Ftr, D.pop("Y"), D.pop("W"), S, lines.dossier.to_numpy()[tr], base, s, evaluate,
                       checkpoint=checkpoint, resume=slots[-1] if slots else None)
    del last
    torch.cuda.empty_cache()
    best = out / f"epoch-{rec['best_epoch']}"
    for name in ("config.json", "model.safetensors", "tokenizer.json", "meta.json"):
        shutil.copy2(best / name, out / name)
    sh = load_sh(best, torch.device("cuda"))
    t1 = time.perf_counter()
    zv, zt = sh_covered_logits(sh, T, feats, lines, va), sh_covered_logits(sh, T, feats, lines, te)
    rec["score_s"] = round(time.perf_counter() - t1)
    rec["val_metric"] = pass_metric(zv[at, I.HOLD], c_hold, pass_share, hold_share)
    rec["val_hold_metric"] = hold_metric(zv[at, I.HOLD], vhold[at]) if (vhold[at] == 1).any() else None
    pd.concat([frame(lines, va, zv, model="sh"), frame(lines, te, zt, model="sh")]).to_parquet(
        out / "scores.parquet", index=False)
    log.info("kept epoch %d of %d: SH passes %.4f of the validation lines %s holds; scored %s validation and %s test "
             "lines in fp32 (%d s)", rec["best_epoch"], rec["stopped_after"], rec["val_metric"], args.c,
             f"{len(zv):,}", f"{len(zt):,}", rec["score_s"])
    rule = {"c": args.c, "pass_share": pass_share, "hold_share": hold_share, "lines": len(cs)}
    write_text(out / "train.json", json.dumps({"settings": s, "counts": c, "teacher": teacher, "epoch_rule": rule,
                                               **rec}, indent=1) + "\n")
    repo, rev, _ = pinned(cfg, "sh")
    what = (f"smoke run on {args.limit:,} lines, " if args.limit else "") + \
        (f"smoke run of {args.max_steps} steps an epoch, " if args.max_steps else "")
    write_text(out / "input.json", json.dumps({
        "model": "SH",
        "run": f"{name}: SH taught by {teacher}, {what}seed {s['seed']}, epoch {rec['best_epoch']} (the best of "
               f"{rec['stopped_after']} by the share of {args.c}'s {100 * hold_share:g}% validation holds it passes at "
               f"{100 * pass_share:g}%)",
        "base": {"repo": repo, "revision": rev}, **meta, "seed": s["seed"], "epoch": rec["best_epoch"],
        "layers": s["layers"], "outputs": I.OUTS, "teacher": {"what": teacher, "from": used}, "epoch_rule": rule,
        "limit": args.limit or None, "max_steps": args.max_steps or None, "settings": s, "counts": c,
        "labels": "the train-split judged Paper lines: kinds from lines.parquet, hold from each [train] owner batch's "
                  "hold.parquet",
        "made_from": {"inputs": D["made"], "hold": hold_now, "c": fingerprint(root / args.c / "scores.parquet")}},
        indent=1) + "\n")
    log.info("%s written (%.1f min)", out, (time.perf_counter() - t0) / 60)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="model", required=True)
    c = sub.add_parser("c", help="train C")
    c.add_argument("--tokens", type=int, required=True, help="C's length: its input cut to this many tokens")
    c.add_argument("--fold", type=int, default=0, help="train teacher fold K of [train.c] folds")
    c.add_argument("--epochs", type=int, default=0, help="a smoke run of this many epochs")
    c.add_argument("--limit", type=int, default=0,
                   help=f"a smoke run on this many training lines (half owner's, half kinds-only, drawn with seed 0), "
                        f"scoring the first {SMOKE_SCORED} of each scored set")
    c.add_argument("--force", action="store_true", help="train again over a finished run of the same name")
    sh = sub.add_parser("sh", help="train SH")
    sh.add_argument("--c", required=True, help="the C run (a folder under [train] out) whose validation holds choose "
                                               "SH's epoch, and whose teacher folds teach SH")
    sh.add_argument("--limit", type=int, default=0, help="a smoke run on this many training lines, drawn with seed 0")
    sh.add_argument("--max-steps", type=int, default=0,
                    help="a smoke run: each epoch stops after this many steps, the run after its second epoch")
    sh.add_argument("--force", action="store_true", help="train again over a finished run of the same name")

    def body(cfg, args):
        if args.model == "c":
            train_c(cfg, args)
        else:
            train_sh(cfg, args)

    run("train", body, ap)


if __name__ == "__main__":
    main()
