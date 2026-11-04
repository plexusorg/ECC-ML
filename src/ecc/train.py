"""uv run train [--model NAME] [--steps N [--seed S] [--tag TAG]]: LoRA fine-tune of a base model as a one-token
(" ban" / " ok") classifier.

Run folder: <work>/models/<NAME>/ (NAME is a [train.models.*] entry, default [train] model).
1. Subset (once per run, subset.parquet): the labelled lines are every ban row of <work>/training/train.parquet, every row from
   the [trainset] extra files, `neg_window` ok rows inside labelled windows (hard negatives) and `neg_outside` ok rows
   outside them. Each normalised target text is kept at most `dup_cap` times per label and rows whose text matches a
   case-file text are dropped. Their conversations, in messages.parquet or the extra files, are cut into the windows
   the model host builds (chat.window_sizes): one window is the prompt of its last line, and every earlier line of it
   ends at that line's own prompt. Each window is kept up to its last labelled line. Other lines (amb, unlabelled,
   over the caps) are context only.
2. Training: one epoch, loss = cross-entropy on the token after each labelled line's newline, in one forward pass
   per window; causal attention lets that position see only the line's own prompt. A step holds whole windows with
   about `batch` labelled lines; windows are length-grouped and split into micro-batches of at most `token_budget`
   padded tokens. AdamW, warmup over 5% of the labelled lines (at most 100 x `batch`), then cosine down to 10%.
   Every `save_every` steps: val AUROC on `val_n` val rows, each with the history its streaming window holds, and a
   checkpoint in ckpt/ (the last two are kept). A rerun resumes from the newest checkpoint; Ctrl+C, or reaching
   --until STEP, finishes the current step and saves one.
Output: adapter/ and train_done.json (with the margins `export` checks the merge against).
Each output records what it was made from (subset.parquet.inputs.json, the checkpoints' and train_done.json's
settings): the input files' fingerprints, a hash of the case-file texts it drops, the recipe, the base model and
chat.PROMPT_FORMAT. A subset made from anything else is rebuilt and the run is trained again; train_done.json and
adapter/ are replaced when it finishes. Checkpoints made with other settings stop training with the changed settings
named: delete ckpt/ to train from the start.

A bounded run (--steps N) compares base models and recipe changes: the same subset, the first steps of the epoch that
hold N x `batch` labelled lines, and warmup and cosine over those steps. --seed (default [train] seed) sets the batch
order, the LoRA initialization and dropout. It goes to <work>/models/bounded/<NAME>[-<TAG>]-s<S>-n<N>/; `uv run export`
with the same flags evaluates it.
"""
import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import signal
import time
from collections import Counter, defaultdict

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .chat import (CLASSIFIER_HEADER, PROMPT_FORMAT, WINDOW_TOKENS, fitted_prompt, load_cases, prompt_line,
                   window_sizes)
from .classifier import (answer_ids, base_model, device, encode, hidden_at, pad_batch, token_count,
                         torch_margins)
from .core import (changed_keys, fingerprint, is_current, log, progress, record_inputs, run, save_state,
                   write_parquet, write_text)
from .evaluate import items_of
from .trainset import extra_cases, extra_files

CKPT = re.compile(r"step-(\d+)$")
LOG_EVERY = 20


def run_args(ap):
    ap.add_argument("--model", help="a [train.models.*] entry in ecc.toml (default: [train] model)")
    ap.add_argument("--steps", type=int, metavar="N", help="a bounded run of N steps")
    ap.add_argument("--seed", type=int, help="a bounded run's seed (default: [train] seed)")
    ap.add_argument("--tag", help="a bounded run's name for a recipe change, put in its folder name")


def run_dir(cfg, args):
    """(model name, seed, steps or None, run folder) of the run the flags select."""
    name = args.model or cfg["train"]["model"]
    if name not in cfg["train"]["models"]:
        raise SystemExit(f"no [train.models.{name}] in ecc.toml; known: {', '.join(cfg['train']['models'])}")
    seed = cfg["train"]["seed"] if args.seed is None else args.seed
    if not args.steps:
        if args.seed is not None or args.tag:
            raise SystemExit("--seed and --tag are for a bounded run; add --steps")
        return name, seed, None, cfg.work / "models" / name
    folder = f"{name}{f'-{args.tag}' if args.tag else ''}-s{seed}-n{args.steps}"
    return name, seed, args.steps, cfg.work / "models" / "bounded" / folder


# ---------------------------------------------------------------- subset

def case_files(cfg):
    return [cfg.path(f) for s in cfg["eval"]["sets"].values() for f in s]


def test_texts(cfg):
    """Normalised texts (over 15 characters) of every eval case file; training rows with these texts are dropped."""
    return {c["text"].strip().lower() for c in load_cases(case_files(cfg)) if len(c["text"].strip()) > 15}


def subset_inputs(cfg, spec):
    """What subset.parquet is made from. The case files count only through the texts they drop (test_texts)."""
    c = cfg["train"]
    texts = hashlib.sha256(json.dumps(sorted(test_texts(cfg)), ensure_ascii=False).encode()).hexdigest()
    return {"train": fingerprint(cfg.work / "training" / "train.parquet"), "labels": fingerprint(cfg.data / "labels.parquet"),
            "messages": fingerprint(cfg.data / "messages.parquet"), "extra": [fingerprint(p) for p in extra_files(cfg)],
            "test_texts": texts, "base": spec["base"],
            "recipe": [c["neg_window"], c["neg_outside"], c["dup_cap"]], "subset": "windows",
            "prompt": PROMPT_FORMAT}


def run_state(cfg, name, seed, steps, out):
    """What the run's adapter is made from now, and how its outputs stand, as `train` decides:
    (subset inputs, whether subset.parquet is current, settings, whether train_done.json is current,
    the keys in which the newest checkpoint's settings differ, or None when there is no checkpoint)."""
    c, spec = cfg["train"], cfg["train"]["models"][name]
    subset = out / "subset.parquet"
    subset_from = subset_inputs(cfg, spec)
    subset_ok = is_current(subset, subset_from)
    # what the adapter is made from; checkpoints and train_done.json carry it
    settings = {"base": spec["base"], "subset": fingerprint(subset) if subset_ok else None, "steps": steps,
                "batch": c["batch"], "lr": c["lr"], "rank": c["rank"], "seed": seed}
    done_path = out / "train_done.json"
    done = (subset_ok and done_path.exists()
            and json.loads(done_path.read_text(encoding="utf-8")).get("settings") == settings)
    found = checkpoints(out)
    ckpt_changed = None
    if found:
        was = checkpoint_state(found[-1]).get("settings") or {}
        # a checkpoint never matches a subset that is about to be rebuilt, even one written before subsets were recorded
        ckpt_changed = sorted(k for k in settings if was.get(k) != settings[k] or (k == "subset" and not subset_ok))
    return subset_from, subset_ok, settings, done, ckpt_changed


def build_subset(cfg, tok, path):
    c = cfg["train"]
    tr = pq.read_table(cfg.work / "training" / "train.parquet").to_pandas()
    window = set(pq.read_table(cfg.data / "labels.parquet", columns=["row"]).column("row").to_pylist())
    m = pq.read_table(cfg.data / "messages.parquet", columns=["conv", "speaker", "text"]).to_pydict()
    n = len(m["conv"])
    for it in extra_cases(cfg):
        for k in ("conv", "speaker", "text"):
            m[k].append(it[k])
    extra = tr.row >= n
    if any(r >= len(m["text"]) or m["text"][r] != t for r, t in zip(tr.row[extra], tr.text[extra])):
        raise SystemExit("train.parquet's rows from the [trainset] extra files do not match those files; run "
                         "`uv run trainset`")
    norm = tr.text.str.strip().str.lower()
    leak = norm.isin(test_texts(cfg))
    tr, norm = tr[~leak], norm[~leak]
    order = list(range(len(tr)))
    random.Random(7).shuffle(order)
    seen = Counter()
    keep = []
    for i in order:
        key = (norm.iat[i], int(tr.label.iat[i]))
        seen[key] += 1
        if seen[key] <= c["dup_cap"]:
            keep.append(i)
    tr = tr.iloc[keep]
    extra = tr.row >= n
    pos = tr[(tr.label == 1) | extra]
    neg = tr[(tr.label == 0) & ~extra]
    in_window = neg.row.isin(window)
    neg_w = neg[in_window].sample(min(c["neg_window"], int(in_window.sum())), random_state=7)
    neg_o = neg[~in_window].sample(min(c["neg_outside"], int((~in_window).sum())), random_state=7)
    log.info("subset: %s rows match a case-file text and are dropped; %s after the duplicate cap; %s ban and %s extra "
             "ok, %s ok in windows, %s ok outside", f"{int(leak.sum()):,}", f"{len(tr):,}",
             f"{int((pos.label == 1).sum()):,}", f"{int((pos.label == 0).sum()):,}", f"{len(neg_w):,}",
             f"{len(neg_o):,}")
    chosen = {int(r): int(y) for part in (pos, neg_w, neg_o) for r, y in zip(part.row, part.label)}
    wanted = {m["conv"][r] for r in chosen}
    convs = defaultdict(list)
    for i, conv in enumerate(m["conv"]):
        if conv in wanted:
            convs[conv].append(i)
    texts = [prompt_line(m["speaker"][i], m["text"][i]) for rows in convs.values() for i in rows]
    # window_sizes counts each line alone; counted in one batch here
    cost = dict(zip(texts, map(len, encode(tok, texts))))
    cost[CLASSIFIER_HEADER] = len(encode(tok, [CLASSIFIER_HEADER])[0])
    lines, targets, labels = [], [], []
    with progress("subset windows", len(convs), "conversations") as advance:
        for rows in convs.values():
            conv = [(m["speaker"][i], m["text"][i]) for i in rows]
            sizes = window_sizes(conv, cost.__getitem__)
            # a line starts a new window where the host rebuilds it: its history is not the previous line's plus one
            starts = [j for j in range(len(rows)) if j == 0 or sizes[j] != sizes[j - 1] + 1] + [len(rows)]
            for a, b in zip(starts, starts[1:]):
                first = a - sizes[a]
                hit = [j for j in range(a, b) if rows[j] in chosen]
                if not hit:
                    continue
                window = [prompt_line(s, t) for s, t in conv[first:hit[-1] + 1]]
                if b - a == 1 and cost[CLASSIFIER_HEADER] + cost[window[-1]] > WINDOW_TOKENS:
                    # a line too long for a prompt on its own is cut, as fitted_prompt cuts it
                    window = [fitted_prompt([], *conv[a], token_count(tok))[len(CLASSIFIER_HEADER):]]
                lines.append(window)
                targets.append([j - first for j in hit])
                labels.append([chosen[rows[j]] for j in hit])
            advance()
    order = list(range(len(lines)))
    random.Random(11).shuffle(order)
    write_parquet(pa.table({"lines": [lines[w] for w in order], "targets": [targets[w] for w in order],
                            "labels": [labels[w] for w in order]}), path)
    n = sum(map(len, targets))
    log.info("subset: %s windows hold the %s labelled lines (%.1f per window); %s lines in all", f"{len(lines):,}",
             f"{n:,}", n / len(lines), f"{sum(map(len, lines)):,}")


# ---------------------------------------------------------------- batches and schedule

def make_batches(lengths, counts, batch, seed):
    """Steps of whole windows with about `batch` labelled lines each (counts: per window). Shuffle, sort within
    chunks of 64 steps to cut padding, close a step when the next window would take it further from `batch` lines,
    shuffle step order; the longest step goes first so the peak memory shows on step 1."""
    idx = list(range(len(lengths)))
    rng = random.Random(seed)
    rng.shuffle(idx)
    batches, step, n = [], [], 0
    chunk = max(1, round(batch * 64 * len(counts) / sum(counts)))
    for c in range(0, len(idx), chunk):
        for i in sorted(idx[c:c + chunk], key=lambda i: lengths[i]):
            if step and abs(n + counts[i] - batch) > abs(n - batch):
                batches.append(step)
                step, n = [], 0
            step.append(i)
            n += counts[i]
    batches.append(step)
    rng.shuffle(batches)
    longest = max(range(len(batches)), key=lambda b: max(lengths[i] for i in batches[b]) * len(batches[b]))
    batches.insert(0, batches.pop(longest))
    return batches


def micro_batches(rows, lengths, token_budget):
    size = max(1, token_budget // max(lengths[i] for i in rows))
    return [rows[m:m + size] for m in range(0, len(rows), size)]


def lr_factors(lines, batch):
    """LR factor of each step from its labelled lines and those of the steps before it: warmup over 5% of the lines
    (at most 100 x batch), then cosine down to 10%."""
    total = sum(lines)
    warm = max(1, min(100 * batch, total // 20))
    out, seen = [], 0
    for n in lines:
        out.append(min(1.0, (seen + n) / warm) if seen < warm
                   else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (seen - warm) / max(1, total - warm))))
        seen += n
    return out


def plan(tok, subset):
    """Per window: token ids (the header's, then each line's, as window_sizes counts them), the positions of its
    labelled lines' last tokens and their answer tokens."""
    data = pq.read_table(subset).to_pydict()
    head = encode(tok, [CLASSIFIER_HEADER])[0]
    ids = encode(tok, [line for w in data["lines"] for line in w])
    ban_id, ok_id = answer_ids(tok)
    enc, positions, targets, k = [], [], [], 0
    for lines, tg, lb in zip(data["lines"], data["targets"], data["labels"]):
        parts = [head, *ids[k:k + len(lines)]]
        k += len(lines)
        ends = np.cumsum([len(p) for p in parts]) - 1
        enc.append(np.concatenate(parts))
        positions.append(ends[1:][tg])
        targets.append([ban_id if y else ok_id for y in lb])
    if max(len(e) for e in enc) > WINDOW_TOKENS:
        raise SystemExit(f"{subset} has windows over {WINDOW_TOKENS} tokens; delete it to rebuild it")
    return enc, positions, targets


def val_sample(cfg, tok, n):
    rows = pq.read_table(cfg.work / "training" / "val.parquet", columns=["row"]).to_pandas().sample(n, random_state=3).row
    count = token_count(tok)
    items = dict(items_of(cfg, "val", count))
    picked = [items[r] for r in rows]
    return ([fitted_prompt(it["history"], it["speaker"], it["text"], count) for it in picked],
            [int(it["label"] == "ban") for it in picked])


# ---------------------------------------------------------------- checkpoints

def checkpoints(out):
    found = [(int(m.group(1)), p) for p in (out / "ckpt").glob("step-*") if (m := CKPT.match(p.name))]
    return [p for _, p in sorted(found)]


def checkpoint_state(path):
    return json.loads((path / "state.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- training

def body(cfg, args):
    name, seed, steps, out = run_dir(cfg, args)
    c, spec = cfg["train"], cfg["train"]["models"][name]
    subset = out / "subset.parquet"
    subset_from, subset_ok, settings, done, ckpt_changed = run_state(cfg, name, seed, steps, out)
    done_path = out / "train_done.json"
    if done:
        log.info("%s is trained (%s); `uv run export` with the same flags is next", out.name, out / "adapter")
        return
    if ckpt_changed:
        why = ", ".join(ckpt_changed)
        if "subset" in ckpt_changed and subset.exists() and not subset_ok:
            why += f" (subset.parquet would be rebuilt: {', '.join(changed_keys(subset, subset_from))} changed)"
        raise SystemExit(f"the checkpoints in {out / 'ckpt'} were trained with other settings: {why}. Delete "
                         f"{out / 'ckpt'} to train from the start, or restore the settings to continue them")

    import torch
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file, save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .evaluate import auroc

    if device() != "cuda":
        raise SystemExit("training needs a CUDA GPU")
    base = base_model(cfg, spec["base"])
    torch.backends.cuda.matmul.fp32_precision = "tf32"
    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(base)
    if done_path.exists():
        log.info("%s was trained from other inputs; training it again (train_done.json and adapter/ are replaced at "
                 "the end)", out.name)
    if not subset_ok:
        if subset.exists():
            log.info("subset.parquet was built from other inputs (%s); rebuilding it",
                     ", ".join(changed_keys(subset, subset_from)))
        build_subset(cfg, tok, subset)
        record_inputs(subset, subset_from)
        settings["subset"] = fingerprint(subset)
    enc, positions, targets = plan(tok, subset)
    lengths, counts = [len(e) for e in enc], [len(p) for p in positions]
    batches = make_batches(lengths, counts, c["batch"], seed)
    step_lines = [sum(counts[i] for i in b) for b in batches]
    if steps:
        reach = np.searchsorted(np.cumsum(step_lines), steps * c["batch"])
        if reach == len(batches):
            raise SystemExit(f"--steps {steps} is more than the epoch's {sum(counts) // c['batch']:,} x {c['batch']} "
                             f"labelled lines")
        batches, step_lines = batches[:reach + 1], step_lines[:reach + 1]
    total = len(batches)
    ban_id = answer_ids(tok)[0]
    log.info("%s: %s labelled lines (%.1f%% ban) in %s windows (mean %d tokens); %s steps, %s lines (%.1f per step); "
             "token budget %d, gradient checkpointing %s", out.name, f"{sum(counts):,}",
             100 * sum(t == ban_id for ts in targets for t in ts) / sum(counts), f"{len(enc):,}",
             sum(lengths) / len(lengths), f"{total:,}", f"{sum(step_lines):,}", sum(step_lines) / total,
             spec["token_budget"], spec["grad_ckpt"])
    val_prompts, val_labels = val_sample(cfg, tok, c["val_n"])

    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").cuda()
    if spec["grad_ckpt"]:
        model.gradient_checkpointing_enable()
    model = get_peft_model(model, LoraConfig(r=c["rank"], lora_alpha=2 * c["rank"], lora_dropout=0.05,
                                             target_modules="all-linear", task_type="CAUSAL_LM"))
    causal = model.base_model.model
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=c["lr"], weight_decay=0.0, fused=True)
    factors = lr_factors(step_lines, c["batch"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: factors[min(s, total - 1)])

    start = 0
    for stale in (out / "ckpt").glob("*.tmp"):
        shutil.rmtree(stale)
    found = checkpoints(out)
    if found:
        last = found[-1]
        state = checkpoint_state(last)
        set_peft_model_state_dict(model, load_file(str(last / "adapter.safetensors")))
        opt.load_state_dict(torch.load(last / "optim.pt", weights_only=True))
        sched.load_state_dict(torch.load(last / "sched.pt", weights_only=True))
        rng = torch.load(last / "rng.pt", weights_only=True)
        torch.set_rng_state(rng["cpu"])
        torch.cuda.set_rng_state(rng["cuda"])
        start = state["step"]
        log.info("resumed from %s", last)
    if args.until and args.until <= start:
        log.info("%s is at step %d already (--until %d)", name, start, args.until)
        return

    def save(step):
        final = out / "ckpt" / f"step-{step}"
        if final.exists():
            return
        d = final.with_name(final.name + ".tmp")
        d.mkdir(parents=True, exist_ok=True)
        save_file(get_peft_model_state_dict(model), str(d / "adapter.safetensors"))
        torch.save(opt.state_dict(), d / "optim.pt")
        torch.save(sched.state_dict(), d / "sched.pt")
        torch.save({"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state()}, d / "rng.pt")
        (d / "state.json").write_text(json.dumps({"step": step, "total": total, "settings": settings}),
                                      encoding="utf-8")
        os.replace(d, final)
        for old in checkpoints(out)[:-2]:
            shutil.rmtree(old)

    def validate(step):
        model.eval()
        m = torch_margins(causal, tok, val_prompts)
        model.train()
        log.info("step %d: val AUROC %.4f on %d rows", step, auroc(
            [x for x, y in zip(m, val_labels) if y], [x for x, y in zip(m, val_labels) if not y]), len(m))

    stop = []
    handlers = {s: signal.signal(s, lambda *_: stop.append(True))
                for s in [signal.SIGINT] + ([signal.SIGBREAK] if hasattr(signal, "SIGBREAK") else [])}
    model.train()
    torch.cuda.reset_peak_memory_stats()
    first_loss, step_times = None, []
    run_loss, run_n, run_tokens, t0 = 0.0, 0, 0, time.perf_counter()
    began = t0
    try:
        with progress(f"train {out.name}", total, "steps", completed=start,
                      info="loss {task.fields[loss]}  {task.fields[step_s]} s/step", loss="-", step_s="-") as advance:
            for step in range(start, total):
                rows = batches[step]
                loss_sum = 0.0
                for micro in micro_batches(rows, lengths, spec["token_budget"]):
                    ids, mask, _ = pad_batch([enc[i] for i in micro], tok.pad_token_id, "cuda")
                    at = torch.tensor([r for r, i in enumerate(micro) for _ in positions[i]], device="cuda")
                    cols = torch.from_numpy(np.concatenate([positions[i] for i in micro])).cuda()
                    tgt = torch.tensor([t for i in micro for t in targets[i]], device="cuda")
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits = causal.lm_head(hidden_at(causal, ids, mask, at, cols)).float()
                    loss = torch.nn.functional.cross_entropy(logits, tgt, reduction="sum") / step_lines[step]
                    loss.backward()
                    loss_sum += loss.item()
                    run_tokens += int(mask.sum())
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                first_loss = loss_sum if first_loss is None else first_loss
                run_loss += loss_sum
                run_n += 1
                advance()
                if (step + 1) % LOG_EVERY == 0 or step == start or step + 1 == total:
                    dt = (time.perf_counter() - t0) / run_n
                    if step > start:
                        step_times.append(dt)
                    advance(0, loss=f"{run_loss / run_n:.4f}", step_s=f"{dt:.2f}")
                    log.debug("step %d/%d loss %.4f lr %.2e step_s %.3f tokens_s %.0f peak_reserved_gib %.1f",
                              step + 1, total, run_loss / run_n, sched.get_last_lr()[0], dt, run_tokens / (dt * run_n),
                              torch.cuda.max_memory_reserved() / 2**30)
                    run_loss, run_n, run_tokens, t0 = 0.0, 0, 0, time.perf_counter()
                if (step + 1) % c["save_every"] == 0 or step + 1 == total:
                    validate(step + 1)
                    save(step + 1)
                    t0 = time.perf_counter()
                if stop or step + 1 == args.until < total:
                    save(step + 1)
                    log.info("checkpoint saved at step %d", step + 1)
                    if stop:
                        raise KeyboardInterrupt
                    return
    finally:
        for s, h in handlers.items():
            signal.signal(s, h)

    minutes = (time.perf_counter() - began) / 60
    model.save_pretrained(out / "adapter")
    tok.save_pretrained(out / "adapter")
    model.eval()
    check = torch_margins(causal, tok, val_prompts[:64])
    peak = torch.cuda.max_memory_reserved() / 2**30
    write_text(done_path, json.dumps({
        "settings": settings, "model": base, "steps": total, "examples": sum(counts), "windows": len(enc),
        "lines_trained": sum(step_lines), "batch": c["batch"], "lr": c["lr"], "rank": c["rank"], "seed": seed,
        "first_loss": first_loss,
        "median_step_s": sorted(step_times)[len(step_times) // 2] if step_times else None,
        # train_minutes covers steps resumed_at+1 to the end; earlier sessions are not counted
        "train_minutes": minutes, "resumed_at": start,
        "peak_reserved_gib": peak, "check_prompts": val_prompts[:64], "check_margins": check}, indent=1))
    log.info("%s trained: %s steps in %.1f min; peak reserved %.1f GiB; adapter in %s", out.name, f"{total:,}",
             minutes, peak, out / "adapter")
    if not steps:
        save_state(cfg, f"train-{name}", [cfg.work / "training" / "train.parquet", cfg.data / "labels.parquet"], steps=total,
                   examples=sum(counts))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    run_args(ap)
    ap.add_argument("--until", type=int, metavar="STEP", help="stop after this step with a checkpoint")
    run("train", body, ap)
