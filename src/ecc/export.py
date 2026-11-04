"""uv run export [--model NAME]: the trained adapter in <work>/models/<NAME>/ to the GGUF that eval and the model host use.

Each output records what it was made from next to it (<output>.inputs.json): the fingerprints of its input files and
tools, and the settings and prompt format it used. A stage whose output was made from exactly the current inputs is
skipped, so a rerun continues after the last finished stage; any other output is made again, and so is everything
made from it.
1. merge (merged/): merges the adapter into the base model. The margins of the reloaded adapter and of the merged
   model on the 64 check prompts from training must stay within [export] merge_tolerance of the margins recorded at
   the end of training (merged/merge_check.json).
2. gguf (gguf/model-bf16.gguf, imatrix.gguf, model-<deploy>.gguf): llama.cpp's converter writes bf16. llama-imatrix
   computes an importance matrix on it from `imatrix_prompts` training-split prompts (imatrix.txt: a quarter ban, each
   the classifier prompt with the earlier lines the streaming window would hold at a random point of its cycle, then
   its answer token). llama-imatrix reads that text as one token stream in fixed 512-token chunks (-c 512), which run
   across prompt boundaries.
   llama-quantize makes the [export] deploy type with that imatrix and the tied token embedding, which is also the
   output head, at Q8_0. For Q8_0 both are no-ops: quantize_q8_0 ignores the imatrix and Q8_0 keeps the head at Q8_0.
3. parity (eval/parity-<deploy>.json): margins from llama-server ([eval] server, with the deployment's settings) on the
   deployed GGUF against PyTorch on `parity_prompts` val prompts, half ban and half ok, each with the history its
   streaming window holds.
4. config, on every run: data/classifier.toml gets [models.classifier] gguf (the [export] deploy quantization) and the
   tokenizer of the base model in train_done.json's settings (eval counts the streaming window with it). It overrides
   ecc.toml; local.toml overrides it. `uv run eval` then scores that GGUF and adds the Platt parameters it fits on
   the GGUF's val margins, so calibration and thresholds come from the file that ships.
The merge and parity's PyTorch side run on the GPU when there is one.

A bounded run (`uv run export --steps N ...`, the flags of its `uv run train`) has no GGUF. It gets a PyTorch eval
(eval/torch-stream/), from its adapter on the base model without merging, after the reloaded adapter's margins pass
the same check: margins on val, every [eval.sets] set and tf, each message with the history its streaming window
holds (a window's messages from one forward pass, as `uv run eval` scores them), summarized as `uv run eval` does, with val also on its rows whose text never appears in train, and a Platt fit
on val (platt-stream.json: a, b and the margins at val false-positive rates of 0.1%, 0.5% and 1%). A set's scores are
kept while they were made from the current model, set files (for messages.parquet, its ingest batches) and prompt
format; messages ingested since are scored and appended. It writes metrics.json, the numbers runs are compared by, and
<work>/models/bounded/table.md is rebuilt with one row per evaluated bounded run.
"""
import argparse
import json
import random
import subprocess
import sys
from collections import defaultdict

from .chat import PROMPT_FORMAT, WINDOW_MIN, fitted_prompt, split_context, window_sizes
from .core import JsonlAppender, fingerprint, is_current, log, progress, read_jsonl, record_inputs, run, write_text
from .evaluate import (FPRS, Server, case_summary, classifier_margin, current_scores, fit_platt, items_of,
                       known_threshold, message_stats, record, scorable, scorable_labels, server_inputs, set_source,
                       start_scores, tf_summary, write_classifier)
from .train import run_args, run_dir

# Scores (eval/torch-<prompt format>/) and Platt fit of the classifier's prompt format; write_table lists every format
# a bounded run has scores for.
EVAL = "torch-stream"
PLATT = "platt-stream.json"


def outputs(cfg, out):
    """Each stage's output."""
    deploy = cfg["export"]["deploy"]
    return {"merge": out / "merged" / "merge_check.json", "bf16": out / "gguf" / "model-bf16.gguf",
            "imatrix": out / "gguf" / "imatrix.gguf", "gguf": out / "gguf" / f"model-{deploy}.gguf",
            "parity": out / "eval" / f"parity-{deploy}.json"}


def stage_inputs(cfg, out, stage):
    """What a full run's stage output is made from."""
    e, files = cfg["export"], outputs(cfg, out)
    if stage == "merge":
        return {"adapter": fingerprint(out / "adapter"), "train_done": fingerprint(out / "train_done.json")}
    if stage == "bf16":
        return {"merged": fingerprint(out / "merged"), "converter": fingerprint(cfg.path(e["converter"]))}
    if stage == "imatrix":
        # its prompts are sampled from train.parquet, but the bf16 fingerprint already pins the run
        return {"bf16": fingerprint(files["bf16"]), "tool": fingerprint(cfg.path(e["imatrix"])),
                "prompts": e["imatrix_prompts"], "prompt": PROMPT_FORMAT}
    if stage == "gguf":
        return {"bf16": fingerprint(files["bf16"]), "imatrix": fingerprint(files["imatrix"]),
                "tool": fingerprint(cfg.path(e["quantize"])), "type": e["deploy"]}
    return {"gguf": fingerprint(files["gguf"]), "merged": fingerprint(out / "merged"),
            **server_inputs(cfg, "classifier"), **set_source(cfg, "val"), "prompt": PROMPT_FORMAT,
            "n": e["parity_prompts"]}


def model_inputs(out, steps):
    """What the scored model is made from: merged/, or for a bounded run the adapter and train_done.json."""
    if steps:
        return {"adapter": fingerprint(out / "adapter"), "train_done": fingerprint(out / "train_done.json")}
    return {"merged": fingerprint(out / "merged")}


def torch_set_inputs(cfg, set_name, model_from):
    return {**set_source(cfg, set_name), "model": model_from, "prompt": PROMPT_FORMAT}


def torch_sets(cfg):
    return ["val", *cfg["eval"]["sets"], "tf"]


def stages_done(cfg, out):
    """{stage: whether `export` would leave it as it is} of a full run, in stage order."""
    files = outputs(cfg, out)
    done = {"merge": is_current(files["merge"], stage_inputs(cfg, out, "merge"))}
    for stage, after in (("bf16", "merge"), ("imatrix", "bf16"), ("gguf", "imatrix")):
        done[stage] = done[after] and is_current(files[stage], stage_inputs(cfg, out, stage))
    done["parity"] = done["gguf"] and is_current(files["parity"], stage_inputs(cfg, out, "parity"))
    return done


def load_merged(out):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .classifier import device
    model = AutoModelForCausalLM.from_pretrained(out / "merged", dtype=torch.bfloat16, attn_implementation="sdpa")
    return model.to(device()).eval(), AutoTokenizer.from_pretrained(out / "merged")


def load_adapter(out, done):
    """The base model with the run's adapter, unmerged, its tokenizer, and the largest change of the margins on the
    check prompts from those recorded at the end of training."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .classifier import device, torch_margins
    tok = AutoTokenizer.from_pretrained(out / "adapter")
    base = AutoModelForCausalLM.from_pretrained(done["model"], dtype=torch.bfloat16, attn_implementation="sdpa")
    peft = PeftModel.from_pretrained(base.to(device()), out / "adapter")
    peft.eval()
    reloaded = torch_margins(peft.base_model.model, tok, done["check_prompts"])
    return peft, tok, max(abs(x - y) for x, y in zip(reloaded, done["check_margins"]))


def merge(cfg, out):
    from .classifier import device, torch_margins
    done = json.loads((out / "train_done.json").read_text(encoding="utf-8"))
    dev = device()
    log.info("merge: %s + %s on the %s", done["model"], out / "adapter", dev)
    peft, tok, reload_diff = load_adapter(out, done)
    merged = peft.merge_and_unload()
    merged_m = torch_margins(merged, tok, done["check_prompts"])
    check = {"reload_max_diff": reload_diff,
             "merge_max_diff": max(abs(x - y) for x, y in zip(merged_m, done["check_margins"])), "device": dev}
    log.info("merge check over %d prompts: reloaded adapter max |margin diff| %.4f, merged %.4f",
             len(merged_m), check["reload_max_diff"], check["merge_max_diff"])
    tolerance = cfg["export"]["merge_tolerance"]
    if max(check["reload_max_diff"], check["merge_max_diff"]) > tolerance:
        raise SystemExit(f"the merged model's margins differ from training's by more than {tolerance}; "
                         f"check {out / 'adapter'} and its base model")
    merged.save_pretrained(out / "merged")
    tok.save_pretrained(out / "merged")
    write_text(out / "merged" / "merge_check.json", json.dumps(check))


def tool(args, what):
    """Runs a llama.cpp tool; its output goes to the log file."""
    p = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace")
    log.debug("%s output:\n%s", what, p.stdout + p.stderr)
    if p.returncode:
        raise SystemExit(f"{what} failed with exit code {p.returncode}:\n{(p.stdout + p.stderr)[-2000:]}")


def imatrix_text(cfg, out, path):
    """llama-imatrix's input: `imatrix_prompts` training-split prompts, a quarter ban, separated by blank lines.
    llama-imatrix tokenizes the whole file as one stream and processes it in fixed chunks of -c tokens, so a chunk
    can start inside one prompt and end inside the next."""
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    from .classifier import token_count
    n = cfg["export"]["imatrix_prompts"]
    t = pq.read_table(cfg.work / "training" / "train.parquet", columns=["context", "speaker", "text", "label"]).to_pydict()
    ban = [i for i, y in enumerate(t["label"]) if y]
    ok = [i for i, y in enumerate(t["label"]) if not y]
    rng = random.Random(12)
    chosen = rng.sample(ban, n // 4) + rng.sample(ok, n - n // 4)
    rng.shuffle(chosen)
    count = token_count(AutoTokenizer.from_pretrained(out / "merged"))
    prompts = []
    for i in chosen:
        # the history the streaming window would give this line at a random point of its cycle, so the importance
        # matrix sees deployment prompts
        lines = split_context(t["context"][i]) + [(t["speaker"][i], t["text"][i])]
        lines = lines[rng.randrange(min(WINDOW_MIN, len(lines))):]
        n = window_sizes(lines, count)[-1]
        prompts.append(fitted_prompt(lines[len(lines) - 1 - n:-1], t["speaker"][i], t["text"][i], count)
                       + (" ban" if t["label"][i] else " ok"))
    write_text(path, "\n\n".join(prompts) + "\n")


def gguf(cfg, out):
    e, files = cfg["export"], outputs(cfg, out)
    folder = out / "gguf"
    folder.mkdir(parents=True, exist_ok=True)
    bf16 = files["bf16"]
    made_from = stage_inputs(cfg, out, "bf16")
    if is_current(bf16, made_from):
        log.info("gguf: %s is current", bf16.name)
    else:
        log.info("gguf: converting %s to bf16", out / "merged")
        tmp = folder / "model-bf16.tmp.gguf"
        tool([sys.executable, str(cfg.path(e["converter"])), str(out / "merged"), "--outfile", str(tmp),
              "--outtype", "bf16"], "convert_hf_to_gguf")
        tmp.replace(bf16)
        record_inputs(bf16, made_from)
    imatrix = files["imatrix"]
    made_from = stage_inputs(cfg, out, "imatrix")
    if is_current(imatrix, made_from):
        log.info("gguf: %s is current", imatrix.name)
    else:
        text = folder / "imatrix.txt"
        imatrix_text(cfg, out, text)
        log.info("gguf: importance matrix from %d training prompts", e["imatrix_prompts"])
        tmp = folder / "imatrix.tmp.gguf"
        tool([str(cfg.path(e["imatrix"])), "-m", str(bf16), "-f", str(text), "-o", str(tmp), "-c", "512", "-b", "512",
              "--no-ppl"], "llama-imatrix")
        tmp.replace(imatrix)
        record_inputs(imatrix, made_from)
    target = files["gguf"]
    made_from = stage_inputs(cfg, out, "gguf")
    if is_current(target, made_from):
        log.info("gguf: %s is current", target.name)
        return
    log.info("gguf: quantizing to %s", e["deploy"])
    tmp = folder / f"model-{e['deploy']}.tmp.gguf"
    tool([str(cfg.path(e["quantize"])), "--imatrix", str(imatrix), "--token-embedding-type", "q8_0", str(bf16),
          str(tmp), e["deploy"]], "llama-quantize")
    tmp.replace(target)
    record_inputs(target, made_from)
    for f in sorted(folder.glob("*.gguf")):
        log.info("%s %s MiB", f.name, f"{f.stat().st_size / 2**20:,.0f}")


def torch_eval(cfg, out, load, model_from):
    """load(): the (model, tokenizer) to score with, called only when a set has unscored messages. model_from: the
    fingerprints of what that model is made from. A set's scores made from other inputs are deleted; messages added
    since (new ingest batches) are scored and appended."""
    from .classifier import token_count, torch_margins
    folder = out / "eval" / EVAL
    loaded = []
    summary, platt, cut, metrics = [f"# {out.name}: PyTorch margins, streaming prompts"], None, None, {}
    for set_name in torch_sets(cfg):
        path = folder / f"{set_name}.jsonl"
        made_from = torch_set_inputs(cfg, set_name, model_from)
        start_scores(path, made_from)
        path.touch()
        done = {r["k"] for r in read_jsonl(path, repair=True)}
        if any(k not in done for k in scorable_labels(cfg, set_name)):
            if not loaded:
                loaded.append(load())
            model, tok = loaded[0]
            count = token_count(tok)
            convs = defaultdict(list)
            for k, it in scorable(items_of(cfg, set_name, count)):
                if k not in done:
                    convs[it["conv"]].append((k, it))
            # in conversation order, so a streaming window's messages share one forward pass (torch_margins)
            items = [x for conv in convs.values() for x in conv]
            prompts = [fitted_prompt(it["history"], it["speaker"], it["text"], count) for _, it in items]
            with progress(f"torch {set_name}", len(prompts), "msgs") as advance:
                margins = torch_margins(model, tok, prompts, advance)
            with JsonlAppender(path) as w:
                for (k, it), m in zip(items, margins):
                    w.write(record(k, it, {"margin": m}))
        rows = current_scores(cfg, path, made_from, set_name)[0]
        lines = []
        if set_name == "val":
            fitted, val, lines = fit_platt(cfg, rows)
            metrics.update(val)
            write_text(out / PLATT, json.dumps(fitted, indent=1))
            platt, cut = (fitted["a"], fitted["b"]), fitted["margin@fpr0.01"]
        if set_name == "tf":
            lines += tf_summary(cfg, rows, platt)
            metrics["tf"] = known_threshold(cfg, rows, platt)
        else:
            lines += case_summary(rows, platt, cut)
            if set_name != "val":
                metrics[set_name] = message_stats(rows, platt, cut)
        summary += ["", f"## {set_name}", *lines]
        for line in lines[:2]:
            log.info("torch %s: %s", set_name, line)
    write_text(folder / "summary.md", "\n".join(summary) + "\n")
    write_text(folder / "metrics.json", json.dumps(metrics, indent=1))
    log.info("full summary: %s", folder / "summary.md")


def parity(cfg, out):
    import numpy as np

    from .classifier import token_count, torch_margins
    from .evaluate import auroc
    n = cfg["export"]["parity_prompts"]
    model, tok = load_merged(out)
    count = token_count(tok)
    items = scorable(items_of(cfg, "val", count))
    random.Random(2).shuffle(items)
    ban = [it for _, it in items if it["label"] == "ban"][:n // 2]
    picked = ban + [it for _, it in items if it["label"] == "ok"][:n - len(ban)]
    labels = np.array([it["label"] == "ban" for it in picked])
    prompts = [fitted_prompt(it["history"], it["speaker"], it["text"], count) for it in picked]
    ref = np.array(torch_margins(model, tok, prompts))
    del model
    (out / "eval").mkdir(parents=True, exist_ok=True)
    q = cfg["export"]["deploy"]
    with Server(cfg, outputs(cfg, out)["gguf"], out / "eval" / f"parity-{q}.server.log") as server, \
            progress(f"parity {q}", len(prompts), "msgs") as advance:
        ids = server.answer_ids()
        got = []
        for p in prompts:
            got.append(classifier_margin(server, p, ids)[0])
            advance()
    unlisted = sum(m is None for m in got)
    if unlisted:
        raise SystemExit(f"parity: llama-server listed neither answer token in its top 20 for {unlisted} of "
                         f"{len(got)} val prompts, so their margins are unknown; check {outputs(cfg, out)['gguf']} "
                         f"and {out / 'eval' / f'parity-{q}.server.log'}")
    g = np.array(got, dtype=float)
    # llama-server reports the top 20 tokens only; a margin past that floor is clipped, so compare within +-10
    inside = (np.abs(ref) < 10) & (np.abs(g) < 10)
    d = np.abs(ref - g)
    r = {"n": len(prompts), "pearson": float(np.corrcoef(ref, g)[0, 1]), "mean_abs_diff": float(d.mean()),
         "mean_abs_diff_within10": float(d[inside].mean()),
         "p95_abs_diff_within10": float(np.percentile(d[inside], 95)), "max_abs_diff": float(d.max()),
         "sign_agreement": float(np.mean(np.sign(ref) == np.sign(g))),
         "auroc_torch": auroc(ref[labels].tolist(), ref[~labels].tolist()),
         "auroc_gguf": auroc(g[labels].tolist(), g[~labels].tolist())}
    log.info("parity %s over %d prompts: Pearson %.4f, mean |diff| %.3f, max %.3f, sign agreement %.1f%%", q,
             r["n"], r["pearson"], r["mean_abs_diff"], r["max_abs_diff"], 100 * r["sign_agreement"])
    write_text(outputs(cfg, out)["parity"], json.dumps(r, indent=1))


def write_table(cfg):
    """The bounded runs side by side. The tf column is recomputed from each run's tf scores with the current known
    offenders: the players the highest threshold that still triggers every known offender found also triggers, not
    counting those offenders. The threshold is picked on tf itself, so the count is in-sample."""
    root = cfg.work / "models" / "bounded"
    sets = list(cfg["eval"]["sets"])
    head = ["run", "prompt", "steps", "train min", "peak GiB", "val AUROC", "PR-AUC",
            *[f"recall@{f:.1%}" for f in FPRS], "val AUROC, novel text", *[f"{s} AUROC, bans at 1%" for s in sets],
            "tf other players at all known (in-sample)"]
    rows = []
    for path in sorted(root.glob("*/eval/torch-*/metrics.json")):
        run_folder = path.parents[2]
        if not (run_folder / "train_done.json").exists():
            continue
        fmt = path.parent.name.removeprefix("torch-")
        d = json.loads((run_folder / "train_done.json").read_text(encoding="utf-8"))
        m = json.loads(path.read_text(encoding="utf-8"))
        platt_path = run_folder / f"platt-{fmt}.json"
        p = json.loads(platt_path.read_text(encoding="utf-8")) if platt_path.exists() else None
        kt = known_threshold(cfg, read_jsonl(path.parent / "tf.jsonl"), (p["a"], p["b"]) if p else None)
        novel = m.get("val_novel")
        rows.append([run_folder.name, fmt, f"{d['steps']:,}",
                     f"{d['train_minutes']:.1f}", f"{d['peak_reserved_gib']:.1f}",
                     f"{m['val']['auroc']:.4f}", f"{m['val']['pr_auc']:.3f}",
                     *[f"{m['val'][f'recall@fpr{f}']:.3f}" for f in FPRS],
                     f"{novel['auroc']:.4f}" if novel else "-",
                     *[f"{m[s]['auroc']:.3f}, {m[s]['caught']}/{m[s]['ban']}" for s in sets],
                     f"{kt['players'] - kt['known']} ({kt['known']} known)" if kt else "-"])
    write_text(root / "table.md", "\n".join("| " + " | ".join(r) + " |"
                                            for r in [head, ["---"] * len(head), *rows]) + "\n")
    log.info("comparison of %d bounded runs: %s", len(rows), root / "table.md")


def body(cfg, args):
    _, _, steps, out = run_dir(cfg, args)
    done_path = out / "train_done.json"
    if not done_path.exists():
        raise SystemExit(f"{out.name} is not trained; run `uv run train` with the same flags first")
    if steps:
        def load():
            done = json.loads(done_path.read_text(encoding="utf-8"))
            peft, tok, diff = load_adapter(out, done)
            log.info("%s reloaded: max |margin diff| %.4f on training's %d check prompts", out / "adapter", diff,
                     len(done["check_prompts"]))
            if diff > cfg["export"]["merge_tolerance"]:
                raise SystemExit(f"the reloaded adapter's margins differ from training's by more than "
                                 f"{cfg['export']['merge_tolerance']}; check {out / 'adapter'} and its base model")
            return peft.base_model.model, tok
        torch_eval(cfg, out, load, model_inputs(out, steps))
        write_table(cfg)
        return
    base = json.loads(done_path.read_text(encoding="utf-8")).get("settings", {}).get("base")
    if base is None:
        raise SystemExit(f"{done_path} has no training settings; run `uv run train` first")
    stage = outputs(cfg, out)
    made_from = stage_inputs(cfg, out, "merge")
    if is_current(stage["merge"], made_from):
        log.info("merge: %s is current", out / "merged")
    else:
        merge(cfg, out)
        record_inputs(stage["merge"], made_from)
    gguf(cfg, out)
    made_from = stage_inputs(cfg, out, "parity")
    if is_current(stage["parity"], made_from):
        log.info("parity: %s is current", stage["parity"])
    else:
        parity(cfg, out)
        record_inputs(stage["parity"], made_from)
    gguf_path = f"{{work}}/models/{out.name}/gguf/model-{cfg['export']['deploy']}.gguf"
    write_classifier(cfg, gguf_path, base)
    log.info("next: `uv run eval` scores %s on val (Platt and thresholds), the case sets and tf", gguf_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    run_args(ap)
    run("export", body, ap)
