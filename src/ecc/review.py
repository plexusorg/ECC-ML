"""uv run review [--model NAME]: the confident-learning queue, labelled lines a trained model strongly disagrees with.

Reads the model's scores on training rows, <work>/eval/<model>/train.jsonl (`uv run eval --model NAME --set train`
scores the labelled training rows of `[review] sample_windows` random windows; every row of the current sample must
have a current score), and data/labels.parquet. Only
training rows are queued: relabelling val or tf where the model disagrees would make the evaluation agree with the
model. Confidence is the model's P(ban), Platt-calibrated when the model has Platt parameters. Per class, the threshold is
the model's mean confidence on lines carrying that label; a line whose model confidence
for the other class reaches that threshold is queued, strongest disagreement first. Writes <work>/review/queue.txt.

To correct a line, write ban, ok or amb after "fix:" in queue.txt and run `uv run review` again: filled-in fixes move
to data/corrections.jsonl (a later fix of the same row wins) and leave the queue. `uv run merge` applies them.
"""
import argparse
import json
import re
from collections import defaultdict, deque

import pyarrow.parquet as pq

from .core import log, read_jsonl, run, save_state, write_text
from .evaluate import score, set_scores

ITEM = re.compile(r"^### row (\d+)", re.M)
FIX = re.compile(r"^fix:[ \t]*(\S*)", re.M)


def harvest(queue, corrections):
    """Fixes filled in on the queue file -> appended to corrections.jsonl. Returns the number taken."""
    if not queue.exists():
        return 0
    text = queue.read_text(encoding="utf-8")
    starts = [(m.start(), int(m.group(1))) for m in ITEM.finditer(text)]
    taken = []
    for k, (pos, row) in enumerate(starts):
        block = text[pos:starts[k + 1][0] if k + 1 < len(starts) else len(text)]
        m = FIX.search(block)
        value = m.group(1).lower() if m else ""
        if value in ("ban", "ok", "amb"):
            taken.append({"row": row, "label": value, "rule": None, "source": "review"})
        elif value:
            raise SystemExit(f"row {row}: fix must be ban, ok or amb, not {value!r}")
    if taken:
        with open(corrections, "a", encoding="utf-8") as f:
            f.write("".join(json.dumps(c) + "\n" for c in taken))
    return len(taken)


def body(cfg, args):
    name = args.model or cfg["review"]["model"]
    data = cfg.data
    queue = cfg.work / "review" / "queue.txt"
    corrections = data / "corrections.jsonl"
    taken = harvest(queue, corrections)
    if taken:
        log.info("%d fixes moved to corrections.jsonl; run `uv run merge` to apply them", taken)
    corrected = {c["row"] for c in read_jsonl(corrections)}

    platt = cfg["models"][name].get("platt")
    path = cfg.work / "eval" / name / "train.jsonl"
    current = set_scores(cfg, name, "train")
    if current is None or current[1]:
        raise SystemExit(f"{name}'s scores on the training sample are "
                         + (f"incomplete ({current[1]:,} rows unscored)" if current else "missing or from other inputs")
                         + f"; run `uv run eval --model {name} --set train` first")
    scores = {r["k"]: score(r, platt) for r in current[0] if r.get("margin") is not None}
    lab = pq.read_table(data / "labels.parquet").to_pydict()
    labelled = [(r, l, rule) for r, l, rule in zip(lab["row"], lab["label"], lab["rule"])
                if l in ("ban", "ok") and r in scores and r not in corrected]
    ban_p = [scores[r] for r, l, _ in labelled if l == "ban"]
    ok_p = [1 - scores[r] for r, l, _ in labelled if l == "ok"]
    t_ban = sum(ban_p) / max(len(ban_p), 1)
    t_ok = sum(ok_p) / max(len(ok_p), 1)
    flagged = []
    for r, l, rule in labelled:
        p = scores[r]
        if l == "ok" and p >= t_ban:
            flagged.append((p, r, l, rule))
        elif l == "ban" and 1 - p >= t_ok:
            flagged.append((1 - p, r, l, rule))
    flagged.sort(reverse=True)
    flagged = flagged[:cfg["review"]["size"]]

    m = pq.read_table(data / "messages.parquet", columns=["conv", "speaker", "text"]).to_pydict()
    want = {r for _, r, _, _ in flagged}
    context = {}
    hist = defaultdict(lambda: deque(maxlen=6))
    last = max(want, default=-1)
    for i in range(last + 1):
        h = hist[m["conv"][i]]
        if i in want:
            context[i] = list(h)
        h.append(f"{m['speaker'][i]}: {m['text'][i]}")
    out = [f"# Review queue for {name}: {len(flagged)} labelled lines the model strongly disagrees with.",
           "# Write ban, ok or amb after fix: to correct a label, leave it empty to keep it, then run `uv run review`.",
           ""]
    for conf, r, l, rule in flagged:
        out += [f"### row {r} | labelled {l}{f' ({rule})' if rule else ''} | model {'ban' if l == 'ok' else 'ok'} "
                f"{conf:.3f}", *[f"    {c}" for c in context[r]], f">   {m['speaker'][r]}: {m['text'][r]}", "fix:", ""]
    write_text(queue, "\n".join(out))
    log.info("%d labelled lines scored; thresholds ban %.3f, ok %.3f; %d queued (labelled ok: %d, labelled ban: %d) "
             "-> %s", len(labelled), t_ban, t_ok, len(flagged), sum(1 for f in flagged if f[2] == "ok"),
             sum(1 for f in flagged if f[2] == "ban"), queue)
    save_state(cfg, "review", [data / "labels.parquet", path], model=name, queued=len(flagged),
               corrections=len(corrected))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", help="a [models.*] entry whose eval scores to use (default: [review] model)")
    run("review", body, ap)
