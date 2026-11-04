"""uv run candidates: risk-term windows over data/messages.parquet, and Codex labeling shards for new windows.

data/windows.jsonl only grows: existing windows, their ids and their rows never change, so their labels stay valid.
Every risk-term hit (chat.terms, data/labeling/terms.toml) outside every existing window gets a new window, so new messages and new risk terms
both add only new windows. So does every row listed in data/audit/rows.jsonl (one {"row": n} per line), such as the
outside-window rows the classifier scores high (`uv run eval --set outside`). A new window may share context rows with
an existing one; merge keeps the existing window's label for those rows. Rows outside every window are ok by construction.
The first shards are a seeded shuffle with a share of windows labelled twice; later windows get sequential shards.
"""
import json
import random
from collections import defaultdict

import pyarrow.parquet as pq

from .chat import render_window, terms
from .core import log, progress, read_jsonl, run, save_state, write_text


def risk_hits(risk, texts):
    hits = set()
    with progress("risk terms", len(texts), "rows") as advance:
        for i, text in enumerate(texts):
            if risk.search(text or ""):
                hits.add(i)
            if i % 10000 == 9999:
                advance(10000)
        advance(len(texts) % 10000)
    return hits


def compute_windows(t, hits, first_id, before, after, max_window):
    """Windows around the hit rows, numbered from first_id: each hit with `before` earlier and `after` later rows of its
    conversation, overlapping spans merged, spans over max_window split."""
    by_conv = defaultdict(list)
    for i, conv in enumerate(t["conv"]):
        by_conv[conv].append(i)
    windows = []
    for conv, rows in by_conv.items():
        merged = []
        for k, r in enumerate(rows):
            if r in hits:
                s, e = max(0, k - before), min(len(rows), k + after + 1)
                if merged and s <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], e)
                else:
                    merged.append([s, e])
        for s, e in merged:
            chunks = []
            while e - s > max_window:
                chunks.append((s, s + max_window))
                s += max_window - before  # overlap so a split never loses a hit's context
            chunks.append((s, e))
            for cs, ce in chunks:
                seg = rows[cs:ce]
                windows.append({"id": first_id + len(windows), "source": t["source"][seg[0]], "conv": conv,
                                "channel": t["channel"][seg[0]],
                                "lines": [[r, t["speaker"][r], t["text"][r]] for r in seg],
                                "hits": [r for r in seg if r in hits]})
    return windows


def existing_windows(t, path):
    """The windows in windows.jsonl, after checking each line still matches its row in messages.parquet."""
    windows = []
    if not path.exists():
        return windows
    with open(path, encoding="utf-8") as f:
        for k, line in enumerate(f):
            w = json.loads(line)
            if w["id"] != k:
                raise SystemExit(f"windows.jsonl line {k + 1} holds window {w['id']}")
            for r, speaker, text in w["lines"]:
                if r >= len(t["text"]) or [t["speaker"][r], t["text"][r]] != [speaker, text]:
                    raise SystemExit(f"window {k} row {r} differs from messages.parquet; the messages changed under "
                                     "existing labels")
            windows.append(w)
    return windows


def new_windows(cfg, t, windows):
    """(windows around the risk-term hits and audit rows outside every existing window, number of risk hits in all)."""
    c = cfg["candidates"]
    hits = risk_hits(terms(cfg)["risk"], t["text"])
    covered = {r for w in windows for r, _, _ in w["lines"]}
    audit = {r["row"] for r in read_jsonl(cfg.data / "audit" / "rows.jsonl")}
    if audit:
        log.info("%s audit rows, %s of them outside every window", f"{len(audit):,}", f"{len(audit - covered):,}")
    return (compute_windows(t, (hits | audit) - covered, len(windows), c["before"], c["after"], c["max_window"]),
            len(hits))


def first_shards(windows, size, overlap):
    rnd = random.Random(11)
    order = list(range(len(windows)))
    rnd.shuffle(order)
    shards = [order[i:i + size] for i in range(0, len(order), size)]
    position = {wid: k for k, wid in enumerate(order)}
    dup = rnd.sample(order, int(len(order) * overlap))
    for k, wid in enumerate(dup):
        target = (position[wid] // size + 1 + k % (len(shards) - 1)) % len(shards)
        shards[target].append(wid)
    return [[windows[i]["id"] for i in ids] for ids in shards]


def body(cfg, args):
    c = cfg["candidates"]
    data = cfg.data
    messages = data / "messages.parquet"
    if not messages.exists():
        raise SystemExit("no messages.parquet yet; run `uv run ingest` first")
    t = pq.read_table(messages, columns=["source", "conv", "channel", "speaker", "text"]).to_pydict()
    n = len(t["text"])
    path = data / "windows.jsonl"
    windows = existing_windows(t, path)
    new, hits = new_windows(cfg, t, windows)
    if new:
        with open(path, "a", encoding="utf-8") as f:
            f.write("".join(json.dumps(w, ensure_ascii=False) + "\n" for w in new))
        windows += new
    covered = len({r for w in windows for r, _, _ in w["lines"]})
    log.info("%s rows, %s risk hits, %s windows covering %s rows (%.1f%%); %s new windows (%s lines)", f"{n:,}",
             f"{hits:,}", f"{len(windows):,}", f"{covered:,}", 100 * covered / max(n, 1), f"{len(new):,}",
             f"{sum(len(w['lines']) for w in new):,}")

    label = data / "label"
    index_path = label / "shards.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
    sharded = {wid for ids in index.values() for wid in ids}
    todo = [w for w in windows if w["id"] not in sharded]
    if todo:
        by_id = {w["id"]: w for w in windows}
        if not index:
            groups = first_shards(todo, c["shard_size"], c["overlap"])
        else:
            groups = [[w["id"] for w in todo[k:k + c["shard_size"]]] for k in range(0, len(todo), c["shard_size"])]
        n = max((int(k.split("_")[1]) for k in index), default=-1) + 1
        for ids in groups:
            name = f"shard_{n:04d}"
            write_text(label / "shards" / f"{name}.txt", "\n\n".join(render_window(by_id[i]) for i in ids) + "\n")
            index[name] = ids
            n += 1
        write_text(index_path, json.dumps(index))
        log.info("%d new shards (%d in all)", len(groups), len(index))
    else:
        log.info("shards up to date: %d shards", len(index))
    save_state(cfg, "candidates", [messages], windows=len(windows), shards=len(index))


def main():
    run("candidates", body)
