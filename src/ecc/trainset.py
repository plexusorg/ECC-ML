"""uv run trainset: classifier training and validation sets in <work>/training/train.parquet and <work>/training/val.parquet.

Only sources with `train = true` are used; the current Total Freedom logs, the eval case files and the red-team cases
are never trained on. Each example is one message with up to WINDOW_MAX - 1 earlier messages of its conversation (the
most a classifier prompt holds), one 'speaker: text' line each. Labels:
ban -> 1, ok -> 0, amb dropped. `outside` ok rows from outside every candidate window are sampled. A stable hash of the
conversation sends `val_percent` of conversations to validation.
The `extra` case files (written conversations, such as synthetic grooming approaches) go to the training split only.
Their labelled lines are rows numbered on from the end of messages.parquet, in extra_cases order; source is the file's
name and conv is "<file name>:<conversation id>".
"""
import hashlib
import random
from collections import defaultdict, deque
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .chat import WINDOW_MAX, files_in, load_cases
from .core import log, progress, run, save_state, write_parquet


def extra_files(cfg):
    """The case files of the [trainset] extra folders."""
    return files_in([cfg.path(f) for f in cfg["trainset"]["extra"]], ".txt")


def extra_cases(cfg):
    """Every line of the `extra` case files, in file order, with conv "<file name>:<conversation id>"."""
    return [{**it, "conv": f"{Path(p).stem}:{it['conv']}", "source": Path(p).stem}
            for p in extra_files(cfg) for it in load_cases([p])]


def body(cfg, args):
    c = cfg["trainset"]
    data = cfg.data
    sources = {s["name"] for s in cfg["ingest"]["sources"] if s.get("train")}
    m = pq.read_table(data / "messages.parquet", columns=["source", "conv", "speaker", "text"]).to_pydict()
    lab = pq.read_table(data / "labels.parquet", columns=["row", "label"]).to_pydict()
    label_of = dict(zip(lab["row"], lab["label"]))
    n = len(m["text"])
    outside = [i for i in range(n) if m["source"][i] in sources and i not in label_of]
    keep_outside = set(random.Random(5).sample(outside, min(c["outside"], len(outside))))
    history = defaultdict(lambda: deque(maxlen=WINDOW_MAX - 1))
    cols = ("context", "speaker", "text", "label", "source", "conv", "row")
    out = {"train": {k: [] for k in cols}, "val": {k: [] for k in cols}}
    with progress("examples", n, "rows") as advance:
        for i in range(n):
            if i % 50000 == 49999:
                advance(50000)
            src, conv = m["source"][i], m["conv"][i]
            if src not in sources:
                continue
            hist = history[conv]
            lbl = label_of.get(i, "ok" if i in keep_outside else None)
            if lbl in ("ban", "ok"):
                is_val = int(hashlib.sha1(conv.encode()).hexdigest(), 16) % 100 < c["val_percent"]
                split = out["val" if is_val else "train"]
                split["context"].append("\n".join(f"{s}: {t}" for s, t in hist))
                split["speaker"].append(m["speaker"][i])
                split["text"].append(m["text"][i])
                split["label"].append(1 if lbl == "ban" else 0)
                split["source"].append(src)
                split["conv"].append(conv)
                split["row"].append(i)
            hist.append((m["speaker"][i], m["text"][i]))
        advance(n % 50000)
    split = out["train"]
    for k, it in enumerate(extra_cases(cfg)):
        if it["label"] in ("ban", "ok"):
            split["context"].append("\n".join(f"{s}: {t}" for s, t in it["history"][-(WINDOW_MAX - 1):]))
            split["speaker"].append(it["speaker"])
            split["text"].append(it["text"])
            split["label"].append(1 if it["label"] == "ban" else 0)
            split["source"].append(it["source"])
            split["conv"].append(it["conv"])
            split["row"].append(n + k)
    counts = {}
    for name, split in out.items():
        table = pa.table(split)
        path = cfg.work / "training" / f"{name}.parquet"
        unchanged = path.exists() and pq.read_table(path).equals(table)
        if not unchanged:   # an unchanged file keeps its fingerprint, so what was made from it stays current
            write_parquet(table, path)
        pos = sum(split["label"])
        counts[name] = [len(split["label"]), pos]
        log.info("%s: %s examples, %s ban (%.1f%%)%s", name, f"{len(split['label']):,}", f"{pos:,}",
                 100 * pos / max(len(split["label"]), 1), f"; {name}.parquet unchanged" if unchanged else "")
    save_state(cfg, "trainset", [data / "messages.parquet", data / "labels.parquet", *extra_files(cfg)],
               counts=counts, history=WINDOW_MAX - 1)


def main():
    run("trainset", body)
