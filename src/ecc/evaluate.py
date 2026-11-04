"""uv run eval [--model NAME] [--set NAME ...]: score a fast-check model on every evaluation set and summarize.

Sets: the case files in [eval.sets], "val" (<work>/training/val.parquet) and "tf" (every message of the current Total Freedom
logs, never trained on). "train" (labelled training rows of `[review] sample_windows` random windows, for `uv run
review`) and "outside" (every messages.parquet row outside every window of data/windows.jsonl, of every source: ok
only because no risk term opened a window) are scored only when named with --set. Scores are appended to <work>/eval/<model>/<set>.jsonl one message at a
time, so a stopped or killed run continues where it stopped. <set>.jsonl.inputs.json records what the margins come
from (the case files, or messages.parquet's ingest batches and for val val.parquet; the GGUF, the llama.cpp build and
its settings, the prompt format and tokenizer, or a chat model's prompt file); a set whose record differs is
scored again from the start. Batches ingested since only add messages, which the resume scores. Case-file messages
are keyed by conversation, text and occurrence, messages.parquet rows by row number.
A classifier sees each message with the earlier lines its conversation's streaming window holds (chat.WINDOW_MIN),
counted with its [models.classifier] tokenizer; val, train, tf and outside messages take their conversation from
messages.parquet, case messages from their case-file conversation. Its margins come from the [eval] server build's
llama.cpp library (ecc.llama), one forward pass per window, and equal llama-server's for each message alone up to
batching's rounding. A chat model is served by a local llama-server started for the run.
Summary: <work>/eval/<model>/summary.md, rebuilt from every set whose scores are current and cover every message of the
set (a partly scored set is named with its unscored count instead) (per-message AUROC/PR-AUC and
player rules on case sets; on val also recall at val FPRs, on all of val and on val rows whose text never appears in
train; triggered players, known offenders and triggers per day on the real log, and its labelled offenders caught at
the operating cut, the margin at val FPR 1%; on outside the messages at each threshold, by source). <work>/eval/<model>/
failures.md is rebuilt with it (ecc.failures). Once the classifier's val is fully
scored, eval fits Platt scaling on those margins and writes it to data/classifier.toml's [models.classifier] platt. A
model with Platt parameters is thresholded on sigmoid(a * margin + b), as the model host does; other models on
sigmoid(margin).
"""
import argparse
import bisect
import hashlib
import json
import math
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict

import pyarrow.parquet as pq

from .chat import CHANNEL_NAMES, PROMPT_FORMAT, WINDOW_MAX, fitted_prompt, load_cases, window_sizes
from .core import (MESSAGES, JsonlAppender, changed_keys, fingerprint, is_current, log, message_batches, progress,
                   read_jsonl, record_inputs, run, sigmoid, write_text)
from .llama import TOP_LOGPROBS

FPRS = (0.001, 0.005, 0.01)
# llama-server settings per model kind. A classifier gets the model host's deployment settings (docs/spec.md,
# section 3) except the slot count, [eval] slots slots of the deployment's 1,024 tokens each. Every layer goes to the
# GPU when the configured build has one (a CPU build ignores -ngl).
SLOT_TOKENS = 1024
# llama.cpp's physical batch when eval scores classifier windows (on the RTX 5090, 512 is 13% slower, 4096 no faster)
WINDOW_UBATCH = 2048


def server_args(cfg, kind):
    if kind == "chat":
        return ["-np", "1", "-c", "8192", "--jinja", "-ngl", "999"]
    slots = cfg["eval"]["slots"]
    return ["-np", str(slots), "-c", str(SLOT_TOKENS * slots), "--slot-prompt-similarity", "0.5", "--cache-ram", "1024",
            "--no-cache-idle-slots", "-ngl", "999"]


def window_settings(cfg):
    """The llama.cpp context eval scores classifier windows in (ecc.llama): [eval] slots windows per decode, each in a
    sequence of the deployment's SLOT_TOKENS tokens, so eval rejects a prompt the deployment would."""
    return {"seqs": cfg["eval"]["slots"], "seq_tokens": SLOT_TOKENS, "ubatch": WINDOW_UBATCH}


# ---------------------------------------------------------------- llama-server

class Server:
    def __init__(self, cfg, gguf, log_path, kind="classifier"):
        e = cfg["eval"]
        port = e["port"]
        self.url = f"http://127.0.0.1:{port}"
        self.args = [str(cfg.path(e["server"])), "-m", str(gguf), "--port", str(port), "-t", str(e["threads"]),
                     "--no-webui", *server_args(cfg, kind)]
        self.log_path = log_path

    def __enter__(self):
        self.logf = open(self.log_path, "w", encoding="utf-8")
        self.proc = subprocess.Popen(self.args, stdout=self.logf, stderr=subprocess.STDOUT)
        start = time.time()
        while time.time() - start < 900:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited with {self.proc.returncode}; see {self.log_path}")
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=2) as r:
                    if r.status == 200:
                        return self
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                pass
            time.sleep(0.5)
        self.__exit__()
        raise RuntimeError("llama-server did not become ready")

    def __exit__(self, *exc):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.logf.close()

    def post(self, path, body):
        req = urllib.request.Request(self.url + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=600) as r:
            data = json.loads(r.read())
        return data, (time.perf_counter() - t0) * 1000

    def answer_ids(self):
        """The token ids of the classifier's answers (" ban", " ok") in the served model's vocabulary."""
        ids = [self.post("/tokenize", {"content": a})[0]["tokens"] for a in (" ban", " ok")]
        if not all(len(i) == 1 for i in ids):
            raise SystemExit(f"the answers are not single tokens in this model: {ids}")
        return ids[0][0], ids[1][0]


def server_inputs(cfg, kind):
    """What llama-server's margins depend on besides the model: its binary and settings (the thread count changes
    only the speed)."""
    return {"server": fingerprint(cfg.path(cfg["eval"]["server"])), "server_args": server_args(cfg, kind)}


def ban_margin(positions):
    """A prompted chat model's log P(ban) - log P(ok): the first position whose top alternatives include a token that
    starts the word "ban" or "ok" (its text can be split or spaced in any way). A word missing from the top
    alternatives is bounded by the lowest listed logprob, so the margin stays finite and keeps ranking where
    probabilities saturate."""
    for pos in positions:
        alts = pos.get("top_logprobs", [])
        best = {}
        for alt in alts:
            tok = alt["token"].strip().lower()
            for w in ("ban", "ok"):
                if tok and w.startswith(tok):
                    best[w] = max(best.get(w, -1e9), alt["logprob"])
        if best:
            floor = min(alt["logprob"] for alt in alts) if alts else -30.0
            return best.get("ban", floor) - best.get("ok", floor)
    return None


def word_probs(positions):
    for pos in positions:
        acc = {"ban": 0.0, "mute": 0.0, "ok": 0.0}
        for alt in pos.get("top_logprobs", []):
            tok = alt["token"].strip().lower()
            for w in acc:
                if tok and w.startswith(tok):
                    acc[w] += math.exp(alt["logprob"])
        total = sum(acc.values())
        if total > 0.5:
            return {w: p / total for w, p in acc.items()}
    return {}


def sections(text):
    parts, current = {}, None
    for line in text.splitlines(keepends=True):
        m = re.match(r"###\s*(system|user)\s*$", line.strip())
        if m:
            current = m.group(1)
            parts[current] = ""
        elif current:
            parts[current] += line
    return {k: v.strip() for k, v in parts.items()}


def classifier_margin(server, prompt, ids):
    """(margin, milliseconds) of a classifier prompt, scored as the model host scores it: log P(" ban") - log P(" ok")
    of the exact answer token ids (Server.answer_ids) at the one predicted position. llama-server lists only the top
    TOP_LOGPROBS tokens. An answer token missing from them gets the lowest listed logprob, an upper bound on its own,
    so the margin stays finite and is at most as extreme as the truth; with neither answer listed the margin is None
    (an error row). ecc.llama.margin applies the same rule to full logits."""
    data, ms = server.post("/completion", {"prompt": prompt, "n_predict": 1, "temperature": 0,
                                           "n_probs": TOP_LOGPROBS, "cache_prompt": True})
    alts = data["completion_probabilities"][0]["top_logprobs"]
    got = {a["id"]: a["logprob"] for a in alts}
    ban, ok = ids
    if ban not in got and ok not in got:
        return None, ms
    floor = min(got.values())
    return got.get(ban, floor) - got.get(ok, floor), ms


def window_scores(cfg, gguf, count_tokens, log_path, convs):
    """Yields (key, item, result record with "margin") of conversations [[(key, item), ...], ...], each message with
    its streaming window's history (items_of with count_tokens), scored by an ecc.llama process that logs to
    log_path."""
    settings = {"build": str(cfg.path(cfg["eval"]["server"]).parent), "gguf": str(gguf), **window_settings(cfg),
                "threads": cfg["eval"]["threads"]}
    items = {k: it for conv in convs for k, it in conv}
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.Popen([sys.executable, "-m", "ecc.llama", json.dumps(settings)], stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=logf, encoding="utf-8")
        failed = []

        def feed():
            # its own thread, so the pipe to the scorer never fills while its results wait to be read
            try:
                for conv in convs:
                    proc.stdin.write(json.dumps([[k, fitted_prompt(it["history"], it["speaker"], it["text"],
                                                                   count_tokens)] for k, it in conv]) + "\n")
            except Exception as e:
                failed.append(e)
            finally:
                try:
                    proc.stdin.close()
                except OSError:
                    pass

        feeder = threading.Thread(target=feed, daemon=True)
        feeder.start()
        finished = False
        try:
            for line in proc.stdout:
                k, margin = json.loads(line)
                yield k, items[k], {"margin": margin}
            finished = True
        finally:
            if not finished:
                proc.kill()
            proc.wait()
            feeder.join()
    if proc.returncode:
        raise RuntimeError(f"the llama.cpp scorer exited with {proc.returncode}; see {log_path}")
    if failed:
        raise failed[0]


class Scorer:
    """A chat model: one message (history, speaker, text, channel) -> result record with "margin", the message seen
    with its last [models.<name>] history lines."""

    def __init__(self, cfg, spec, server):
        self.server = server
        self.n = spec["history"]
        self.sec = sections(cfg.path(spec["prompt"]).read_text(encoding="utf-8"))

    def __call__(self, history, speaker, text, channel):
        hist = list(history)[-self.n:] if self.n > 0 else []
        f = {"history": "\n".join(f"{s}: {t}" for s, t in hist) or "(none)", "speaker": speaker, "text": text,
             "channel": CHANNEL_NAMES[channel]}
        messages = ([{"role": "system", "content": self.sec["system"].format(**f)}] if self.sec.get("system") else [])
        messages.append({"role": "user", "content": self.sec["user"].format(**f)})
        data, ms = self.server.post("/v1/chat/completions", {"messages": messages, "temperature": 0, "logprobs": True,
                                                             "top_logprobs": 20, "max_tokens": 8})
        choice = data["choices"][0]
        positions = (choice.get("logprobs") or {}).get("content", [])
        probs = word_probs(positions)
        return {"verdict": max(probs, key=probs.get) if probs else "error", "p_ban": probs.get("ban"),
                "margin": ban_margin(positions), "raw": choice["message"].get("content") or "", "ms": ms}


# ---------------------------------------------------------------- sets

def earlier_lines(lines, count_tokens):
    """Per line of one conversation, the number of earlier lines its history keeps: the classifier's streaming window
    with count_tokens, else up to WINDOW_MAX - 1 (more than a prompted model's history)."""
    if count_tokens:
        return window_sizes(lines, count_tokens)
    return [min(j, WINDOW_MAX - 1) for j in range(len(lines))]


def case_key(conv, text, occurrence):
    """A case-file message's key: its conversation, its text and which occurrence of that text in the conversation it
    is (0 for the first)."""
    return json.dumps([conv, text, occurrence], ensure_ascii=False)


def case_keys(lines):
    """case_key of each (conv, text), in conversation order."""
    seen = Counter()
    keys = []
    for conv, text in lines:
        keys.append(case_key(conv, text, seen[(conv, text)]))
        seen[(conv, text)] += 1
    return keys


def attack_conversations(cfg):
    """conversations() of the attacks set's code (the .py file in data/cases/attacks/): private like the other case
    sets, and built in code because most attacks use characters a case file cannot hold."""
    path = next((cfg.data / "cases" / "attacks").glob("*.py"))
    namespace = {}
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), namespace)
    return namespace["conversations"]()


def attack_items(cfg):
    """The attacks set as case items."""
    items = []
    for conv, lines in attack_conversations(cfg):
        for j, (label, speaker, text) in enumerate(lines):
            items.append({"conv": conv, "channel": "chat", "speaker": speaker, "text": text, "label": label,
                          "history": [(s, t) for _, s, t in lines[:j]]})
    return items


def case_items(cfg, files, count_tokens):
    """Each case-file conversation is replayed as one conversation, as the model host would score it."""
    return windowed(load_cases([cfg.path(f) for f in files]), count_tokens)


def case_set_items(cfg, set_name, count_tokens):
    """case_items of an [eval.sets] entry, or of the attacks set."""
    if set_name == "attacks":
        return windowed(attack_items(cfg), count_tokens)
    return case_items(cfg, cfg["eval"]["sets"][set_name], count_tokens)


def windowed(items, count_tokens):
    """(key, item) of case items, each history cut to its conversation's streaming window."""
    starts = [j for j, it in enumerate(items) if not it["history"]] + [len(items)]
    for a, b in zip(starts, starts[1:]):
        last = items[b - 1]
        sizes = earlier_lines(last["history"] + [(last["speaker"], last["text"])], count_tokens)
        for j in range(a, b):
            items[j]["history"] = items[j]["history"][j - a - sizes[j - a]:]
    return list(zip(case_keys([(it["conv"], it["text"]) for it in items]), items))


def message_items(cfg, labels, count_tokens):
    """Messages of messages.parquet, {row: label}, with their history from their conversation's earlier messages."""
    m = pq.read_table(cfg.data / "messages.parquet", columns=["ts", "channel", "conv", "speaker", "text"]).to_pydict()
    wanted = {m["conv"][i] for i in labels}
    convs = defaultdict(list)
    for i, conv in enumerate(m["conv"]):
        if conv in wanted:
            convs[conv].append(i)
    items = []
    for rows in convs.values():
        lines = [(m["speaker"][i], m["text"][i]) for i in rows]
        for j, (i, n) in enumerate(zip(rows, earlier_lines(lines, count_tokens))):
            if i in labels:
                items.append((i, {"conv": m["conv"][i], "channel": m["channel"][i], "speaker": m["speaker"][i],
                                  "text": m["text"][i], "ts": m["ts"][i], "label": labels[i],
                                  "history": lines[j - n:j]}))
    return sorted(items, key=lambda x: x[0])


MESSAGE_SETS = ("val", "tf", "train", "outside")


def message_labels(cfg, set_name):
    """{messages.parquet row: label} of a set of messages.parquet rows."""
    if set_name == "val":
        v = pq.read_table(cfg.work / "training" / "val.parquet", columns=["row", "label"]).to_pydict()
        return {r: "ban" if y else "ok" for r, y in zip(v["row"], v["label"])}
    if set_name in ("tf", "outside"):
        src = pq.read_table(cfg.data / "messages.parquet", columns=["source"]).column("source").to_pylist()
        if set_name == "tf":
            return {i: "ok" for i, s in enumerate(src) if s == "tf"}
        covered = {r for w in read_jsonl(cfg.data / "windows.jsonl") for r, _, _ in w["lines"]}
        return {i: "ok" for i in range(len(src)) if i not in covered}
    return train_sample(cfg)


def items_of(cfg, set_name, count_tokens=None):
    """(key, item) for every message of a set. count_tokens: the classifier's tokenizer, which sets the streaming
    window (chat.WINDOW_MIN); without it each history holds up to WINDOW_MAX - 1 earlier lines."""
    if set_name in MESSAGE_SETS:
        return message_items(cfg, message_labels(cfg, set_name), count_tokens)
    return case_set_items(cfg, set_name, count_tokens)


def scorable_labels(cfg, set_name):
    """{key: label} of a set's scorable messages, without building their histories."""
    if set_name in MESSAGE_SETS:
        return message_labels(cfg, set_name)
    return {k: it["label"] for k, it in scorable(case_set_items(cfg, set_name, None))}


def train_sample(cfg):
    """{row: label} of the training split's labelled rows in `[review] sample_windows` seeded random windows. Whole
    windows, so consecutive rows of a conversation share their cached prompt prefix in llama-server."""
    t = pq.read_table(cfg.work / "training" / "train.parquet", columns=["row", "label"]).to_pydict()
    lab = pq.read_table(cfg.data / "labels.parquet", columns=["row", "window"]).to_pydict()
    train = dict(zip(t["row"], t["label"]))
    by_window = defaultdict(list)
    for r, w in zip(lab["row"], lab["window"]):
        if r in train:
            by_window[w].append(r)
    # ranked by a seeded hash of the window id, so new windows displace only a few picks and the scores stay reusable
    picked = sorted(by_window, key=lambda w: hashlib.sha256(f"4:{w}".encode()).digest())[:cfg["review"]["sample_windows"]]
    return {r: "ban" if train[r] else "ok" for w in picked for r in by_window[w]}


def set_source(cfg, set_name):
    """What a set's messages and labels come from: messages.parquet's batch list (core.MESSAGES), plus val.parquet
    for val, or the case files. train's rows are picked from train.parquet and labels.parquet, but a row's margin
    does not depend on them: a row that newly enters the sample is scored by the resume, and consumers take the
    sample and its labels from those files (set_scores)."""
    if set_name in MESSAGE_SETS:
        return {MESSAGES: message_batches(cfg),
                **({"set": [fingerprint(cfg.work / "training" / "val.parquet")]} if set_name == "val" else {})}
    if set_name == "attacks":
        return {"set": hashlib.sha256(json.dumps(attack_conversations(cfg)).encode()).hexdigest()}
    return {"set": [fingerprint(cfg.path(f)) for f in cfg["eval"]["sets"][set_name]]}


def set_inputs(cfg, spec, set_name, gguf):
    """What a set's scores from a llama-served model come from (see the module docstring)."""
    inputs = {**set_source(cfg, set_name), "model": fingerprint(gguf)}
    if spec["kind"] == "classifier":
        # the thread count changes only the speed
        inputs.update(llama=fingerprint(cfg.path(cfg["eval"]["server"]).parent), context=window_settings(cfg),
                      prompt=PROMPT_FORMAT, tokenizer=spec.get("tokenizer"))
    else:
        inputs.update(**server_inputs(cfg, "chat"), prompt=fingerprint(cfg.path(spec["prompt"])),
                      history=spec["history"])
    return inputs


def current_scores(cfg, path, inputs, set_name):
    """A set's scores for its current scorable messages, with their current labels, and the number of those messages
    still unscored; None when the file is missing or was scored from other inputs. A consumer uses a set only when
    nothing is unscored."""
    if not is_current(path, inputs):
        return None
    wanted = scorable_labels(cfg, set_name)
    rows, seen = [], set()
    for r in read_jsonl(path):
        if r["k"] in wanted and r["k"] not in seen:
            seen.add(r["k"])
            rows.append(dict(r, label=wanted[r["k"]]))
    return rows, len(wanted) - len(seen)


def set_scores(cfg, model, set_name):
    """current_scores of a [models.*] entry's `uv run eval` scores on a set."""
    spec = cfg["models"][model]
    if "gguf" not in spec:
        return None
    return current_scores(cfg, cfg.work / "eval" / model / f"{set_name}.jsonl",
                          set_inputs(cfg, spec, set_name, cfg.path(spec["gguf"])), set_name)


def scorable(items):
    # real-chat case files mark context lines amb; they are history only, never scored
    return [(k, it) for k, it in items if not (it["label"] == "amb" and it["conv"].startswith("real-"))]


def record(k, it, res):
    """One scored message as stored in <set>.jsonl."""
    return {"k": k, **{f: it[f] for f in ("conv", "channel", "speaker", "text", "label") if f in it},
            **({"ts": it["ts"]} if "ts" in it else {}), **res}


def start_scores(out, inputs):
    """Keeps a set's score file when it was made from `inputs` (a partial run continues), else starts it over."""
    if out.exists() and not is_current(out, inputs):
        log.info("%s was scored from other inputs (%s); scoring it again", out.name, ", ".join(changed_keys(out, inputs)))
        out.unlink()
    record_inputs(out, inputs)


def score_set(name, set_name, scorer, items, out, inputs):
    """scorer(conversations) yields (key, item, result) for the messages of conversations [[(key, item), ...], ...],
    each in conversation order."""
    start_scores(out, inputs)
    done = {r["k"] for r in read_jsonl(out, repair=True)}
    todo = [(k, it) for k, it in scorable(items) if k not in done]
    if not todo:
        return
    convs = defaultdict(list)
    for k, it in todo:
        convs[it["conv"]].append((k, it))
    with JsonlAppender(out) as w, progress(f"{name} {set_name}", len(todo), "msgs") as advance:
        for k, it, res in scorer(list(convs.values())):
            w.write(record(k, it, res))
            advance()


# ---------------------------------------------------------------- summaries

def score(r, platt=None):
    """P(ban): sigmoid(a * margin + b) with the model's Platt parameters [a, b], sigmoid(margin) without them."""
    if r.get("margin") is None:
        return r.get("p_ban") or 0.0
    a, b = platt or (1.0, 0.0)
    return sigmoid(a * r["margin"] + b)


def auroc(pos, neg):
    if not pos or not neg:
        return float("nan")
    neg = sorted(neg)
    wins = sum(bisect.bisect_left(neg, p) + 0.5 * (bisect.bisect_right(neg, p) - bisect.bisect_left(neg, p))
               for p in pos)
    return wins / (len(pos) * len(neg))


def pr_auc(scored):
    scored = sorted(scored, key=lambda x: -x[0])
    total = sum(1 for _, y in scored if y)
    tp = fp = 0
    area = prev = 0.0
    for _, y in scored:
        tp += y
        fp += not y
        recall = tp / max(total, 1)
        area += (recall - prev) * (tp / (tp + fp))
        prev = recall
    return area


def message_stats(rows, platt=None, cut=None):
    """Per-message ban AUROC and PR-AUC; with cut (the margin at val FPR 1%), the bans and others at or above it."""
    labelled = [r for r in rows if r["label"] in ("ban", "ok")]
    scored = [(score(r, platt), r["label"] == "ban") for r in labelled]
    pos = [s for s, y in scored if y]
    neg = [s for s, y in scored if not y]
    stats = {"auroc": auroc(pos, neg), "pr_auc": pr_auc(scored), "ban": len(pos), "not": len(neg)}
    if cut is not None:
        hits = [r["label"] == "ban" for r in labelled if r["margin"] is not None and r["margin"] >= cut]
        stats.update(caught=sum(hits), flagged=len(hits) - sum(hits))
    return stats


def ranking(rows):
    """AUROC, PR-AUC and, at each val false-positive rate in FPRS, the recall and the margin it takes, of rows with a
    ban/ok label and a margin."""
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
    rows = [r for r in rows if r["label"] in ("ban", "ok") and r.get("margin") is not None]
    y = [int(r["label"] == "ban") for r in rows]
    m = [r["margin"] for r in rows]
    out = {"auroc": float(roc_auc_score(y, m)), "pr_auc": float(average_precision_score(y, m)), "ban": sum(y),
           "ok": len(y) - sum(y)}
    fpr, tpr, thr = roc_curve(y, m)
    for f in FPRS:
        k = np.searchsorted(fpr, f, side="right") - 1
        out[f"recall@fpr{f}"] = float(tpr[k])
        out[f"margin@fpr{f}"] = float(thr[k])
    return out


def novel_rows(cfg, rows):
    """The rows whose normalized text (stripped, lowercased, as train's duplicate cap counts it) appears nowhere in
    train.parquet. Spam and common phrases repeat across conversations, so val shares many texts with train."""
    seen = {t.strip().lower() for t in pq.read_table(cfg.work / "training" / "train.parquet", columns=["text"]).column("text")
            .to_pylist()}
    return [r for r in rows if r["text"].strip().lower() not in seen]


def val_metrics(cfg, rows):
    """{"val": ranking of every val row, "val_novel": ranking of the rows with text unseen in train} and their summary
    lines."""
    metrics = {"val": ranking(rows), "val_novel": ranking(novel_rows(cfg, rows))}
    lines = []
    for key, what in (("val", "all val rows"), ("val_novel", "val rows whose text never appears in train")):
        s = metrics[key]
        lines.append(f"{what}: AUROC {s['auroc']:.4f}, PR-AUC {s['pr_auc']:.4f} ({s['ban']} ban, {s['ok']} ok); "
                     + ", ".join(f"recall {s[f'recall@fpr{f}']:.3f} at FPR {f:.1%}" for f in FPRS))
    return metrics, lines


def case_summary(rows, platt=None, cut=None):
    s = message_stats(rows, platt, cut)
    lines = [f"per-message ban AUROC {s['auroc']:.3f}, PR-AUC {s['pr_auc']:.3f} ({s['ban']} ban, {s['not']} not)"]
    if cut is not None:
        lines.append(f"at the val FPR 1% margin {cut:.3f}: {s['caught']}/{s['ban']} bans caught, {s['flagged']} "
                     f"others flagged")
    players = defaultdict(list)
    for r in rows:
        if r["label"] != "amb":
            players[(r["conv"], r["speaker"])].append(r)
    for name, k, n in (("single message", 1, 1), ("2 of last 3", 2, 3), ("3 of last 5", 3, 5)):
        offenders, innocents = [], []
        for msgs in players.values():
            s = [score(m, platt) for m in msgs]
            vals = [sorted(s[max(0, i + 1 - n):i + 1], reverse=True)[k - 1] if i + 1 >= k else 0.0
                    for i in range(len(s))]
            (offenders if any(m["label"] == "ban" for m in msgs) else innocents).append(max(vals))
        t = max(innocents, default=0.0) + 1e-9
        lines.append(f"{name:15s} player AUROC {auroc(offenders, innocents):.3f}; at zero innocent bans "
                     f"{sum(o >= t for o in offenders)}/{len(offenders)} offenders caught")
    return lines


def known_threshold(cfg, rows, platt=None):
    """Single-message rule at the highest threshold that still triggers every known offender found in rows: the
    threshold, the offender who sets it, how many known offenders were found and how many players trigger."""
    best = {}
    for r in rows:
        p = r["speaker"].lower()
        best[p] = max(best.get(p, 0.0), score(r, platt))
    known = {p: best[p] for p in (k.lower() for k in cfg["eval"].get("known_offenders", [])) if p in best}
    if not known:
        return None
    setter = min(known, key=known.get)
    return {"threshold": known[setter], "set_by": setter, "known": len(known),
            "players": sum(v >= known[setter] for v in best.values())}


def tf_summary(cfg, rows, platt=None, cut=None):
    known = [p.lower() for p in cfg["eval"].get("known_offenders", [])]
    by_player = defaultdict(list)
    for r in sorted(rows, key=lambda r: r["k"]):
        by_player[r["speaker"].lower()].append(r)
    days = len({(r.get("ts") or "")[:10] for r in rows if r.get("ts")}) or 1
    lines = [f"{len(rows):,} messages, {len(by_player):,} players, {days} days"]
    if cut is not None:
        from .failures import message_labels
        labels = message_labels(cfg)
        offenders = {p for p, ms in by_player.items() if any(labels.get(r["k"], ("ok",))[0] == "ban" for r in ms)}
        hit = {p for p, ms in by_player.items() if any(r["margin"] is not None and r["margin"] >= cut for r in ms)}
        lines.append(f"at the val FPR 1% cut, P(ban) >= {score({'margin': cut}, platt):.4g}: {len(hit & offenders)} "
                     f"of {len(offenders)} labelled offenders (a row labelled ban) triggered, {len(hit - offenders)} "
                     f"other players ({len(hit) / days:.2f} players/day)")
    kt = known_threshold(cfg, rows, platt)
    if kt:
        lines.append(f"every known offender found ({kt['known']} of {len(known)}) triggers at P(ban) >= "
                     f"{kt['threshold']:.4g}, single message, set by {kt['set_by']}: {kt['players']} players "
                     f"triggered ({kt['players'] / days:.2f}/day)")
    for t in cfg["eval"]["thresholds"]:
        for name, (k, n) in {"single": (1, 1), "2-of-3": (2, 3), "3-of-5": (3, 5)}.items():
            banned = {}
            for p, msgs in by_player.items():
                window, last = [], None
                for r in msgs:
                    ts = r.get("ts")
                    now = time.mktime(time.strptime(ts, "%Y-%m-%dT%H:%M:%S")) if ts else None
                    if last is not None and now is not None and now - last > 600:
                        window = []  # the rule state resets after 10 quiet minutes
                    last = now if now is not None else last
                    window = (window + [score(r, platt) >= t])[-n:]
                    if sum(window) >= k:
                        banned[p] = r
                        break
            caught = [p for p in known if p in banned]
            lines.append(f"t={t} {name:6s}: {len(banned)} players triggered ({len(banned) / days:.2f}/day); known "
                         f"offenders {len(caught)}/{len(known)}" + (f", missed {sorted(set(known) - set(caught))}"
                                                                     if len(caught) < len(known) else ""))
    return lines


def outside_summary(cfg, rows, platt=None):
    """Rows outside every window are ok only because no risk term opened a window: how many score high, by source."""
    p = [(score(r, platt), r["k"]) for r in rows]
    src = pq.read_table(cfg.data / "messages.parquet", columns=["source"]).column("source").to_pylist()
    lines = [f"{len(rows):,} messages outside every window"]
    for t in cfg["eval"]["thresholds"]:
        by = Counter(src[k] for s, k in p if s >= t)
        lines.append(f"P(ban) >= {t}: {sum(by.values()):,} messages ("
                     + ", ".join(f"{s} {n:,}" for s, n in by.most_common()) + ")")
    return lines


def set_summary(cfg, set_name, rows, platt, cut=None):
    """The summary lines of one set's scores. cut: the margin at val FPR 1%, the operating cut."""
    if set_name == "tf":
        return tf_summary(cfg, rows, platt, cut)
    if set_name == "outside":
        return outside_summary(cfg, rows, platt)
    if set_name == "val":
        return val_metrics(cfg, rows)[1] + case_summary(rows, platt)
    return case_summary(rows, platt, cut)


def fit_platt(cfg, rows):
    """Platt scaling and ranking metrics from val rows. Returns (platt dict, metrics, summary lines)."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    y = [int(r["label"] == "ban") for r in rows]
    m = [r["margin"] for r in rows]
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(np.array([[x] for x in m]), y)
    platt = {"a": float(lr.coef_[0][0]), "b": float(lr.intercept_[0])}
    metrics, lines = val_metrics(cfg, rows)
    for f in FPRS:
        platt[f"margin@fpr{f}"] = metrics["val"][f"margin@fpr{f}"]
    lines += [f"at val FPR {f:.1%}: margin >= {platt[f'margin@fpr{f}']:.3f}" for f in FPRS]
    lines.append(f"Platt (calibrated on val's {sum(y) / len(y):.1%} ban, not live chat's): P(ban) = "
                 f"sigmoid({platt['a']:.4f} * margin {'-' if platt['b'] < 0 else '+'} {abs(platt['b']):.4f})")
    return platt, metrics, lines


def toml_string(s):
    """A TOML basic string: JSON's string escapes are all valid TOML escapes, so a Windows path keeps its
    backslashes."""
    return json.dumps(s, ensure_ascii=False)


def write_classifier(cfg, gguf, tokenizer, platt=None):
    """data/classifier.toml's [models.classifier]. `export` writes it for a new GGUF without Platt parameters; `eval`
    adds them, fitted on that GGUF's own val margins, so calibration and thresholds come from the file that ships."""
    text = (f"# Written by `uv run export` and `uv run eval`. Overrides ecc.toml; local.toml overrides it.\n"
            f"[models.classifier]\ngguf = {toml_string(gguf)}\n"
            f"tokenizer = {toml_string(tokenizer)}   # the base model the run was trained on\n")
    if platt:
        text += f"platt = [{platt[0]!r}, {platt[1]!r}]   # P(ban) = sigmoid(a * margin + b), fitted on this GGUF's val\n"
    write_text(cfg.data / "classifier.toml", text)
    log.info("[models.classifier] in %s: %s%s", cfg.data / "classifier.toml", gguf,
             f", Platt a {platt[0]:.4f}, b {platt[1]:.4f}" if platt else ", no Platt yet (`uv run eval` fits it on val)")


def body(cfg, args):
    name = args.model or cfg["eval"]["model"]
    spec = cfg["models"][name]
    every = [*cfg["eval"]["sets"], "attacks", "val", "tf", "train", "outside"]
    if spec["kind"] != "classifier":
        every = [*cfg["eval"]["sets"], "attacks", "tf"]   # val, train and outside prompts are built for the classifier
    sets = args.set or [s for s in every if s not in ("train", "outside")]
    if "gguf" not in spec:
        raise SystemExit(f"[models.{name}] has no gguf; run `uv run export` first")
    platt = spec.get("platt")
    count = None
    if spec["kind"] == "classifier":
        if "tokenizer" not in spec:
            raise SystemExit(f"[models.{name}] has no tokenizer; run `uv run export`")
        from .classifier import base_model, token_count   # first: it puts the Hugging Face hub offline
        from transformers import AutoTokenizer
        count = token_count(AutoTokenizer.from_pretrained(base_model(cfg, spec["tokenizer"])))
    folder = cfg.work / "eval" / name
    folder.mkdir(parents=True, exist_ok=True)
    gguf = cfg.path(spec["gguf"])
    servers = []

    def scorer(convs):
        if not gguf.exists():
            raise SystemExit(f"model file {gguf} does not exist" +
                             ("; run `uv run export` first" if spec["kind"] == "classifier" else ""))
        if spec["kind"] == "classifier":
            yield from window_scores(cfg, gguf, count, folder / "llama.log", convs)
            return
        if not servers:
            log.info("starting llama-server on %s", gguf.name)
            servers.append(Server(cfg, gguf, folder / "server.log", "chat").__enter__())
        chat = Scorer(cfg, spec, servers[0])
        for conv in convs:
            for k, it in conv:
                yield k, it, chat(it["history"], it["speaker"], it["text"], it["channel"])

    items = {}

    def set_items(set_name):
        if set_name not in items:
            items[set_name] = items_of(cfg, set_name, count)
        return items[set_name]

    try:
        for set_name in sets:
            if set_name not in every:
                raise SystemExit(f"{name} has no set {set_name!r}; its sets: {', '.join(every)}")
            out = folder / f"{set_name}.jsonl"
            score_set(name, set_name, scorer, set_items(set_name), out, set_inputs(cfg, spec, set_name, gguf))
    finally:
        for s in servers:
            s.__exit__()
    val = set_scores(cfg, name, "val")
    cut = None
    if val and not val[1]:
        if name == "classifier":
            fitted, metrics, lines = fit_platt(cfg, val[0])
            platt = (fitted["a"], fitted["b"])
            if list(platt) != list(spec.get("platt") or []):
                write_classifier(cfg, spec["gguf"], spec["tokenizer"], platt)
            for line in lines[-len(FPRS) - 1:]:
                log.info("%s val: %s", name, line)
        else:
            metrics = val_metrics(cfg, val[0])[0]
        cut = metrics["val"]["margin@fpr0.01"]
    summary = [f"# {name}"]
    current = {}
    for set_name in every:
        scores = set_scores(cfg, name, set_name)
        if scores is None:
            if (folder / f"{set_name}.jsonl").exists():
                log.info("%s %s: scored from other inputs; left out of the summary until it is scored again", name,
                         set_name)
            continue
        rows, missing = scores
        if missing:
            summary += ["", f"## {set_name}", f"partly scored: {missing:,} of {missing + len(rows):,} messages have no "
                        f"score; `uv run eval --model {name} --set {set_name}` scores them"]
            log.info("%s %s: %s messages unscored; left out of the summary", name, set_name, f"{missing:,}")
            continue
        current[set_name] = rows
        text = set_summary(cfg, set_name, rows, platt, cut)
        summary += ["", f"## {set_name}", *text]
        for k, line in enumerate(text):
            shown = (" single" in line or "labelled offenders" in line) if set_name == "tf" else k < 2
            log.log(20 if shown else 10, "%s %s: %s", name, set_name, line)
    summary_path = folder / "summary.md"
    write_text(summary_path, "\n".join(summary) + "\n")
    log.info("full summary: %s", summary_path)

    def alone(targets):
        """{(set, key): margin} of each target message scored with no earlier lines."""
        if spec["kind"] != "classifier" or not gguf.exists():
            return {}
        convs = [[(json.dumps([s, k], ensure_ascii=False), dict(it, history=[]))] for s, k, it in targets]
        return {tuple(json.loads(key)): res["margin"]
                for key, _, res in window_scores(cfg, gguf, count, folder / "alone.log", convs)}

    from .failures import report
    failures_path = folder / "failures.md"
    write_text(failures_path, "\n".join(report(cfg, name, current, platt, cut, set_items, alone)) + "\n")
    log.info("where it fails: %s", failures_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", help="a [models.*] entry in ecc.toml (default: [eval] model)")
    ap.add_argument("--set", nargs="+", help="only these sets")
    run("eval", body, ap)

