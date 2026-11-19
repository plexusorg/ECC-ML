"""uv run replay-export: the test weeks as ECC-Server's `ecc-server replay` reads them, into [replay] out.

- lines.jsonl: every run of [export] batch that holds a test file (the line-label table's `split`), in table order, one JSON event a
  line:
  - a `stream` event per run (`day`: the run's first day), with `coverage_rule: complete_target_v1`;
  - a `presence` event for each join and leave line of the run's log files, from the transcript build of the batch's
    saved parts (`account` lowercased);
  - each line as ECC-Server receives it, prepared (`text`, `name`, `name_shown`) and raw (`raw_text`, `raw_shown`),
    with `sent` 1: every copy the transcript folded into a line is a line of its own at its own time (`copy_of`: the key
    of the line it repeats), so the server folds and counts copies live. `id` is the log's line id (as the judge is
    shown it); `key` (`<transcript>:<id>`) is unique within a stream, which `id` is not when a run spans files. `account`
    (the live engine's player key; the plugin sends the UUID) is the speaker's resolved account, so a player who shows
    several names is one player; a line with none (a Discord relay) keeps the shown name's;
  - every line carries its own coverage route, including raw copies. A rejected line carries `verdict: reject` and
    enters no model context or judge session. Every covered target goes directly to the classifier;
  - a judged line of a test file carries its session, the fast system's verdict (`hold` when the classifier's hold
    logit reaches the recall threshold) and its category marks (the classifier's category logits at the
    category thresholds), from the torch.parquet and thresholds.json of [export] classifier's export slot. A folded copy uses the
    original line's decision. The owner's own
    sessions ([replay] owner) are `trusted`. Every other line, and a judged test line without a score, is context only.
    The exporter refuses recorded logits when this stream changes their coverage, context, tokens or features. It
    preserves previous replay files on refusal. Score the actual replay stream before using changed inputs.
- judge/: an `-fold` folder for `ecc-server replay -mode live`: question.md (the question and the added sentence),
  format.md, cases.json (the owner's ruled cases dated before [periods] test_from, with his rulings, in the bundle's
  order), answers.json (no saved answers) and expected.json (no parity prompts).
"""
import argparse
import json
import os
import re
from datetime import date
from pathlib import Path

import numpy as np

from . import bundle as BU
from . import transcript as T
from . import inputs as I
from .core import log, run


def shown_account(label):
    """The account a shown name names: the part after '=' when the shown name hides it, else its last word without
    rank tags."""
    label = re.sub(r"\s*\(replying to .*\)$", "", label)
    if "=" in label:
        label = label.rsplit("=", 1)[1]
    label = re.sub(r"\[[^\]]*\]", " ", label).split()
    return (label[-1] if label else "").lstrip("~").lower()


def scored(cfg):
    """Read the export slot's exact scores and use the validation recall threshold."""
    import pandas as pd
    from .export import folders, slot_thresholds
    f = folders(cfg)
    thr = slot_thresholds(cfg, f)
    path = f["out"] / "torch.parquet"
    return system(pd.read_parquet(path)), thr["hold_recall"], thr["categories"]


def system(t):
    """The classifier's logits for every judged test key. Rejected targets have no model logits."""
    t = t[t["period"] == "test"]
    if "coverage" not in t or not t.coverage.isin(["scored", "rejected"]).all():
        raise SystemExit("replay requires complete-target scores; run `uv run export`")
    out = t[["transcript", "id", "coverage"]].copy()
    for o in ["hold"] + I.CATEGORIES:
        out["z_" + o] = t["classifier_" + o].to_numpy()
    if out.duplicated(["transcript", "id"]).any():
        raise SystemExit("torch.parquet scores a line twice")
    return out.reset_index(drop=True)


def lines_table(t, sc, hold, categories):
    """The rows of the runs that hold a test file, with the score (`scored`), verdict (`held`) and category marks (`mk`) of
    each judged test line."""
    t = t[t["run"].isin(set(t.loc[t["split"] == "test", "run"]))].reset_index(drop=True)
    t["partner"] = t["partner"].fillna("").astype(str)
    t["session"] = t["session"].astype(object).where(t["session"].notna(), None)
    t = t.merge(sc, on=["transcript", "id"], how="left", validate="1:1")
    t["test"] = t["split"] == "test"
    t["scored"] = t["coverage"].notna()
    if (t["scored"] & ~(t["test"] & t["judged"])).any():
        raise SystemExit("torch.parquet scores a line that is not a judged line of a test file")
    if not (t["judged"] == t["session"].notna()).all():
        raise SystemExit("a judged line has no session, or a line not judged has one")
    t["held"] = t["scored"] & (t["z_hold"] >= hold)
    t["mk"] = [[k for k in I.CATEGORIES if r.scored and getattr(r, "z_" + k) >= categories[k]] for r in t.itertuples()]
    return t


def presence(cfg, t, folder):
    """{transcript: [(seconds from the run's start, line id, event)]}: the join and leave lines of the transcripts of t, from
    the transcript build of the batch's saved parts."""
    manifest = T.load_manifest(folder)
    want = set(t["transcript"])
    run_of = dict(zip(t["transcript"], t["run"]))
    out = {}
    for _, d in T.build_batch(cfg, {"kind": "server_log", "transcripts": [x for x in manifest["transcripts"]
                                                                  if x["transcript"] in want]}, folder):
        r = run_of[d["stem"]]
        base = (date.fromisoformat(d["day"]) - date.fromisoformat(r[:10])).days * 86400
        evs, day, last = [], 0, None
        for lid, x in zip(d["ids"], d["raw"]):
            sec = x["sec"]
            if last is not None and sec + day < last - 6 * 3600:      # past midnight, as line-labels reads the parts
                day += 86400
            last = sec = sec + day
            if x["channel"] == "start" and not (base == 0 and not evs):
                raise SystemExit(f"{d['stem']} {lid}: a server start inside a run")
            if x["channel"] in ("join", "leave"):
                evs.append((base + sec, lid, {"type": "presence", "kind": x["channel"], "account": x["name"].lower()}))
        out[d["stem"]] = evs
    return out


def at_key(rank, time, lid):
    """The transcript's order: file, second, log lines before record lines, line number."""
    return rank, time, lid[0] != "L", int(lid[1:])


def iso(day, seconds):
    import pandas as pd
    return (day + pd.Timedelta(seconds=int(seconds))).strftime("%Y-%m-%dT%H:%M:%SZ")


def events(t, pres, owner, coverage):
    """(the replay's events in order, counts)."""
    import pandas as pd
    out = []
    counts = {"lines": 0, "copies": 0, "presence": 0, "context only": 0, "judged test lines without a score": 0,
              "held": 0, "sessions with a held line": 0}
    for run_name, g in t.groupby("run", sort=False):
        day = pd.Timestamp(run_name[:10], tz="UTC")
        out.append({"type": "stream", "name": run_name, "day": run_name[:10],
                    "coverage_rule": "complete_target_v1", "model_text_rule": "v3_meaning_v1",
                    "canonical_link": coverage.b.text.canonical})
        rank = {d: k for k, d in enumerate(dict.fromkeys(g["transcript"]))}
        evs = []
        for d, ps in pres.items():
            if d in rank:
                for sec, lid, ev in ps:
                    evs.append((at_key(rank[d], sec, lid), {**ev, "time": iso(day, sec)}))
        for r in g.itertuples():
            shown = shown_account(r.speaker)
            speaker = (r.account or "").lower() or "discord:" + r.speaker.lower()
            account = (r.account or "").lower() or shown
            private = r.channel == "msg"
            target = shown_account(r.partner) if r.partner else ""
            base = {"type": "line", "channel": r.channel,
                    "conversation": ("msg/" + "|".join(sorted({shown, target}))) if private else r.channel,
                    "private": private, "account": account, "name": r.speaker, "name_shown": r.speaker,
                    "target_account": target, "target": r.partner, "text": r.text, "sent": 1, "speaker": speaker,
                    "_judged": bool(r.judged), "_scored": bool(r.judged and r.test and r.scored)}
            if not (r.judged and r.test and r.scored):
                base["context"] = True
                counts["context only"] += 1
                counts["judged test lines without a score"] += bool(r.judged and r.test)
            else:
                base["session"] = r.session
                trusted = r.session.split("|", 1)[1] == owner
                if trusted:
                    base["trusted"] = True
                base["verdict"] = "hold" if r.held else "deliver"
                base["categories"] = list(r.mk)
            k = f"{r.transcript}:{r.id}"
            evs.append((at_key(rank[r.transcript], r.time, r.id),
                        {**base, "id": r.id, "key": k, "time": iso(day, r.time), "raw_text": r.raw_text,
                         "raw_shown": r.raw_shown}))
            counts["lines"] += 1
            for c in r.copies:
                evs.append((at_key(rank[r.transcript], c["time"], c["id"]),
                            {**base, "id": c["id"], "key": f"{r.transcript}:{c['id']}", "time": iso(day, c["time"]),
                             "raw_text": c["raw_text"], "raw_shown": c["raw_shown"], "copy_of": k}))
                counts["copies"] += 1
        counts["presence"] += sum(1 for _, e in evs if e["type"] == "presence")
        prev = None
        for _, e in sorted(evs, key=lambda x: x[0]):
            if prev is not None and e["time"] < prev:
                raise SystemExit(f"run {run_name}: time runs backwards at {e.get('key') or e}")
            prev = e["time"]
            out.append({k: v for k, v in e.items() if not (k == "raw_shown" and (v is None or v != v))})
    annotate_coverage(out, coverage)
    counts["context only"] = sum(e.get("coverage") == "context" and not e.get("copy_of") for e in out)
    counts["rejected"] = sum(e.get("coverage") == "rejected" for e in out)
    counts["rejected copies"] = sum(e.get("coverage") == "rejected" and bool(e.get("copy_of")) for e in out)
    counts["held"] = sum(e.get("verdict") == "hold" and not e.get("trusted") for e in out)
    counts["sessions with a held line"] = len({e["session"] for e in out
                                               if e.get("verdict") == "hold" and not e.get("trusted")})
    return out, counts


def coverage_scorer(cfg):
    """Use the reference stream builder with current tokenizers. This performs no model inference."""
    from types import SimpleNamespace
    from .export import classifier_run, folders, links
    from .reference import Scorer, Text
    f = folders(cfg)
    allow, canonical = links(cfg)
    allowed = [host + prefix for host, prefix in allow]
    b = SimpleNamespace(text=Text({"text": BU.text_rules(), "links": BU.links_rules(canonical),
                                  "model_text": {"rule": "v3_meaning_v1"}}, allowed),
                        tok={k: I.tokenizer(f[k] / "tokenizer.json") for k in ("classifier",)},
                        channels=set(I.CHANNELS), n_ctx=I.CTX_LINES, secs=I.CTX_SECONDS,
                        public=set(I.PUBLIC), placed=set(I.PLACED), private=I.PRIVATE,
                        max_tokens=classifier_run(f["classifier"])["max_tokens"], BOS=I.BOS, EOS=I.EOS,
                        HOLD=I.HOLD, hold={"recall": 0}, default="recall", categories=[])
    return Scorer(b, ids=True, inputs_only=True)


def annotate_coverage(evs, scorer):
    """Route each original and raw copy before folding. Keep the complete source in each wire event."""
    from datetime import datetime
    stream = None
    traces = {}
    for e in evs:
        if e["type"] == "stream":
            stream = e["name"]
        if e["type"] != "line":
            continue
        wire = BU.wire(stream, e["key"], int(datetime.fromisoformat(e["time"]).timestamp()), e["channel"],
                       e["account"], e["target_account"], e["raw_text"], e.get("raw_shown"), e.get("trusted", False))
        # A held-out measurement does not turn an earlier judged target into a trusted exemption.
        wire["context"] = e.get("context", False) and not e["_judged"]
        trace = scorer.score(wire)
        if "refused" in trace:
            raise SystemExit(f"replay {e['key']}: unknown channel")
        route = "dropped" if trace.get("dropped") else "fold" if "folded_into" in trace else \
            "context" if trace.get("context_only") else trace["coverage"]
        if route == "scored" and e.get("context"):
            route = "context"
        if route == "scored" and e.get("copy_of"):
            raise SystemExit(f"replay {e['key']}: copy needs its own model score; score the actual replay stream")
        e["coverage"] = route
        if route == "rejected":
            e["verdict"], e["categories"] = "reject", []
            e.pop("context", None)
        elif route in ("fold", "context", "dropped"):
            e.pop("verdict", None)
            e.pop("categories", None)
        traces[e["key"]] = trace
    return traces


def verify_recorded_inputs(cfg, t, evs):
    """Refuse logits from a different route, context or input before replacing previous replay files."""
    from .export import load_lines, model_inputs
    df, _ = load_lines(cfg)
    rows_of = {(r.transcript, r.id): i for i, r in enumerate(df.itertuples())}
    selected = t[t.scored & t.test & t.judged]
    rows = np.array([rows_of[(r.transcript, r.id)] for r in selected.itertuples()], dtype=np.int64)
    ctx, cids, routes = model_inputs(cfg, df, rows)
    traces = annotate_coverage(evs, coverage_scorer(cfg))
    check_recorded_inputs(evs, traces, df, selected, rows, ctx, cids, routes)


def check_recorded_inputs(evs, traces, df, selected, rows, ctx, cids, routes):
    """Check the classifier inputs used by recorded scores against the inputs of the actual replay stream."""
    expected = {f"{r.transcript}:{r.id}": (j, rows[j], r.coverage) for j, r in enumerate(selected.itertuples())}
    for e in evs:
        if e["type"] != "line" or not e["_scored"] or e.get("trusted") or e.get("copy_of"):
            continue
        key = e["key"]
        if key not in expected:
            raise SystemExit(f"replay {key}: no canonical score input")
        j, row, saved_route = expected[key]
        trace = traces[key]
        if e["coverage"] != saved_route or e["coverage"] != routes[row]:
            raise SystemExit(f"replay {key}: recorded and actual coverage differ; score the actual replay stream")
        if e["coverage"] == "rejected":
            continue
        context_keys = [f"{df.transcript[k]}:{df.id[k]}" for k in ctx[row]]
        if trace.get("context_ids") != context_keys or trace.get("classifier_input_ids") != cids[j].tolist():
            raise SystemExit(f"replay {key}: recorded and actual context or input differ; score the actual replay "
                             "stream")


def judge_files(cfg, test_from):
    """{name: text} of the judge/ folder: the bundle's question with its added sentence, its answer format, and the
    owner's ruled cases dated before test_from."""
    from .export import links
    fixed = BU.judge_fixed(cfg)
    _, examples = BU.judge_examples(cfg)
    cases = [{"ruling": e["ruling"], "text": e["text"]} for e in examples if e["date"] < test_from]
    expected = {"prompts": {}, "judged_sessions": [], "example_sessions": [], "outcomes": {}}
    return {"question.md": fixed["QUESTION"] + "\n\n" + fixed["ADDED"], "format.md": fixed["FORMAT"],
            "cases.json": json.dumps(cases, ensure_ascii=False, indent=1),
            "answers.json": "{}", "expected.json": json.dumps(expected, indent=1),
            "allowlist.txt": "".join(host + prefix + "\n" for host, prefix in links(cfg)[0])}


def write(path, text):
    """UTF-8 with LF line ends, through a temporary file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(text.encode("utf-8"))
    os.replace(tmp, path)


def export(cfg, t, sc, hold, categories, folder, out, test_from, unscored_ok=False):
    t = lines_table(t, sc, hold, categories)
    n = int((t["judged"] & t["test"] & ~t["scored"]).sum())
    if n and not unscored_ok:
        raise SystemExit(f"{n} judged lines of the test files have no score in torch.parquet: run `uv run export` on "
                         "the current table")
    evs, counts = events(t, presence(cfg, t, folder), cfg["replay"]["owner"], coverage_scorer(cfg))
    verify_recorded_inputs(cfg, t, evs)
    wire = [{k: v for k, v in e.items() if not k.startswith("_")} for e in evs]
    write(out / "lines.jsonl", "".join(json.dumps(e, ensure_ascii=False, allow_nan=False) + "\n" for e in wire))
    files = judge_files(cfg, test_from)
    for name, text in files.items():
        write(out / "judge" / name, text)
    streams = sum(1 for e in evs if e["type"] == "stream")
    log.info("%s: %d streams (runs); %s", out / "lines.jsonl", streams, ", ".join(f"{k} {v:,}" for k, v in counts.items()))
    log.info("%s: %d of the owner's cases dated before %s", out / "judge", len(json.loads(files["cases.json"])),
             test_from)
    return counts


def table(cfg):
    import pandas as pd
    from .export import lines_path
    cols = ["transcript", "run", "id", "time", "channel", "partner", "speaker", "account", "text", "raw_text", "raw_shown",
            "copies", "judged", "session", "split"]
    return pd.read_parquet(lines_path(cfg), columns=cols)


def body(cfg, args):
    from .transcript import batch_dir
    test_from = cfg["periods"]["test_from"].isoformat()
    sc, hold, categories = scored(cfg)
    export(cfg, table(cfg), sc, hold, categories, batch_dir(cfg, cfg["export"]["batch"]), cfg.path(cfg["replay"]["out"]),
           test_from)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    run("replay-export", body, ap)
