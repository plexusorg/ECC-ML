"""uv run read BATCH READER: Codex reads each part of a batch in a fresh session and answers the reader's question.

Each part is sent once per reader, alone: `codex exec` in an ephemeral session with a read-only sandbox and an empty
working folder (<work>/read/<batch>-<reader>/<part>), no user config and no rules, the prompt on stdin: the question,
its answer format and, for a "details" question, the lines to describe, then the part. The answer is checked (JSON
only, ids of the part, known categories in each mark's `kind` field) and saved to BATCH/reads/READER/<part>.json; every attempt goes to
BATCH/reads/READER/log.jsonl and an answer that fails the check to BATCH/reads/READER/invalid/. A run reads only the
parts that have no answer yet, so it continues where the last one stopped. A part that fails `uv run transcript`'s check
is never sent. After [read] stop_after usage-limit answers in a row the run starts no new attempt.

    uv run read full codex4-a --parallel 4      every part of full not yet answered by codex4-a
    uv run read full codex4-c --disputed        the parts full/disputed.json lists (the third read)
    uv run read full codex5-a                   the second question, on the lines full's line-label table marks
    uv run read full codex4-a --dry-run         write each prompt beside its working folder and show the command

READER is <reader>-<copy>; the reader's question is the file [read.readers.<reader>] question names in [read]
questions.
"""
import argparse
import json
import re
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import transcript as T
from .core import log, run, write_text
from .line_labels import CATEGORIES

LIMIT = re.compile(r"rate.?limit|usage limit|too many requests|429|at capacity|try again (in|later)", re.I)
PERSONAL = {"age", "gender", "name", "school", "place", "other"}
DETAILS = {
    "ask_contact": {"phone", "account", "call", "meet", "other"},
    "give_contact": {"phone", "account", "link", "address", "other"},
    "ask_personal": PERSONAL,
    "give_personal": PERSONAL,
    "off_server": {"server", "platform", "call", "other"},
    "sexual": {"request", "describe", "remark"},
    "romantic": {"ask", "declare", "endear", "reply"},
}
AGES = {"under18", "adult", "unknown"}


def question(cfg, reader):
    """(the reader's question settings, the prompt's head: the question and its answer format)."""
    q = cfg["read"]["readers"].get(reader.split("-", 1)[0])
    if q is None:
        raise SystemExit(f"no [read.readers.{reader.split('-', 1)[0]}] for the read {reader}")
    text = (cfg.path(cfg["read"]["questions"]) / q["question"]).read_text(encoding="utf-8")
    parts = text.split("## Question", 1)[1].split("## Answer format", 1)
    head = "# Question\n\n" + parts[0].split("\n", 1)[1].strip() + "\n\n# Answer format\n\n" + \
        parts[1].split("## For whoever runs", 1)[0].strip()
    return q, head


def validate_marks(raw, part_text):
    raw = raw.strip()
    if raw.startswith("```"):
        return None, "wrapped in a code fence"
    try:
        obj = json.loads(raw)
    except Exception as e:
        return None, f"not JSON: {e}"
    if not isinstance(obj, dict):
        return None, "not an object"
    for k in ("lines", "unsure"):
        if not isinstance(obj.get(k), list):
            return None, f"{k} missing or not a list"
    ids = set(re.findall(r"^([LR]\d+) ", part_text, re.M))
    for e in obj["lines"]:
        if not isinstance(e, dict) or e.get("id") not in ids:
            return None, f"line id not in part: {e}"
        if e.get("kind") not in CATEGORIES:
            return None, f"bad kind: {e.get('kind')}"
        if not isinstance(e.get("speaker"), str):
            return None, "line without speaker"
    for e in obj["unsure"]:
        if not isinstance(e, dict) or e.get("id") not in ids:
            return None, f"unsure id not in part: {e}"
    return obj, None


def validate_details(raw, asked):
    raw = raw.strip()
    if raw.startswith("```"):
        return None, "wrapped in a code fence"
    try:
        obj = json.loads(raw)
    except Exception as e:
        return None, f"not JSON: {e}"
    if not isinstance(obj, dict) or not isinstance(obj.get("details"), list):
        return None, "details missing or not a list"
    seen = set()
    for e in obj["details"]:
        if not isinstance(e, dict):
            return None, f"entry not an object: {e}"
        key = (e.get("id"), e.get("kind"))
        if key not in asked:
            return None, f"pair not asked: {key}"
        if key in seen:
            return None, f"pair twice: {key}"
        seen.add(key)
        if e.get("detail") not in DETAILS[key[1]]:
            return None, f"bad detail for {key}: {e.get('detail')}"
        if e.get("target_age") not in AGES:
            return None, f"bad target_age for {key}: {e.get('target_age')}"
        if not isinstance(e.get("after_refusal"), bool):
            return None, f"bad after_refusal for {key}: {e.get('after_refusal')}"
    if seen != asked:
        return None, f"pairs missing: {sorted(asked - seen)[:5]}"
    return obj, None


def body_ids(text):
    lines = text.split("\n")
    marks = [i for i, ln in enumerate(lines) if re.match(r"^--- part \d+ begins here ---$", ln)]
    start = marks[0] + 1 if marks else 0
    return [m.group(1) for ln in lines[start:] if (m := re.match(r"^([LR]\d+) ", ln))]


def pairs_by_part(cfg, folder, manifest):
    """{part stem: [(id, category)]} for a "details" read, in the part's line order, categories in CATEGORIES order:
    every (line, category) at 1 in the batch's line-label table, and every (line, category) at -1 that a read marked
    with that category. Each line is
    asked with the part whose body holds it."""
    import pandas as pd
    from .line_labels import table_path
    path = table_path(cfg, folder.name)
    if not path.exists():
        raise SystemExit(f"{path} does not exist; `uv run line-labels {folder.name}` builds it")
    t = pd.read_parquet(path, columns=["transcript", "id", *CATEGORIES, "read_a", "read_b", "read_c"])
    t = t[(t[CATEGORIES] != 0).any(axis=1)]
    want = {}
    for r in t.itertuples(index=False):
        read = set(r.read_a) | set(r.read_b) | (set(r.read_c) if r.read_c is not None else set())
        ks = [k for k in CATEGORIES if getattr(r, k) == 1 or (getattr(r, k) == -1 and k in read)]
        if ks:
            want[(r.transcript, r.id)] = ks
    out, placed = {}, 0
    for d in manifest["transcripts"]:
        for p in d["parts"]:
            ids = body_ids((folder / "parts" / p["file"]).read_text(encoding="utf-8"))
            if ids[0] != p["first"] or ids[-1] != p["last"]:
                raise SystemExit(f"{p['file']}: body {ids[0]}..{ids[-1]} is not the manifest's {p['first']}..{p['last']}")
            got = [(i, k) for i in ids for k in want.get((d["transcript"], i), [])]
            placed += len({i for i, _ in got})
            if got:
                out[p["file"][:-4]] = got
    if placed != len(want):
        raise SystemExit(f"{len(want) - placed} marked lines are in no part body")
    return out


def codex_command(cfg, exe, work, last):
    r = cfg["read"]
    return [exe, "exec", "--skip-git-repo-check", "--ephemeral", "--json", "--ignore-user-config", "--ignore-rules",
            "-m", r["model"], "-c", f'model_reasoning_effort="{r["effort"]}"', "--sandbox", "read-only", "-C",
            str(work), "-o", str(last), "-"]


def prompt_for(head, text, pairs=None):
    if pairs is None:
        return head + "\n\n# Dossier part\n\n" + text
    return head + "\n\n# Lines to describe\n\n" + "\n".join(f"{i} {k}" for i, k in pairs) + "\n\n# Dossier part\n\n" + text


def usage_of(stdout):
    """(the turn's usage event, tool items, the commands it ran) from codex's --json events."""
    usage, tools, cmds = None, 0, []
    for ln in stdout.splitlines():
        try:
            ev = json.loads(ln)
        except ValueError:
            continue
        s = json.dumps(ev)
        if "token" in s and "usage" in s:
            usage = ev
        if ev.get("type") in ("item.completed", "item.started") and \
                (ev.get("item") or {}).get("type") in ("command_execution", "mcp_tool_call", "web_search", "file_change"):
            tools += 1
            it = ev["item"]
            if ev["type"] == "item.completed":
                cmds.append(str(it.get("command") or it.get("query") or it.get("tool") or it.get("type"))[:300])
    return usage, tools, cmds


def body(cfg, args):
    r = cfg["read"]
    folder = T.batch_dir(cfg, args.batch)
    manifest = T.load_manifest(folder)
    if manifest is None:
        raise SystemExit(f"{folder} has no manifest.json")
    q, head = question(cfg, args.reader)
    parallel = max(1, min(args.parallel, q["max_parallel"]))
    out_dir = args.out or folder / "reads" / args.reader
    details = q["answer"] == "details"
    plan = pairs_by_part(cfg, folder, manifest) if details else None
    parts = [p for p in T.part_files(folder) if plan is None or p.stem in plan]
    if args.disputed:
        dis = json.loads((folder / "disputed.json").read_text(encoding="utf-8"))
        parts = [p for p in parts if p.stem in dis["by_part"]]
    if args.parts:
        want = set(args.parts.split(","))
        parts = [p for p in parts if p.stem in want]
    todo = [p for p in parts if not (out_dir / f"{p.stem}.json").exists()]
    if args.limit:
        todo = todo[:args.limit]
    log.info("%s %s: %d parts, %d without an answer to run, %d at a time%s", args.batch, args.reader, len(parts),
             len(todo), parallel, f"; {sum(len(v) for v in plan.values())} pairs" if details else "")
    exe = shutil.which("codex")
    if exe is None and not args.dry_run:
        raise SystemExit("the Codex CLI is not installed (codex)")
    checker = T.Checker(cfg, manifest)
    runs = cfg.work / "read" / f"{args.batch}-{args.reader}"
    if not args.dry_run:
        (out_dir / "invalid").mkdir(parents=True, exist_ok=True)
        if details:
            write_text(out_dir / "asked.json", json.dumps({s: [f"{i} {k}" for i, k in v] for s, v in plan.items()},
                                                          indent=0))
    lock = threading.Lock()
    pause_until = [0.0]
    limited_run = [0]       # usage-limit answers in a row, over all sessions
    counts = {"ok": 0, "invalid": 0, "refused by the check": 0, "not run: usage limit": 0, "gave up": 0}

    def note(rec):
        with lock:
            with open(out_dir / "log.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

    def one(part):
        text = part.read_text(encoding="utf-8")
        problems, _ = checker.part(part.name, text)
        if problems:
            log.warning("%s: not sent, it fails the check (%s)", part.stem, problems[0])
            counts["refused by the check"] += 1
            return
        pairs = plan[part.stem] if details else None
        prompt = prompt_for(head, text, pairs)
        work = runs / part.stem
        last = runs / f"{part.stem}.last.txt"
        cmd = codex_command(cfg, exe or "codex", work, last)
        if args.dry_run:
            write_text(runs / f"{part.stem}.prompt.txt", prompt)
            log.info("%s: %s < %s", part.stem, " ".join(cmd), runs / f"{part.stem}.prompt.txt")
            return
        for attempt in range(1, r["tries"] + 1):
            if limited_run[0] >= r["stop_after"]:
                log.warning("%s: not run, usage limit", part.stem)
                counts["not run: usage limit"] += 1
                return
            wait = pause_until[0] - time.time()
            if wait > 0:
                time.sleep(wait)
            shutil.rmtree(work, ignore_errors=True)
            work.mkdir(parents=True)
            last.unlink(missing_ok=True)
            t0, start = time.time(), time.strftime("%H:%M:%S")
            proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding="utf-8",
                                  errors="replace", timeout=r["timeout"])
            secs = time.time() - t0
            usage, tools, cmds = usage_of(proc.stdout)
            rec = {"part": part.stem, "attempt": attempt, **({"pairs": len(pairs)} if details else {}), "start": start,
                   "end": time.strftime("%H:%M:%S"), "seconds": round(secs, 1), "exit": proc.returncode,
                   "tool_items": tools, "commands": cmds, "usage": usage}
            answer = last.read_text(encoding="utf-8") if last.exists() else ""
            limited = proc.returncode != 0 and LIMIT.search(proc.stdout + proc.stderr)
            limited_run[0] = limited_run[0] + 1 if limited else 0
            if limited:
                rec["rate_limit"] = True
                rec["note"] = (proc.stdout + proc.stderr)[-300:]
                pause_until[0] = max(pause_until[0], time.time() + 120 * attempt)
            if proc.returncode == 0:
                obj, why = validate_details(answer, set(pairs)) if details else validate_marks(answer, text)
            else:
                obj, why = None, f"exit {proc.returncode}"
            rec["invalid"] = why
            note(rec)
            if obj is not None:
                write_text(out_dir / f"{part.stem}.json", json.dumps(obj, indent=1))
                shutil.rmtree(work, ignore_errors=True)
                last.unlink(missing_ok=True)
                log.info("%s: %.0f s, ok", part.stem, secs)
                counts["ok"] += 1
                return
            write_text(out_dir / "invalid" / f"{part.stem}.try{attempt}.txt", f"{why}\n---\n{answer}")
            log.warning("%s: %.0f s, invalid: %s", part.stem, secs, str(why)[:200])
            counts["invalid"] += 1
        log.warning("%s: gave up after %d tries", part.stem, r["tries"])
        counts["gave up"] += 1

    with ThreadPoolExecutor(parallel) as pool:
        list(pool.map(one, todo))
    if not args.dry_run:
        log.info("%s", ", ".join(f"{k} {v}" for k, v in counts.items()))
        if counts["not run: usage limit"]:
            raise SystemExit("stopped on the usage limit; run the same command again later to continue")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("batch", help="the batch: a folder under [transcript] batches")
    p.add_argument("reader", help="the read's name, <question>-<copy> (codex4-a)")
    p.add_argument("--parallel", type=int, default=1, help="Codex sessions at a time (at most the question's "
                   "max_parallel)")
    p.add_argument("--limit", type=int, default=0, help="run at most this many parts")
    p.add_argument("--parts", default="", help="only these parts (stems, comma-separated)")
    p.add_argument("--disputed", action="store_true", help="only the parts the batch's disputed.json lists")
    p.add_argument("--out", type=Path, help="save the answers here instead of BATCH/reads/READER")
    p.add_argument("--dry-run", action="store_true", help="write each prompt and show the command; run nothing")
    run("read", body, p)
