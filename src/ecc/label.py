"""uv run label: label every shard and regrade set with the Codex CLI, up to 8 at a time, validating each output.

A shard is done when its output validates; a rerun labels only missing or invalid shards. Interrupted shards are
discarded whole and relabelled next time. `--select-regrade CUTOFF` writes regrade sets for the known error groups
of shards labelled before CUTOFF (when the settled policy reached Codex); `--select-unsure` writes a set of every
window whose current label is marked unsure. The next `uv run label` labels them.
"""
import argparse
import json
import random
import re
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from .chat import load_windows, render_window, terms
from .core import fingerprint, log, progress, run, save_state, load_state, write_text

LABELS = {"ban", "amb"}
RULES = {"sexual-aimed", "explicit-content", "pushing", "continuing", "sexual-threat", "minors", "grooming",
         "stated-minor-flirt"}
ROLES = {"offender", "at_risk"}
CAPACITY = re.compile(r"at capacity", re.I)
REGRADES = ("regrade", "unsure")    # regrade sets in merge precedence, each above the ones before it


def label_sets(cfg):
    """(name, index file, shard dir, output dir) relative to data/label, first pass first and then the regrade sets in
    precedence order; a regrade set exists once selected."""
    label = cfg.data / "label"
    sets = [("shards", label / "shards.json", "shards", "out")]
    for name in REGRADES:
        if (label / name / f"{name}.json").exists():
            sets.append((name, label / name / f"{name}.json", name, f"{name}_out"))
    return sets


def read_outputs(directory, names):
    """{window id: [outputs]} from the named outputs, oldest file first."""
    out = {}
    paths = sorted((directory / f"{n}.jsonl" for n in names), key=lambda p: p.stat().st_mtime)
    for p in paths:
        for line in open(p, encoding="utf-8-sig"):
            o = json.loads(line)
            out.setdefault(o["id"], []).append(o)
    return out


def check(path, want, windows):
    """None when the output is valid, else the problem."""
    if not path.exists():
        return "missing"
    got, problems = set(), []
    for n, line in enumerate(open(path, encoding="utf-8-sig"), 1):
        if not line.strip():
            continue
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            problems.append(f"line {n} not JSON")
            continue
        wid = o.get("id")
        if wid not in want:
            problems.append(f"unknown window {wid}")
            continue
        if wid in got:
            problems.append(f"window {wid} twice")
        got.add(wid)
        rows = {r for r, _, _ in windows[wid]["lines"]}
        for f in o.get("flags", []):
            if f.get("row") not in rows:
                problems.append(f"window {wid} row {f.get('row')} not in window")
            elif f.get("label") not in LABELS or f.get("rule") not in RULES:
                problems.append(f"window {wid} bad label/rule {f.get('label')}/{f.get('rule')}")
        for p, role in (o.get("roles") or {}).items():
            if role not in ROLES:
                problems.append(f"window {wid} bad role {p}={role}")
    missing = [w for w in want if w not in got]
    if missing:
        problems.insert(0, f"{len(missing)} windows missing")
    return "; ".join(problems[:3]) or None


def scan(cfg, windows=None):
    """Validate every set. Returns ({set: (index, valid names, invalid {name: problem})}, windows).
    Outputs whose fingerprint matches the last validation are trusted without reading windows.jsonl."""
    cache = (load_state(cfg, "label") or {}).get("valid", {})
    label = cfg.data / "label"
    result, valid_fp = {}, {}
    for name, index_path, _, out_dir in label_sets(cfg):
        index = json.loads(index_path.read_text(encoding="utf-8"))
        valid, invalid = [], {}
        for shard, want in index.items():
            out = label / out_dir / f"{shard}.jsonl"
            key = f"{out_dir}/{shard}"
            fp = fingerprint(out)
            if fp is not None and cache.get(key) == fp:
                valid.append(shard)
                valid_fp[key] = fp
                continue
            if fp is not None and windows is None:
                windows = load_windows(cfg.data / "windows.jsonl")
            problem = check(out, set(want), windows) if fp is not None else "missing"
            if problem:
                invalid[shard] = problem
            else:
                valid.append(shard)
                valid_fp[key] = fp
        result[name] = (index, valid, invalid)
    return result, windows, valid_fp


def sync_instructions(cfg):
    label = cfg.data / "label"
    shutil.copyfile(cfg.data / "labeling" / "LABELING.md", label / "LABELING.md")
    shutil.copyfile(cfg.data / "policy.md", label / "policy.md")


def codex_command(cfg, shard_dir, out_dir, shard):
    exe = shutil.which("codex")
    if not exe:
        raise SystemExit("the Codex CLI (codex) is not on PATH")
    lc = cfg["label"]
    return [exe, "exec", "--skip-git-repo-check", "-m", lc["model"], "-c", f'model_reasoning_effort="{lc["effort"]}"',
            "--sandbox", "workspace-write", "-C", str(cfg.data / "label"),
            f"Follow LABELING.md and policy.md in this directory. Label every window in {shard_dir}/{shard}.txt and "
            f"write the JSONL to {out_dir}/{shard}.jsonl."]


def kill_tree(proc):
    subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)


def label_all(cfg, todo, windows):
    """todo: [(set name, shard, want ids, shard dir, out dir)]. Labels them in parallel; returns the failures."""
    label = cfg.data / "label"
    stop = threading.Event()
    running, lock = {}, threading.Lock()
    retries = cfg["label"]["capacity_retries"]

    def one(item):
        _, shard, want, shard_dir, out_dir = item
        out = label / out_dir / f"{shard}.jsonl"
        logs = label / "logs" if out_dir == "out" else label / out_dir / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        (label / out_dir).mkdir(parents=True, exist_ok=True)
        problem = "not run"
        for attempt in range(retries + 1):
            if stop.is_set():
                return shard, "stopped"
            out.unlink(missing_ok=True)
            log_file = logs / f"{shard}.log"
            with open(log_file, "w", encoding="utf-8") as lf:
                proc = subprocess.Popen(codex_command(cfg, shard_dir, out_dir, shard), stdin=subprocess.DEVNULL,
                                        stdout=lf, stderr=subprocess.STDOUT, cwd=label)
                with lock:
                    running[shard] = proc
                code = proc.wait()
                with lock:
                    running.pop(shard, None)
            if stop.is_set():
                out.unlink(missing_ok=True)  # an interrupted shard is discarded whole
                return shard, "stopped"
            problem = check(out, set(want), windows)
            if problem is None:
                return shard, None
            text = log_file.read_text(encoding="utf-8", errors="replace")
            if not CAPACITY.search(text):
                return shard, f"exit {code}: {problem}"
            wait = min(300, 15 * 2 ** attempt)
            log.debug("%s: model at capacity, retry %d in %d s", shard, attempt + 1, wait)
            for _ in range(wait):
                if stop.is_set():
                    return shard, "stopped"
                time.sleep(1)
        return shard, f"model at capacity after {retries} retries: {problem}"

    failures = {}
    pool = ThreadPoolExecutor(cfg["label"]["parallel"])
    try:
        with progress("labeling", len(todo), "shards") as advance:
            futures = [pool.submit(one, item) for item in todo]
            for fut in as_completed(futures):
                shard, problem = fut.result()
                if problem:
                    failures[shard] = problem
                    log.debug("%s: %s", shard, problem)
                advance()
    except KeyboardInterrupt:
        stop.set()
        with lock:
            for proc in running.values():
                kill_tree(proc)
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    pool.shutdown()
    return failures


def select_regrade(cfg, cutoff, windows):
    """Likely misses (lures, pro-pedophile, age plus flirting left ok) and likely over-bans (ban as sexual-aimed,
    explicit-content or sexual-threat on a line with no sexual or risk vocabulary) among first-pass outputs written
    before the cutoff, plus the blind-audit windows when an audit sample exists."""
    label = cfg.data / "label"
    size = cfg["candidates"]["shard_size"]
    cut = datetime.fromisoformat(cutoff).timestamp()
    old = {}
    for p in sorted((label / "out").glob("shard_*.jsonl")):
        if p.stat().st_mtime < cut:
            for line in open(p, encoding="utf-8-sig"):
                o = json.loads(line)
                old.setdefault(o["id"], o)
    miss, over = set(), set()
    tm = terms(cfg)
    for wid, o in old.items():
        text = {r: t for r, _, t in windows[wid]["lines"]}
        flagged = {f["row"]: f for f in o.get("flags", [])}
        if any(r not in flagged and tm["miss"].search(t) for r, t in text.items()):
            miss.add(wid)
        if any(f["label"] == "ban" and f.get("rule") in {"sexual-aimed", "explicit-content", "sexual-threat"}
               and f["row"] in text and not tm["risk"].search(text[f["row"]])
               and not tm["sexual"].search(text[f["row"]])
               for f in flagged.values()):
            over.add(wid)
    audit = label / "audit" / "audit_ids.json"
    groups = {"miss": sorted(miss), "over": sorted(over - miss),
              "audit": json.loads(audit.read_text(encoding="utf-8")) if audit.exists() else []}
    write_set(cfg, "regrade", groups, windows)
    log.info("%d windows labelled before %s: likely misses in %d, likely over-bans in %d; regrade sets: %s",
             len(old), cutoff, len(miss), len(over), {g: -(-len(v) // size) for g, v in groups.items()})


def select_unsure(cfg, windows):
    """Every window whose current label (the latest first-pass output, or a regrade's) is marked unsure, in a seeded
    shuffle as in the first pass, so any run of its shards is a sample of the whole set."""
    sets, _, _ = scan(cfg, windows)
    current = {}
    for name, _, _, out_dir in label_sets(cfg):
        for wid, versions in read_outputs(cfg.data / "label" / out_dir, sets[name][1]).items():
            current[wid] = versions[-1]
    ids = sorted(wid for wid, o in current.items() if o.get("unsure"))
    random.Random(11).shuffle(ids)
    write_set(cfg, "unsure", {"unsure": ids}, windows)
    log.info("%d of %d labelled windows are unsure; %d shards", len(ids), len(current),
             -(-len(ids) // cfg["candidates"]["shard_size"]))


def write_set(cfg, set_name, groups, windows):
    """Writes the regrade set data/label/<set_name>: shards of each group's windows and their index."""
    size = cfg["candidates"]["shard_size"]
    out = cfg.data / "label" / set_name
    index = {}
    for group, gids in groups.items():
        for k in range(0, len(gids), size):
            name = f"{group}_{k // size:04d}"
            chunk = gids[k:k + size]
            write_text(out / f"{name}.txt", "\n\n".join(render_window(windows[i]) for i in chunk) + "\n")
            index[name] = chunk
    write_text(out / f"{set_name}.json", json.dumps(index))


def body(cfg, args):
    if not (cfg.data / "label" / "shards.json").exists():
        raise SystemExit("no shards yet; run `uv run candidates` first")
    windows = None
    for set_name, selected in (("regrade", args.select_regrade), ("unsure", args.select_unsure)):
        if selected and (cfg.data / "label" / set_name / f"{set_name}.json").exists():
            raise SystemExit(f"the {set_name} set already exists (data/label/{set_name}); move it away to select anew")
    if args.select_regrade or args.select_unsure:
        windows = load_windows(cfg.data / "windows.jsonl")
    if args.select_regrade:
        select_regrade(cfg, args.select_regrade, windows)
    if args.select_unsure:
        select_unsure(cfg, windows)
    sets, windows, valid_fp = scan(cfg, windows)
    todo = []
    dirs = {name: (shard_dir, out_dir) for name, _, shard_dir, out_dir in label_sets(cfg)}
    for name, (index, valid, invalid) in sets.items():
        log.info("%s: %d of %d labelled", name, len(valid), len(index))
        todo += [(name, s, index[s], *dirs[name]) for s in sorted(invalid)]
    if args.limit is not None and len(todo) > args.limit:
        log.info("labeling %d of the %d shards left (--limit)", args.limit, len(todo))
        todo = todo[:args.limit]
    if todo and args.check:
        for name, shard, *_ in todo:
            log.info("%s: %s", shard, sets[name][2][shard])
    elif todo:
        sync_instructions(cfg)
        if windows is None:
            windows = load_windows(cfg.data / "windows.jsonl")
        failures = label_all(cfg, todo, windows)
        sets, _, valid_fp = scan(cfg, windows)
        for shard, problem in sorted(failures.items()):
            log.warning("%s: %s", shard, problem)
        if failures:
            log.warning("%d shards failed; run `uv run label` again to retry them", len(failures))
    save_state(cfg, "label", [cfg.data / "label" / "shards.json"], valid=valid_fp,
               counts={name: [len(v), len(i)] for name, (i, v, _) in sets.items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--select-regrade", metavar="CUTOFF", help="ISO time; write regrade sets for shards labelled before it")
    ap.add_argument("--select-unsure", action="store_true", help="write a regrade set of every window labelled unsure")
    ap.add_argument("--limit", type=int, metavar="N", help="label at most N of the shards left")
    ap.add_argument("--check", action="store_true", help="validate outputs and list what is left; start no Codex run")
    run("label", body, ap)
