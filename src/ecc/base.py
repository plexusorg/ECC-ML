"""The base model the classifier trains from, and the ranker that ranks the training lines.

The base model ([train.classifier] base) is a Hugging Face repo pinned to a commit sha. Its copy lives in
data/models/base/<repo>@<sha>/ with source.json (repo, revision, each file's size and SHA-256); training reads the
copy and never the hub. `path` refuses a copy of another repo or revision, or with a missing or resized file. The
ranker ([train] ranker) is a trained classifier folder whose input.json names the base revision it was trained from.

    uv run python -m ecc.base fetch           download the pinned base model when it has no copy yet
    uv run python -m ecc.base check           the copy's files against their SHA-256, and the ranker's base revision
"""
import argparse
import hashlib
import json
import os
import re
import shutil
from pathlib import Path

from .core import log, run, write_text


def pinned(cfg):
    """(repo, revision, files) of the base model, from [train.classifier]. The revision must be a full commit sha: a
    branch such as main names whatever the repo holds on the day of the download."""
    t = cfg["train"]["classifier"]
    repo, rev = t["base"], t["revision"]
    if not re.fullmatch(r"[0-9a-f]{40}", rev):
        raise SystemExit(f"[train.classifier] revision {rev!r} is not a commit sha")
    return repo, rev, list(t["files"])


def folder(cfg):
    repo, rev, _ = pinned(cfg)
    return cfg.data / "models" / "base" / f"{repo}@{rev}"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def path(cfg):
    """The base model's copy for from_pretrained, after checking it is the pinned repo and revision with its files at
    their recorded sizes."""
    repo, rev, files = pinned(cfg)
    d = folder(cfg)
    src = d / "source.json"
    if not src.exists():
        raise SystemExit(f"{d} holds no copy of {repo}@{rev}; run `uv run python -m ecc.base fetch`")
    s = json.loads(src.read_text(encoding="utf-8"))
    if (s["repo"], s["revision"]) != (repo, rev):
        raise SystemExit(f"{d} is {s['repo']}@{s['revision']}, not the pinned {repo}@{rev}")
    if sorted(s["files"]) != sorted(files):
        raise SystemExit(f"{d} holds {sorted(s['files'])}, not [train.classifier] files {sorted(files)}")
    for name, rec in s["files"].items():
        p = d / name
        if not p.is_file() or p.stat().st_size != rec["size"]:
            raise SystemExit(f"{p} is missing or not the {rec['size']:,} bytes fetched")
    return d


def ranker(cfg):
    """The trained classifier that ranks the training lines ([train] ranker), after checking its input.json names the
    pinned [train.classifier] base revision."""
    d = cfg.path(cfg["train"]["ranker"])
    meta = d / "input.json"
    if not meta.exists():
        raise SystemExit(f"[train] ranker {d} has no input.json")
    base = json.loads(meta.read_text(encoding="utf-8"))["base"]
    repo, rev, _ = pinned(cfg)
    if (base["repo"], base["revision"]) != (repo, rev):
        raise SystemExit(f"[train] ranker {d} was trained from {base['repo']}@{base['revision']}, not {repo}@{rev}")
    return d


def fetch(cfg):
    """Downloads the pinned files of the base model into its copy folder, through a cache of its own in the work
    folder, so the user's Hugging Face cache (and its refs) is never written. Refuses when the copy exists."""
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    from huggingface_hub import HfApi, snapshot_download
    repo, rev, files = pinned(cfg)
    d = folder(cfg)
    if d.exists():
        raise SystemExit(f"{d} exists; `check` verifies it")
    hub = {s.rfilename: s for s in HfApi().model_info(repo, revision=rev, files_metadata=True).siblings}
    missing = [f for f in files if f not in hub]
    if missing:
        raise SystemExit(f"{repo}@{rev} has no {', '.join(missing)}")
    tmp = cfg.work / "base-fetch"
    shutil.rmtree(tmp, ignore_errors=True)
    got = Path(snapshot_download(repo, revision=rev, allow_patterns=files, local_dir=tmp / "files",
                                 cache_dir=tmp / "cache"))
    src = {"repo": repo, "revision": rev,
           "files": {f: {"size": (got / f).stat().st_size, "sha256": sha256(got / f)} for f in files}}
    for f, r in src["files"].items():      # the hub's size, and its SHA-256 of a large (LFS) file
        h = hub[f]
        if r["size"] != h.size or (h.lfs and r["sha256"] != h.lfs.sha256):
            raise SystemExit(f"{repo}@{rev} {f}: downloaded {r['size']} bytes, sha256 {r['sha256']}; the hub lists "
                             f"{h.size} bytes{f', sha256 {h.lfs.sha256}' if h.lfs else ''}")
    d.mkdir(parents=True)
    for f in files:
        shutil.move(got / f, d / f)
    write_text(d / "source.json", json.dumps(src, indent=1) + "\n")
    shutil.rmtree(tmp)
    for f, r in src["files"].items():
        log.info("%s@%s: %s %s bytes, sha256 %s", repo, rev[:8], f, f"{r['size']:,}", r["sha256"])
    log.info("copied to %s; run `uv run backup`", d)


def check(cfg):
    d = path(cfg)
    s = json.loads((d / "source.json").read_text(encoding="utf-8"))
    bad = [f for f, r in s["files"].items() if sha256(d / f) != r["sha256"]]
    if bad:
        raise SystemExit(f"{d}: {', '.join(bad)} differ from the SHA-256 fetched")
    log.info("%s: %d files as fetched", d, len(s["files"]))
    log.info("ranker %s", ranker(cfg))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch", help="download the pinned base model when it has no copy yet")
    sub.add_parser("check", help="the copy's files against their SHA-256, and the ranker's base revision")

    def body(cfg, args):
        if args.cmd == "check":
            return check(cfg)
        fetch(cfg)

    run("base", body, ap)


if __name__ == "__main__":
    main()
