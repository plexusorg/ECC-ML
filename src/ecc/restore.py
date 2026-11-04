"""uv run restore: bring a backup made by `uv run backup` (ecc.backup) back from the ecc-data repository.

Downloads need a read token of the repository (Telesphorepo: Settings > Tokens > New read token), given as
TELESPHOREPO_USER and TELESPHOREPO_READ_TOKEN in .env or the environment.

It restores the newest backup (latest.txt), or the folder named with --backup. Each bundle's SHA-256 is checked
against the manifest before anything is written. A file already in the data folder with the same content is left as
it is; when one differs, restore writes nothing and lists them, so it never overwrites newer work. Each written file
is checked against its SHA-256 and gets its recorded modification time back, so every step that was current at the
backup is current after it.

--check downloads and verifies the backup, then compares every file in it with the data folder, and writes nothing:
the proof that a backup can bring the data back.
"""
import argparse
import base64
import json
import os
import shutil
import tarfile
import tomllib
import urllib.request
from pathlib import Path

from .backup import sha256
from .core import REPO, log, run


def fetch(url, auth, dest=None):
    """The body of url, or with dest, the body streamed to that file."""
    req = urllib.request.Request(url, headers={"Authorization": f"Basic {auth}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        if dest is None:
            return r.read()
        with open(dest, "wb") as f:
            while block := r.read(1 << 20):
                f.write(block)


def body(cfg, args):
    cfg.load_env()
    user, token = os.environ.get("TELESPHOREPO_USER"), os.environ.get("TELESPHOREPO_READ_TOKEN")
    if not user or not token:
        raise SystemExit("set TELESPHOREPO_USER and TELESPHOREPO_READ_TOKEN (a read token of ecc-data) in .env")
    auth = base64.b64encode(f"{user}:{token}".encode()).decode()
    project = tomllib.loads((REPO / "telesphorepo.toml").read_text(encoding="utf-8"))
    base = f"{project['server'].rstrip('/')}/raw/{project['repo']}"
    folder = args.backup or fetch(f"{base}/latest.txt", auth).decode().strip()
    manifest = json.loads(fetch(f"{base}/{folder}/manifest.json", auth))
    log.info("backup %s: %d bundles", folder, len(manifest["bundles"]))

    work = cfg.work / "restore-download" / Path(folder).name
    work.mkdir(parents=True, exist_ok=True)
    plan, conflicts, same = [], [], 0
    for name, b in manifest["bundles"].items():
        archive = work / name
        if not archive.exists() or sha256(archive) != b["sha256"]:
            log.info("downloading %s (%.1f MB)", name, b["bytes"] / 1e6)
            fetch(f"{base}/{folder}/{name}", auth, archive)
        if sha256(archive) != b["sha256"]:
            raise SystemExit(f"{name}: SHA-256 differs from the manifest; the download or the backup is damaged")
        for f in b["files"]:
            target = cfg.data / f["name"]
            if target.exists():
                if sha256(target) == f["sha256"]:
                    same += 1
                else:
                    conflicts.append(f["name"])
                continue
            plan.append((archive, f))
    log.info("%d files match the data folder, %d are missing, %d differ", same, len(plan), len(conflicts))
    if conflicts:
        log.warning("differing files (first 20): %s", ", ".join(conflicts[:20]))
    if args.check:
        log.info("check only: nothing written; downloads kept in %s", work)
        return
    if conflicts:
        raise SystemExit("files in the data folder differ from the backup; nothing restored. Move them aside first")
    by_archive = {}
    for archive, f in plan:
        by_archive.setdefault(archive, {})[f["name"]] = f
    for archive, wanted in by_archive.items():
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar:
                f = wanted.get(member.name)
                if f is None:
                    continue
                target = cfg.data / f["name"]
                target.parent.mkdir(parents=True, exist_ok=True)
                tmp = target.with_name(target.name + ".restoring")
                with tar.extractfile(member) as src, open(tmp, "wb") as dst:
                    while block := src.read(1 << 20):
                        dst.write(block)
                if sha256(tmp) != f["sha256"]:
                    raise SystemExit(f"{f['name']}: restored content differs from the manifest")
                os.utime(tmp, ns=(f["mtime_ns"], f["mtime_ns"]))
                os.replace(tmp, target)
    log.info("restored %d files from %s", len(plan), folder)
    shutil.rmtree(work)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backup", help="the backup folder, such as backups/20261104T144713Z (default: latest.txt)")
    ap.add_argument("--check", action="store_true", help="verify the backup against the data folder; write nothing")
    run("restore", body, ap)
