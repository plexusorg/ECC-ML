"""uv run restore: bring a backup made by `uv run backup` (ecc.backup) back from the ecc-data repository.

Downloads go through the telesphorepo CLI (`telesphorepo get`), which reads the private repository as the account
this computer is logged in as (`telesphorepo login`); no token is stored.

It restores the newest backup (latest.txt), or the folder named with --backup. Its manifest names every archive of
the data folder; each is downloaded from the backup folder that holds its bytes (its "folder", else the manifest's
own folder). An archive uploaded in parts (ecc.backup.split) is joined from them, each part checked against its size
and SHA-256. Each archive's SHA-256 is checked against the manifest before anything is written. A file already in the
data folder with the same content is left as it is; when one differs, restore writes nothing and lists them, so it
never overwrites newer work. Each written file is checked against its SHA-256 and gets its recorded modification time
back, so every step that was current at the backup is current after it.

--check downloads and verifies every archive the manifest names, wherever its bytes are, then compares every file in
it with the data folder, and writes nothing: the proof that a backup can bring the data back.
"""
import argparse
import json
import os
import shutil
import tarfile
from pathlib import Path

from .backup import base_url, fetch, sha256, url
from .core import log, run


def download(base, work, name, b):
    """Downloads archive name from its backup folder b["folder"] into work and checks it: an archive in parts is
    joined from them, each part checked against its SHA-256 first. Returns the number of parts (1 for an archive in
    one file)."""
    archive = work / name
    parts = b.get("parts")
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists() and sha256(archive) == b["sha256"]:
        log.info("%s: already downloaded", name)
    elif not parts:
        log.info("downloading %s (%.1f MB) from %s", name, b["bytes"] / 1e6, b["folder"])
        fetch(url(base, b["folder"], name), archive)
    else:
        log.info("downloading %s (%.1f MB) in %d parts from %s", name, b["bytes"] / 1e6, len(parts), b["folder"])
        with open(archive, "wb") as joined:
            for p in parts:
                part = work / p["name"]
                fetch(url(base, b["folder"], p["name"]), part)
                if part.stat().st_size != p["bytes"] or sha256(part) != p["sha256"]:
                    raise SystemExit(f"{b['folder']}/{p['name']}: size or SHA-256 differs from the manifest; the "
                                     "download or the backup is damaged")
                with open(part, "rb") as src:
                    shutil.copyfileobj(src, joined, 1 << 24)
                part.unlink()
    if sha256(archive) != b["sha256"]:
        raise SystemExit(f"{b['folder']}/{name}: SHA-256 differs from the manifest; the download or the backup is "
                         "damaged")
    return len(parts) if parts else 1


def body(cfg, args):
    base = base_url()
    folder = args.backup or fetch(url(base, "latest.txt")).decode().strip()
    manifest = json.loads(fetch(url(base, folder, "manifest.json")))
    for b in manifest["bundles"].values():
        b.setdefault("folder", folder)
    folders = sorted({b["folder"] for b in manifest["bundles"].values()})
    log.info("backup %s: %d archives, their bytes in %d backup folders: %s", folder, len(manifest["bundles"]),
             len(folders), ", ".join(folders))

    work = cfg.work / "restore-download" / Path(folder).name
    work.mkdir(parents=True, exist_ok=True)
    plan, conflicts, same, stored = [], [], 0, 0
    for name, b in manifest["bundles"].items():
        archive = work / name
        stored += download(base, work, name, b)
        for f in b["files"]:
            target = cfg.data / f["name"]
            if target.exists():
                if sha256(target) == f["sha256"]:
                    same += 1
                else:
                    conflicts.append(f["name"])
                continue
            plan.append((archive, f))
    log.info("%d archives (%d files on the server, in %d backup folders) match the manifest's SHA-256",
             len(manifest["bundles"]), stored, len(folders))
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
