"""uv run backup: copy the whole data folder (the private folder; everything the pipeline cannot rebuild) to the
ecc-data raw repository on Telesphorepo (telesphorepo.toml), with the telesphorepo CLI. `uv run restore`
(ecc.restore) brings it back. What the pipeline rebuilds lives in the work folder ([paths] work) and is never backed
up.

The repository holds one file per path, and a publish over a path replaces it. So each backup goes to a folder of
its own, backups/<UTC time>/, and is uploaded with --if-new, which never replaces a file: a damaged local copy can
never overwrite an earlier backup. latest.txt, the one file that is replaced, names the newest backup folder.

Each backup is complete on its own: one gzipped tar per folder in the data folder, one for the files directly in it,
and manifest.json. Every archived file is named by its path under the data folder, and the manifest lists, per
archive, its SHA-256 and size and, per file, its SHA-256, size and modification time in nanoseconds. Restore puts the
times back, because the pipeline's change markers (core.fingerprint) compare them: a restored folder is current to
every step that was current before.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
from datetime import datetime, timezone

from .core import REPO, log, run, write_text

# Credentials never leave this machine; .env belongs in the repository folder, never in data/.
SECRETS = {".env"}
TOP_FILES = "files"     # the archive of the files directly in the data folder


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def bundles(cfg):
    """{archive name: [paths]}: each folder in the data folder, and the files directly in it."""
    entries = sorted(cfg.data.iterdir())
    out = {p.name: [p] for p in entries if p.is_dir()}
    if TOP_FILES in out:
        raise SystemExit(f"{cfg.data / TOP_FILES} is a folder; that name is the archive of the data folder's files")
    files = [p for p in entries if p.is_file()]
    if files:
        out[TOP_FILES] = files
    return out


def bundle(cfg, out, paths):
    """A tar.gz at out of the files under paths; returns their manifest entries."""
    files = []
    with tarfile.open(out, "w:gz") as tar:
        for path in paths:
            for f in [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file()):
                if f.name in SECRETS:
                    raise SystemExit(f"refusing to back up {f}: credentials stay local")
                name = f.relative_to(cfg.data).as_posix()
                st = f.stat()
                tar.add(f, arcname=name)
                files.append({"name": name, "sha256": sha256(f), "bytes": st.st_size, "mtime_ns": st.st_mtime_ns})
    return files


def body(cfg, args):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    work = cfg.work / "backup-out" / stamp
    work.mkdir(parents=True)
    manifest = {"created": stamp, "bundles": {}}
    for name, paths in bundles(cfg).items():
        out = work / f"{name}.tar.gz"
        log.info("bundling %s", name)
        files = bundle(cfg, out, paths)
        manifest["bundles"][out.name] = {"sha256": sha256(out), "bytes": out.stat().st_size, "files": files}
        log.info("%s: %d files, %.1f MB", out.name, len(files), out.stat().st_size / 1e6)
    write_text(work / "manifest.json", json.dumps(manifest, indent=1) + "\n")
    folder = f"backups/{stamp}"
    uploads = [work / n for n in manifest["bundles"]] + [work / "manifest.json"]
    if args.dry_run:
        log.info("dry run: %s holds the bundles; nothing published", work)
        return
    for f in uploads:
        # one file per call, so a failure names the file
        subprocess.run(["telesphorepo", "publish", "--if-new", "--into", folder, str(f)], cwd=REPO, check=True)
    latest = work / "latest.txt"
    latest.write_text(folder + "\n", encoding="utf-8")
    subprocess.run(["telesphorepo", "publish", "--into", ".", str(latest)], cwd=REPO, check=True)
    log.info("backed up to ecc-data %s (%.1f MB); latest.txt points to it. `uv run restore --check` verifies it",
             folder, sum(f.stat().st_size for f in uploads) / 1e6)
    shutil.rmtree(work)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="build the bundles without publishing them")
    run("backup", body, ap)
