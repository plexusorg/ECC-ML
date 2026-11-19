"""uv run backup: copy the data folder (the private folder; everything the pipeline cannot rebuild), less the base
model (SKIPPED), to the ecc-data raw repository on Telesphorepo (telesphorepo.toml), with the telesphorepo CLI.
`uv run restore` (ecc.restore) brings it back. What the pipeline rebuilds lives in the work folder ([paths] work) and is never backed
up.

The repository holds one file per path, and a publish over a path replaces it. So each backup goes to a folder of
its own, backups/<UTC time>/, and is uploaded with --if-new, which never replaces a file: a damaged local copy can
never overwrite an earlier backup. latest.txt, the one file that is replaced, names the newest backup folder; it is
published last, after every file of the backup is on the server.

Each backup's manifest.json describes the whole data folder in gzipped tars (units). A folder whose files total at
most UNIT bytes is one unit, <folder>.tar.gz; a larger one (and the data folder itself) is split: each subfolder by
the same rule, each file over UNIT a unit of its own, <file>.tar.gz, and the other files directly in it one unit,
<folder>/files.tar.gz (files.tar.gz for the data folder). So a change uploads the unit that holds it: one batch
folder, one model file, a folder's small files. Archive names are paths under the backup folder. Every archived file
is named by its path under the data folder, and the manifest lists, per archive, its SHA-256 and size and, per file,
its SHA-256, size and modification time in nanoseconds. Restore puts the times back, because the pipeline's change
markers (core.fingerprint) compare them: a restored folder is current to every step that was current before.

A backup uploads only the archives that changed. Archives are built deterministically (sorted paths, tar headers of
name, size and modification time only, a gzip header without a time or name), so unchanged files give the same
bytes. An archive whose SHA-256 equals its entry in the manifest latest.txt names is not uploaded again: its entry
keeps that entry's "folder", the backup folder that holds its bytes, and its parts. An entry without "folder" has its
bytes in the manifest's own folder. Restore reads each archive from its folder, so one manifest restores the whole
data folder; a backup folder stays on the server while a later manifest names it. --full uploads every archive.

Telesphorepo offers no way to put a stored file at a second path without sending its bytes again (a raw repository
writes only a PUT body or a resumable upload; a write by digest serves container image mounts and an admin's merge
of repositories only: internal/format/kit.go, internal/api/pipeline.go, internal/assets/commit.go), and its
preconditions (--if-new, --if-match) are checked after the bytes arrive. So an unchanged archive is named by the
folder that holds it, never copied.

An archive over PART bytes is uploaded as parts of PART bytes, <archive>.000, .001, ..., each in a publish of its
own, and its manifest entry lists them with their SHA-256 and size; restore joins them. Two limits of the server make
this necessary (Telesphorepo's source: cmd/telesphorepo/publish.go and upload.go, internal/publish/api.go and
sessions.go, internal/settings/general.go):
- Each `telesphorepo publish` mints one publish session, which ends publish.session_ttl after it starts (15 minutes
  by default, at most 1 hour) and cannot be renewed. Every chunk of an upload needs it, so an upload still running
  then fails with "Unauthorized: publishing needs a publish session", and the CLI warns "Cannot close publish
  session ...; it ends at <time>". At this connection's 2.5 MB/s, a 2.3 GB archive outlives a 15-minute session.
- The server refuses one upload over max_upload_size (2 GiB by default) with 413 "upload too large".
A 256 MiB part uploads in under 2 minutes at 2.5 MB/s, and within a 15-minute session down to 0.3 MB/s.

The CLI sends a file under 64 MiB in one PUT and does not try it again when the connection drops (upload.go: putFile;
only the chunks of a resumable upload are tried again), so backup tries a failed publish again (publish), and restore
a failed download (fetch). Every line telesphorepo prints goes to the backup's log, so a failed publish's message is
kept.
"""
import argparse
import gzip
import hashlib
import json
import posixpath
import shutil
import subprocess
import tarfile
import time
import tomllib
from datetime import datetime, timezone
from urllib.parse import quote

from .core import REPO, log, run, write_text

# Credentials never leave this machine; .env belongs in the repository folder, never in data/.
SECRETS = {".env"}
# Folders under the data folder that are left out: the pinned base model, which `uv run python -m ecc.base fetch`
# downloads again.
SKIPPED = {"models/base"}
FILES = "files.tar.gz"  # the unit of a split folder's own small files
UNIT = 64 << 20         # the most file bytes a folder unit holds before it is split
PART = 256 << 20        # the largest file one publish uploads
TRIES, WAIT = 4, 60     # a failed download or publish is tried TRIES times, WAIT seconds apart


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def telesphorepo(*args, stdout=False, check=True):
    """Runs the telesphorepo CLI in the repository folder and logs every line it prints. Returns its standard output
    (with stdout, the bytes of `get -o -`, which are not logged). When it fails it stops, or without check returns
    None."""
    got = subprocess.run(["telesphorepo", *args], cwd=REPO, capture_output=True)
    lines = got.stderr.decode("utf-8", "replace").splitlines()
    if not stdout:
        lines = got.stdout.decode("utf-8", "replace").splitlines() + lines
    for line in lines:
        if line.strip():
            (log.info if not got.returncode else log.error if check else log.warning)("telesphorepo: %s", line.rstrip())
    if got.returncode:
        if check:
            raise SystemExit(f"telesphorepo {' '.join(args)} failed (exit {got.returncode}); its messages are above")
        return None
    return got.stdout


def base_url():
    """The URL of the ecc-data raw repository, from telesphorepo.toml."""
    project = tomllib.loads((REPO / "telesphorepo.toml").read_text(encoding="utf-8"))
    return f"{project['server'].rstrip('/')}/raw/{project['repo']}"


def url(base, *path):
    """The URL of the file at the repository path joined from path, each segment quoted."""
    return "/".join([base, *(quote(p) for p in path)])


def fetch(link, dest=None, check=True, tries=TRIES):
    """The body of link, or with dest, the file the CLI downloads it to (written only when the whole body arrived;
    the return value is then not None). A failed download is tried tries times, WAIT seconds apart, then it stops,
    or without check returns None."""
    for attempt in range(1, tries + 1):
        got = telesphorepo("get", "-o", str(dest) if dest else "-", link, stdout=not dest,
                           check=check and attempt == tries)
        if got is not None:
            return got
        if attempt < tries:
            log.warning("downloading %s failed (try %d of %d); trying again in %d s", link, attempt, tries, WAIT)
            time.sleep(WAIT)
    return None


def publish(base, path, into, replace=False):
    """Publishes the file path into the repository folder into, with --if-new unless replace. A failed publish is
    tried TRIES times, WAIT seconds apart. A connection can drop after the server stored the file, and a second
    --if-new publish is then refused; so after a failure the file on the server is downloaded and compared first."""
    for attempt in range(1, TRIES + 1):
        if telesphorepo("publish", *([] if replace else ["--if-new"]), "--into", into, str(path),
                        check=False) is not None:
            return
        landed = path.with_name(path.name + ".landed")
        if not replace and fetch(url(base, into, path.name), landed, check=False, tries=1) is not None:
            same = sha256(landed) == sha256(path)
            landed.unlink()
            if same:
                log.info("%s/%s is on the server with the same SHA-256", into, path.name)
                return
            raise SystemExit(f"{into}/{path.name} is on the server with other bytes; the backup stops")
        if attempt < TRIES:
            log.warning("publishing %s failed (try %d of %d); trying again in %d s", path.name, attempt, TRIES, WAIT)
            time.sleep(WAIT)
    raise SystemExit(f"publishing {path} into {into} failed {TRIES} times; the messages are above")

def latest(base):
    """(folder, manifest) of the backup latest.txt names."""
    folder = fetch(url(base, "latest.txt")).decode().strip()
    return folder, json.loads(fetch(url(base, folder, "manifest.json")))


def units(cfg):
    """{archive name: [files]}, the units of the data folder (module docstring), each list sorted."""
    out = {}

    def add(name, files):
        if name in out:
            raise SystemExit(f"two units of the data folder are both named {name}; rename the folder or file")
        out[name] = files

    def kept(p):
        rel = p.relative_to(cfg.data).as_posix()
        return not any(rel == s or rel.startswith(s + "/") for s in SKIPPED)

    def walk(folder):
        rel = folder.relative_to(cfg.data).as_posix() if folder != cfg.data else ""
        small = []
        for p in sorted(folder.iterdir()):
            if not kept(p):
                continue
            if p.is_dir():
                files = sorted(f for f in p.rglob("*") if f.is_file() and kept(f))
                if not files:
                    continue
                if sum(f.stat().st_size for f in files) <= UNIT:
                    add(p.relative_to(cfg.data).as_posix() + ".tar.gz", files)
                else:
                    walk(p)
            elif p.stat().st_size > UNIT:
                add(p.relative_to(cfg.data).as_posix() + ".tar.gz", [p])
            elif p.is_file():
                small.append(p)
        if small:
            add(f"{rel}/{FILES}" if rel else FILES, small)

    walk(cfg.data)
    return out


def bundle(cfg, out, files):
    """A tar.gz at out of files, in their order; returns their manifest entries. The bytes depend only on the files'
    paths, contents and modification times (whole seconds): each tar header holds the name, size and time, with fixed
    mode, owner and type, and the gzip header has no time and no name."""
    entries = []
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0) as gz, \
            tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT, encoding="utf-8") as tar:
        for f in files:
            if f.name in SECRETS:
                raise SystemExit(f"refusing to back up {f}: credentials stay local")
            name = f.relative_to(cfg.data).as_posix()
            st = f.stat()
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.mode = st.st_size, st.st_mtime_ns // 1_000_000_000, 0o644
            with open(f, "rb") as src:
                tar.addfile(info, src)
            entries.append({"name": name, "sha256": sha256(f), "bytes": st.st_size, "mtime_ns": st.st_mtime_ns})
    return entries


def split(archive, name):
    """Cuts archive (the unit name) into parts of PART bytes beside it (<name>.000, .001, ...) and removes it; returns
    the parts' manifest entries."""
    parts = []
    with open(archive, "rb") as f:
        while block := f.read(PART):
            part = archive.with_name(f"{archive.name}.{len(parts):03d}")
            part.write_bytes(block)
            parts.append({"name": f"{name}.{len(parts):03d}", "sha256": hashlib.sha256(block).hexdigest(),
                          "bytes": len(block)})
    archive.unlink()
    return parts


def body(cfg, args):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder = f"backups/{stamp}"
    base = base_url()
    if args.full:
        last, before = None, {}
        log.info("--full: every archive is uploaded")
    else:
        last, manifest = latest(base)
        before = manifest["bundles"]
        log.info("latest backup %s: %d archives; an archive with the same SHA-256 is not uploaded again", last,
                 len(before))
    work = cfg.work / "backup-out" / stamp
    work.mkdir(parents=True)
    manifest = {"created": stamp, "bundles": {}}
    uploads, kept = [], 0   # uploads: the names of the files to publish under the backup folder
    for name, paths in units(cfg).items():
        out = work / name
        files = bundle(cfg, out, paths)
        entry = manifest["bundles"][name] = {"sha256": sha256(out), "bytes": out.stat().st_size}
        old = before.get(name)
        if old and old["sha256"] == entry["sha256"] and old["bytes"] == entry["bytes"]:
            entry["folder"] = old.get("folder", last)
            if "parts" in old:
                entry["parts"] = old["parts"]
            out.unlink()
            kept += entry["bytes"]
            log.info("%s: %d files, %.1f MB, unchanged: its bytes are in %s", name, len(files), entry["bytes"] / 1e6,
                     entry["folder"])
        elif entry["bytes"] > PART:
            entry["parts"] = split(out, name)
            uploads += [p["name"] for p in entry["parts"]]
            log.info("%s: %d files, %.1f MB in %d parts", name, len(files), entry["bytes"] / 1e6, len(entry["parts"]))
        else:
            uploads.append(name)
            log.info("%s: %d files, %.1f MB", name, len(files), entry["bytes"] / 1e6)
        entry["files"] = files
    write_text(work / "manifest.json", json.dumps(manifest, indent=1) + "\n")
    uploads.append("manifest.json")
    total = sum((work / n).stat().st_size for n in uploads)
    log.info("%d archives; %.1f MB in %d files to upload; %.1f MB unchanged in earlier backups",
             len(manifest["bundles"]), total / 1e6, len(uploads), kept / 1e6)
    if args.dry_run:
        log.info("dry run: %s holds the archives to upload and the manifest; nothing published", work)
        return
    sent = 0
    for n in uploads:
        # one publish per file: each gets a session of its own, and a failure names the file
        size = (work / n).stat().st_size
        log.info("uploading %s (%.1f MB; %.0f of %.0f MB sent)", n, size / 1e6, sent / 1e6, total / 1e6)
        into = posixpath.dirname(n)
        publish(base, work / n, f"{folder}/{into}" if into else folder)
        sent += size
    pointer = work / "latest.txt"
    pointer.write_text(folder + "\n", encoding="utf-8")
    publish(base, pointer, ".", replace=True)
    log.info("backed up to ecc-data %s (%.1f MB uploaded in %d files, %.1f MB unchanged); latest.txt points to it. "
             "`uv run restore --check` verifies it", folder, total / 1e6, len(uploads), kept / 1e6)
    shutil.rmtree(work)

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="build the archives and the manifest and log what would be uploaded; publish nothing")
    ap.add_argument("--full", action="store_true",
                    help="upload every archive, also those the latest backup holds unchanged")
    run("backup", body, ap)
