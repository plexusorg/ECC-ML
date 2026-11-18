"""Configuration, logging, progress bars, what each output was made from, and crash-safe file writes shared by every
step."""
import json
import hashlib
import logging
import os
import signal
import sys
import time
import tomllib
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import BarColumn, MofNCompleteColumn, Progress, ProgressColumn, TextColumn, TimeRemainingColumn
from rich.text import Text

REPO = Path(__file__).resolve().parents[2]
console = Console()
log = logging.getLogger("ecc")


class Config:
    """ecc.toml in the repository, overlaid by <data>/local.toml (this machine's paths and private settings).

    data is the private folder: everything the pipeline cannot rebuild, all of it backed up. work ([paths] work in
    local.toml, outside the repository) holds what the pipeline rebuilds: models, bundles, eval scores, the train and
    val tables, logs and tools. Nothing in work is backed up. The environment variables ECC_CONFIG
    and ECC_DATA name another config file and data folder, to run the steps against a copy."""

    def __init__(self):
        path = Path(os.environ.get("ECC_CONFIG") or REPO / "ecc.toml")
        self.file = path
        self.raw = tomllib.loads(path.read_text(encoding="utf-8"))
        self.data = self._abs(os.environ.get("ECC_DATA") or self.raw["data"], path.parent)
        local = self.data / "local.toml"
        if local.exists():
            _merge(self.raw, tomllib.loads(local.read_text(encoding="utf-8")))
        self.paths = {k: str(v) for k, v in self.raw.get("paths", {}).items()}
        if "work" not in self.paths:
            raise SystemExit(f"set [paths] work, a folder outside the repository, in {self.data / 'local.toml'}")
        self.work = Path(self.paths["work"])

    @staticmethod
    def _abs(p, base):
        p = Path(p)
        return p if p.is_absolute() else (base / p).resolve()

    def __getitem__(self, key):
        return self.raw[key]

    def path(self, value):
        """A configured path: {data}, {repo} and [paths] keys (such as {work}) are substituted; relative paths are under
        the repo."""
        text = str(value).format(data=self.data, repo=REPO, **self.paths)
        return self._abs(text, REPO)

    def load_env(self):
        """Credentials from .env in the repository (.env.example lists them) into the environment. Values are never
        logged."""
        from dotenv import load_dotenv
        load_dotenv(REPO / ".env")


def _merge(base, over):
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v


# ---------------------------------------------------------------- running a step

def run(name, body, parser=None):
    """Entry point wrapper: parse flags, log to the console and to <work>/logs, stop cleanly on Ctrl+C."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = parser.parse_args() if parser else None
    cfg = Config()
    logs = cfg.work / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / f"{datetime.now():%Y%m%d-%H%M%S}-{name}.log"
    log.setLevel(logging.DEBUG)
    log.handlers.clear()
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    ch = RichHandler(console=console, show_time=False, show_path=False, markup=False, rich_tracebacks=False)
    ch.setLevel(logging.INFO)
    log.addHandler(fh)
    log.addHandler(ch)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _interrupt)
    log.debug("command: %s", " ".join(sys.argv))
    t0 = time.perf_counter()
    try:
        body(cfg, args)
    except KeyboardInterrupt:
        log.warning("Stopped. Finished work is saved; run `uv run %s` again to continue.", name)
        log.info("log: %s", log_path)
        sys.exit(130)
    except SystemExit as e:
        if e.code not in (None, 0):
            log.error("%s", e.code)
            log.info("log: %s", log_path)
            sys.exit(1)
        raise
    except Exception:
        log.exception("%s failed", name)
        log.info("log: %s", log_path)
        sys.exit(1)
    log.info("%s finished in %s; log: %s", name, _duration(time.perf_counter() - t0), log_path)


def _interrupt(*_):
    raise KeyboardInterrupt


def _duration(s):
    return f"{s:.1f} s" if s < 90 else f"{s / 60:.1f} min"


class RateColumn(ProgressColumn):
    def __init__(self, unit):
        super().__init__()
        self.unit = unit

    def render(self, task):
        speed = task.finished_speed or task.speed
        return Text("" if speed is None else f"{speed:,.1f} {self.unit}/s", style="progress.data.speed")


@contextmanager
def progress(description, total, unit="items", completed=0, info=None, **fields):
    """A progress bar with counts, rate and time left. Yields advance(n=1, **fields). `info` is an extra text column
    over the task's fields, e.g. "loss {task.fields[loss]}"; `fields` are their starting values."""
    columns = [TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn(), RateColumn(unit),
               TimeRemainingColumn()]
    if info:
        columns.append(TextColumn(info))
    with Progress(*columns, console=console) as p:
        task = p.add_task(description, total=total, completed=completed, **fields)
        yield lambda n=1, **values: p.update(task, advance=n, **values)


# ---------------------------------------------------------------- step state

def fingerprint(path):
    """Cheap change marker: size and mtime for a file; count, total size and newest mtime for a directory."""
    path = Path(path)
    if not path.exists():
        return None
    if path.is_file():
        st = path.stat()
        return [st.st_size, st.st_mtime_ns]
    files = [p.stat() for p in path.rglob("*") if p.is_file()]
    return [len(files), sum(s.st_size for s in files), max((s.st_mtime_ns for s in files), default=0)]


def source_hash(*names):
    """Hash pipeline source files so cached outputs expire when their builder changes."""
    h = hashlib.sha256()
    for name in sorted(names):
        h.update(name.encode("utf-8") + b"\0")
        h.update(Path(__file__).with_name(name + ".py").read_bytes())
    return h.hexdigest()


def _made_from_path(output):
    output = Path(output)
    return output.with_name(output.name + ".inputs.json")


def made_from(output):
    """What `output` was made from, as recorded by record_inputs, or None."""
    p = _made_from_path(output)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def is_current(output, inputs):
    """Whether `output` exists and was made from these inputs: a dict of fingerprints, settings and code constants. A
    stage that skips its work because its output exists must ask this instead."""
    return Path(output).exists() and not changed_keys(output, inputs)


def record_inputs(output, inputs):
    """Records what `output` is (being) made from, next to it."""
    write_text(_made_from_path(output), json.dumps(inputs, indent=1))


def changed_keys(output, inputs):
    """The keys of `inputs` that differ from what `output` was made from (all of them when nothing is recorded)."""
    was, now = made_from(output), json.loads(json.dumps(inputs))
    if was is None:
        return sorted(now)
    return sorted(k for k in set(was) | set(now) if k not in was or k not in now or was[k] != now[k])


# ---------------------------------------------------------------- crash-safe files

def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_parquet(table, path):
    import pyarrow.parquet as pq
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, path)
