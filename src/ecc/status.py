"""uv run status: each step's state: done, partly done (with counts), stale (an input changed since) or not started.
Train, export and eval are judged by the same records and rules those steps use, so a step shown done does nothing
when run again."""
from rich.table import Table

from .core import changed_inputs, console, load_state


def row(cfg, step, state, detail, partial=False):
    if state is None:
        return step, "[dim]not started[/]", detail
    stale = changed_inputs(cfg, state)
    if stale:
        return step, "[yellow]stale[/]", f"{detail}; changed since: {', '.join(stale)}"
    return step, "[cyan]partly done[/]" if partial else "[green]done[/]", detail


def train_row(cfg, name, out):
    from .train import checkpoint_state, checkpoints, run_state
    step = f"train {name}"
    _, _, _, done, ckpt_changed = run_state(cfg, name, cfg["train"]["seed"], None, out)
    if done:
        return step, "[green]done[/]", str(out / "adapter")
    if ckpt_changed:
        return step, "[red]blocked[/]", (f"the checkpoints were trained with other settings ({', '.join(ckpt_changed)}); "
                                         f"delete {out / 'ckpt'} to train from the start")
    found = checkpoints(out)
    if found:
        s = checkpoint_state(found[-1])
        return step, "[cyan]partly done[/]", f"step {s['step']:,}/{s['total']:,}; `uv run train` continues"
    if (out / "train_done.json").exists():
        return step, "[yellow]stale[/]", "trained from other inputs; `uv run train` trains it again"
    return step, "[dim]not started[/]", ""


def export_row(cfg, name, out):
    from .export import stages_done
    step = f"export {name}"
    if not (out / "train_done.json").exists():
        return step, "[dim]not started[/]", ""
    done = stages_done(cfg, out)
    finished = [k for k, ok in done.items() if ok]
    if len(finished) == len(done):
        return step, "[green]done[/]", f"[models.classifier] in {cfg.data / 'classifier.toml'}"
    left = ", ".join(k for k, ok in done.items() if not ok)
    if finished:
        return step, "[cyan]partly done[/]", f"done: {', '.join(finished)}; left: {left}"
    return step, "[dim]not started[/]", f"left: {left}"


def eval_row(cfg, name):
    from .evaluate import model_sets, set_scores
    spec = cfg["models"][name]
    step = f"eval {name}"
    if "gguf" not in spec:
        return step, "[dim]not started[/]", "no gguf; `uv run export` writes it"
    sets = [s for s in model_sets(cfg, spec)
            if s not in ("train", "outside") or (cfg.work / "eval" / name / f"{s}.jsonl").exists()]
    parts, current, complete = [], 0, 0
    for set_name in sets:
        scores = set_scores(cfg, name, set_name)
        if scores is None:
            stale = (cfg.work / "eval" / name / f"{set_name}.jsonl").exists()
            parts.append(f"{set_name} {'stale' if stale else '-'}")
            continue
        rows, missing = scores
        parts.append(f"{set_name} {len(rows):,}/{len(rows) + missing:,}")
        current += 1
        complete += not missing
    state = ("[green]done[/]" if complete == len(sets) else "[cyan]partly done[/]" if current
             else "[yellow]stale[/]" if any(p.endswith(" stale") for p in parts) else "[dim]not started[/]")
    return step, state, ", ".join(parts)


def main():
    from .core import Config
    from .ingest import pending
    from .label import scan
    cfg = Config()
    data = cfg.data
    rows = []

    s = load_state(cfg, "ingest")
    try:
        new = {k: len(v) for k, v in pending(cfg, s).items() if v}
    except (OSError, SystemExit) as e:
        new = {"?": str(e)}
    detail = f"{s['rows']:,} rows in {len(s['batches'])} batches" if s else "no messages.parquet"
    if new:
        detail += "; new files: " + ", ".join(f"{k} {v}" for k, v in new.items())
    rows.append(row(cfg, "ingest", s, detail, partial=bool(new) and s is not None))

    s = load_state(cfg, "candidates")
    rows.append(row(cfg, "candidates", s, f"{s['windows']:,} windows, {s['shards']} shards" if s else ""))

    s = load_state(cfg, "label")
    if (data / "label" / "shards.json").exists():
        sets, _, _ = scan(cfg) if s else ({}, None, None)
        if s:
            parts = [f"{n} {len(v)}/{len(i)}" for n, (i, v, _) in sets.items()]
            left = sum(len(bad) for _, _, bad in sets.values())
            rows.append(row(cfg, "label", s, ", ".join(parts) + (f"; {left} to label" if left else ""),
                            partial=left > 0))
        else:
            rows.append(row(cfg, "label", None, "run `uv run label`"))
    else:
        rows.append(row(cfg, "label", None, "no shards"))

    s = load_state(cfg, "merge")
    rows.append(row(cfg, "merge", s, (f"{s['rows']:,} rows: " + ", ".join(f"{k} {v:,}" for k, v in s["counts"].items())
                                      + f"; corrections {s['corrections']}") if s else ""))
    s = load_state(cfg, "trainset")
    rows.append(row(cfg, "trainset", s, ", ".join(f"{k} {n:,} ({p:,} ban)" for k, (n, p) in s["counts"].items())
                    + f"; history {s['history']}" if s else ""))

    for name in cfg["train"]["models"]:
        out = cfg.work / "models" / name
        if name != cfg["train"]["model"] and not out.exists():
            continue
        rows += [train_row(cfg, name, out), export_row(cfg, name, out)]

    for name in cfg["models"]:
        rows.append(eval_row(cfg, name))

    s = load_state(cfg, "review")
    queue = cfg.work / "review" / "queue.txt"
    rows.append(row(cfg, "review", s, f"{s['model']}: {s['queued']} queued, {s['corrections']} corrections"
                    if s else ("" if not queue.exists() else str(queue))))

    s = load_state(cfg, "redteam")
    if s:
        detail = "; ".join(f"{t} {r['decisions']} decisions, {r['failures']} failures" for t, r in s["runs"].items())
        rows.append(row(cfg, "redteam", s, detail))
    else:
        rows.append(row(cfg, "redteam", None, ""))

    table = Table(title=f"Module-ECC pipeline ({data})", title_justify="left")
    for col in ("step", "state", "detail"):
        table.add_column(col)
    for r in rows:
        table.add_row(*r)
    console.print(table)
