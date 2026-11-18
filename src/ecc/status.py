"""uv run status: each step's state: done, partly done (with counts), stale (an input changed since) or not started.
Export is judged by the records export itself reads (export.slot_stale), so an export shown done makes no graph, torch
score or threshold again when run."""
from rich.table import Table

from .core import console


def export_row(cfg):
    """The v4 export slot of [export] c: the graphs, torch.parquet and thresholds.json each current with what it was
    made from, and the last parity run."""
    from .export import folders, slot_stale
    f = folders(cfg)
    if not (f["models"] / "graphs.json").exists():
        return "export", "[dim]not started[/]", ""
    why = slot_stale(cfg, f)
    if why:
        missing = why.startswith("no ")
        return "export", "[cyan]partly done[/]" if missing else "[yellow]stale[/]", why
    if not (f["out"] / "parity.json").exists():
        return "export", "[cyan]partly done[/]", f"graphs, torch scores and thresholds in {f['out']}; no parity run yet"
    return "export", "[green]done[/]", f"graphs in {f['models']}; parity in {f['out'] / 'parity.json'}"


def bundle_row(cfg):
    built = sorted((p.stat().st_mtime, p.parent) for p in (cfg.work / "bundles").glob("v4-*/manifest.json")
                   if not p.parent.name.endswith(".tmp"))
    if not built:
        return "bundle", "[dim]not started[/]", ""
    out = built[-1][1]
    checked = (out.with_name(out.name + "-vectors") / "check.md").exists()
    return "bundle", "[green]done[/]" if checked else "[cyan]partly done[/]", (
        f"last built {out.name}" + ("" if checked else "; its vectors were not checked"))


def batch_rows(cfg):
    """One row per reader batch: its parts, each read's answers, and whether its truth table is newer than every
    answer of the reads it is built from."""
    from .dossier import load_manifest, part_files
    from .truth import table_path
    root = cfg.path(cfg["dossier"]["batches"])
    out = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []:
        if load_manifest(folder) is None:
            continue
        stems = {p.stem for p in part_files(folder)}
        reads, newest = [], 0
        for r in sorted(p for p in (folder / "reads").iterdir() if p.is_dir()) if (folder / "reads").exists() else []:
            answers = [p for p in r.glob("*.json") if p.stem in stems]
            if r.name in (*cfg["truth"]["reads"], cfg["truth"]["third"]):
                newest = max([newest] + [p.stat().st_mtime for p in answers])
            reads.append(f"{r.name} {len(answers)}")
        truth = table_path(cfg, folder.name)
        detail = f"{len(stems)} parts; reads: {', '.join(reads) or 'none'}"
        if not truth.exists():
            state = "[cyan]partly done[/]" if reads else "[dim]not started[/]"
        elif truth.stat().st_mtime < newest:
            state, detail = "[yellow]stale[/]", detail + "; an answer is newer than truth.parquet"
        else:
            state, detail = "[green]done[/]", detail + "; truth.parquet"
        out.append((f"batch {folder.name}", state, detail))
    return out


def main():
    from .core import Config
    cfg = Config()
    data = cfg.data
    rows = batch_rows(cfg)
    rows.append(export_row(cfg))
    rows.append(bundle_row(cfg))

    table = Table(title=f"Module-ECC pipeline ({data})", title_justify="left")
    for col in ("step", "state", "detail"):
        table.add_column(col)
    for r in rows:
        table.add_row(*r)
    console.print(table)
