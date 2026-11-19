"""uv run trainset: the inputs C trains on, built by ecc.inputs, the code export, the bundle and its vectors
build the served inputs with, so training sees what ECC-Server feeds the models.

The Paper table is every row of the [train] owner batches together in arrival order: by run, each run's log files in
log order (the stem's date and number), each file's rows in stored order, so a run whose log files went to two batches
is one stream again. The step stops on a batch whose stored order is not its runs' arrival order (time running
backwards, or a log file out of order or in two blocks), on a merged run whose time runs backwards
(ecc.inputs.contexts), and on a log file name in two batches or a run over two logs folders. A kinds-only batch
([train] kinds_only, Creayun) is its own table, in its truth order (each dossier's parts by part number). The text is
what the models read live: meaning normalization, redaction of revealed addresses, then each link the module's
default allowlist allows replaced by links.canonical (inputs.model_text, export.links).

Writes <[train] out>/inputs/:
- paper.parquet: the Paper table's rows in order (batch, dossier, id, run, coverage); row k is line k of every context.
- lines.parquet: the lines the model trains and is scored on, in training row order: the judged Paper rows in Paper
  order, then every row of the kinds-only batches. batch, dossier, id, run, paper_row (-1 for a kinds-only line),
  split, coverage (c or reject), the seven kinds, unsure_masked, left_out, kmask (out of the kind loss:
  unsure_masked, left_out, -1 on every kind or rejected coverage). Rejected rows retain their keys but train no loss.
- c-<N>.npy, c-<N>.offsets.npy: C's input ids cut to N tokens for every line of lines.parquet (row k:
  ids[offsets[k]:offsets[k + 1]]), for each N of [train.c] tokens and the length the ranker ([train] ranker) was
  trained at.
- inputs.json: what the inputs were made from, the lengths and the counts. The step does nothing when it is current.

With --rank, the ranker (the trained C of [train] ranker, at the length it was trained at, bf16) scores the candidates,
the unmasked train lines with every kind at 0 outside the dated batch ([periods] owner): <[train] out>/rank.parquet
(row of lines.parquet, batch, dossier, id, z_<out> logits, score = the larger of the hold logit and the largest kind
logit). Then, once every Paper batch has its hold.parquet (uv run labels), the training lines,
<[train] out>/pick.parquet (row, batch, dossier, id, reason, hold, score), in training row order:
- the dated batch: every line of its train weeks ("all");
- the other Paper batches: every hold = 1 line ("hold"), every unmasked line with a kind at 1 or -1 ("non-zero"), the
  top [train] share of each batch's candidates by score ("hard") and a random share of its other candidates
  ("random");
- each kinds-only batch: every unmasked line with a non-zero kind, the top [train] kinds_share of its candidates and a
  random kinds_share of the rest, every one with hold -1 (out of the hold loss);
- a line out of both losses (kind-masked, hold -1) is dropped.
The random draws are numpy's default_rng([train] seed) over each batch's other candidates in row order: one stream
for the Paper batches in [train] owner order, one of its own for each kinds-only batch. A line outside the train split
in any of them stops the step.
"""
import argparse
import json
import os
import time

import numpy as np

from . import inputs as I
from .core import fingerprint, is_current, log, record_inputs, run, source_hash, write_text

CHUNK = 50000           # lines rendered and tokenised at a time
COLS = ["dossier", "run", "id", "time", "channel", "partner", "speaker", "account", "text", "judged", "split",
        "unsure_masked", "left_out"] + I.KINDS
RANK_TOKENS = 65536     # tokens in one ranking batch
RANK_CHUNK = 100000     # lines between progress lines
RANK_RULE = ("candidates: nonrejected, split train, kmask false, every kind 0, batch not [periods] owner; score max(z_hold, "
             "max z_kind); bf16 autocast, fp32 weights")


def folder(cfg):
    return cfg.path(cfg["train"]["out"]) / "inputs"


def rank_path(cfg):
    return cfg.path(cfg["train"]["out"]) / "rank.parquet"


def pick_path(cfg):
    return cfg.path(cfg["train"]["out"]) / "pick.parquet"


def hold_path(cfg, batch):
    return cfg.path(cfg["dossier"]["batches"]) / batch / "hold.parquet"


def lengths(cfg):
    """C's lengths: [train.c] tokens and the ranker's own (input.json trained_tokens), longest first."""
    from .base import ranker
    rank = json.loads((ranker(cfg) / "input.json").read_text(encoding="utf-8"))["trained_tokens"]
    return sorted({int(n) for n in cfg["train"]["c"]["tokens"]} | {int(rank)}, reverse=True)


def table_path(cfg, batch):
    from .truth import table_path as selected_table
    return selected_table(cfg, batch)


def load(cfg, batch, allow, canon):
    """A batch's truth table with the text the model reads ('' for a line with no text)."""
    import pandas as pd
    import pyarrow.parquet as pq
    path = table_path(cfg, batch)
    names = pq.ParquetFile(path).schema_arrow.names
    raw_cols = ["raw_text"] if "raw_text" in names else ["row"]
    df = with_raw(cfg, pd.read_parquet(path, columns=COLS + raw_cols))
    df["text"] = [I.model_text(t, allow, canon) if isinstance(t, str) else "" for t in df.text]
    df.insert(0, "batch", batch)
    return df


def with_raw(cfg, df):
    """Preserve Paper raw text, or recover Creayun text through its stable source row. Never use shortened text as raw."""
    import pandas as pd
    if "raw_text" in df:
        if not df.raw_text.map(lambda value: isinstance(value, str)).all():
            raise SystemExit("a line has no source raw text for the coverage check")
        return df
    if "row" not in df:
        raise SystemExit("a line table has neither raw_text nor a Creayun source row")
    raw = pd.read_parquet(cfg.path(cfg["dossier"]["messages"]), columns=["row", "text"])
    if raw.row.duplicated().any():
        raise SystemExit("the Creayun source table has duplicate row keys")
    if not (df.id.to_numpy() == ("L" + df.row.astype(str)).to_numpy()).all():
        raise SystemExit("Creayun line ids do not match their source row keys")
    pos = pd.Index(raw.row).get_indexer(df.row)
    if (pos < 0).any():
        raise SystemExit("the Creayun source table does not cover every line")
    out = df.copy()
    out["raw_text"] = raw.text.to_numpy()[pos]
    if not out.raw_text.map(lambda value: isinstance(value, str)).all():
        raise SystemExit("a Creayun line has no source raw text")
    return out


def file_rank(dossiers):
    """Each Paper log file's place in log order, from its stem: (date, number) of YYYY-MM-DD-N, <day>-latest after
    every dated file of its day."""
    import pandas as pd
    s = pd.Series(dossiers)
    m = s.str.extract(r"^(\d{4}-\d\d-\d\d)-(\d+|latest)$")
    if m[0].isna().any():
        raise SystemExit("not a Paper log file's stem: " + ", ".join(sorted(set(s[m[0].isna()]))[:10]))
    num = pd.to_numeric(m[1].where(m[1] != "latest", str(2**62))).astype(np.int64)
    return pd.DataFrame({"day": m[0], "num": num}).groupby(["day", "num"]).ngroup().to_numpy()


def check_arrival(df, batch):
    """Refuses a batch table whose stored order is not each run's arrival order: within a run, time never runs
    backwards and its log files come one block each, in log order."""
    sec, dos, rank = df.time.to_numpy(), df.dossier.to_numpy(), file_rank(df.dossier.to_numpy())
    bad = []
    for run, idx in df.groupby("run", sort=False).indices.items():
        back = int((np.diff(sec[idx]) < 0).sum())
        starts = np.flatnonzero(np.r_[True, dos[idx][1:] != dos[idx][:-1]])
        blocks = (np.diff(rank[idx][starts]) <= 0).sum()
        if back or blocks:
            bad.append(f"{run} (time backwards {back} times, {int(blocks)} log files out of order or in two blocks)")
    if bad:
        raise SystemExit(f"{batch}: {len(bad)} runs are not stored in arrival order: " + "; ".join(bad[:10]))


def log_folders(cfg, batches):
    """{dossier: the logs folder of its log file} from the batches' manifests. A stem in two of them (two log files of
    one name) stops the step."""
    out, twice = {}, set()
    for b in batches:
        m = json.loads((cfg.path(cfg["dossier"]["batches"]) / b / "manifest.json").read_text(encoding="utf-8"))
        for e in m["dossiers"]:
            if e["dossier"] is None:
                continue
            if e["dossier"] in out:
                twice.add(e["dossier"])
            out[e["dossier"]] = e["source"].split("/")[0]
    if twice:
        raise SystemExit(f"{len(twice)} log file names are in more than one [train] owner batch or logs folder: "
                         + ", ".join(sorted(twice)[:20]))
    return out


def paper_table(cfg, allow, canon):
    """Every row of the [train] owner batches in arrival order: by run, each run's log files in log order, each file's
    rows in stored order. Each batch's stored order is checked first (check_arrival); ecc.inputs.contexts checks the
    merged runs' time. A run whose log files are in two logs folders (two files of one stem) stops the step."""
    import pandas as pd
    owner = cfg["train"]["owner"]
    parts = []
    for b in owner:
        d = load(cfg, b, allow, canon)
        check_arrival(d, b)
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    if df.duplicated(["dossier", "id"]).any():
        raise SystemExit("a (dossier, id) is in more than one [train] owner batch")
    where = log_folders(cfg, owner)
    unknown = sorted(set(df.dossier) - set(where))
    if unknown:
        raise SystemExit(f"{len(unknown)} log files of the tables are in no manifest: " + ", ".join(unknown[:10]))
    folders = df.dossier.map(where).groupby(df.run).unique()
    mixed = folders[folders.map(len) > 1]
    if len(mixed):
        raise SystemExit(f"{len(mixed)} runs join log files of two logs folders (two files of one stem): "
                         + "; ".join(f"{r} ({', '.join(f)})" for r, f in mixed.items()))
    order = np.lexsort((np.arange(len(df)), file_rank(df.dossier.to_numpy()), pd.factorize(df.run, sort=True)[0]))
    return df.iloc[order].reset_index(drop=True)


def kmask(df):
    """Lines out of the kind loss: unsure lines no read marked, spam raids, -1 on every kind, or rejected coverage."""
    return (df.unsure_masked.to_numpy() | df.left_out.to_numpy() | (df[I.KINDS].to_numpy() == -1).all(1)
            | (df.coverage.to_numpy() == "reject"))


def meta(df, rows, paper):
    m = df.iloc[rows][["batch", "dossier", "id", "run", "split", "coverage"] + I.KINDS +
                      ["unsure_masked", "left_out"]].copy()
    m.insert(4, "paper_row", rows if paper else -1)
    m["kmask"] = kmask(df)[rows]
    return m.reset_index(drop=True)


class Ragged:
    """Arrays of int32 ids appended in order, written as one flat .npy and its offsets."""

    def __init__(self, path):
        self.path = path
        self.tmp = path.with_name(path.name + ".tmp")
        self.f = open(self.tmp, "wb")
        self.offsets = [0]

    def add(self, arrays):
        if arrays:
            self.f.write(np.concatenate(arrays).astype(np.int32, copy=False).tobytes())
        for a in arrays:
            self.offsets.append(self.offsets[-1] + len(a))

    def close(self):
        self.f.close()
        flat = np.memmap(self.tmp, np.int32, "r", shape=(self.offsets[-1],)) if self.offsets[-1] else \
            np.zeros(0, np.int32)
        out = np.lib.format.open_memmap(self.path.with_name(self.path.name + ".part"), "w+", np.int32, flat.shape)
        for i in range(0, len(flat), 1 << 26):
            out[i:i + (1 << 26)] = flat[i:i + (1 << 26)]
        out.flush()
        del out, flat
        os.replace(self.path.with_name(self.path.name + ".part"), self.path)
        np.save(offsets_path(self.path), np.asarray(self.offsets, np.int64))
        os.remove(self.tmp)


def offsets_path(path):
    return path.with_name(path.name.removesuffix(".npy") + ".offsets.npy")


def ragged(path):
    """(flat ids, offsets) of a Ragged file, the ids memory-mapped: row k is ids[offsets[k]:offsets[k + 1]]."""
    return np.load(path, mmap_mode="r"), np.load(offsets_path(path))


def write_c(writers, df, ctx, rows, builder, what):
    t0 = time.perf_counter()
    for i in range(0, len(rows), CHUNK):
        rendered = I.render(df, ctx, rows[i:i + CHUNK])
        for n, w in writers.items():
            w.add(builder.build(rendered, n))
        log.debug("%s: C ids of %d of %d lines (%.0f s)", what, min(len(rows), i + CHUNK), len(rows),
                  time.perf_counter() - t0)
    log.info("%s: C ids of %s lines at %s tokens (%.0f s)", what, f"{len(rows):,}", ", ".join(map(str, writers)),
             time.perf_counter() - t0)


def build(cfg, out, made_from):
    import pandas as pd
    import pyarrow as pa

    from .base import path as base_path
    from .core import write_parquet
    from .export import links
    from .coverage import contexts
    t0 = time.perf_counter()
    allow, canon = links(cfg)
    ns = made_from["c_tokens"]
    out.mkdir(parents=True, exist_ok=True)
    for p in (out / "inputs.json", out / "inputs.json.inputs.json"):     # not current until the build ends
        p.unlink(missing_ok=True)
    builder = I.CInputs(base_path(cfg, "c") / "tokenizer.json")
    writers = {n: Ragged(out / f"c-{n}.npy") for n in ns}

    # the Paper table: C's ids of every judged row
    df = paper_table(cfg, allow, canon)
    ctx, routes = contexts(df, builder.tok, made_from["coverage_tokens"])
    df["coverage"] = routes
    log.info("Paper table: %s rows of %s, %s runs (%.0f s)", f"{len(df):,}", ", ".join(cfg["train"]["owner"]),
             f"{df.run.nunique():,}", time.perf_counter() - t0)
    write_parquet(pa.Table.from_pandas(df[["batch", "dossier", "id", "run", "coverage"]], preserve_index=False),
                  out / "paper.parquet")
    judged = np.flatnonzero(df.judged.to_numpy())
    write_c(writers, df, ctx, judged, builder, "Paper")
    metas = [meta(df, judged, True)]
    paper_rows = len(df)
    del df, ctx

    # the kinds-only batches: C's ids of every row
    for b in cfg["train"]["kinds_only"]:
        d = load(cfg, b, allow, canon)
        rows = np.arange(len(d))
        cctx, routes = contexts(d, builder.tok, made_from["coverage_tokens"])
        d["coverage"] = routes
        write_c(writers, d, cctx, rows, builder, b)
        metas.append(meta(d, rows, False))
        del d
    for w in writers.values():
        w.close()
    m = pd.concat(metas, ignore_index=True)
    write_parquet(pa.Table.from_pandas(m, preserve_index=False), out / "lines.parquet")
    counts = {"paper_rows": paper_rows, "coverage": m.coverage.value_counts().to_dict(),
              "lines": {f"{b}/{s}": int(v) for (b, s), v in m.groupby(["batch", "split"]).size().items()}}
    for n in ns:
        ids, off = ragged(out / f"c-{n}.npy")
        ln = np.diff(off)
        counts[f"c-{n}"] = {"median": int(np.median(ln)), "at_cut": round(float(np.mean(ln >= n)), 4)}
    write_text(out / "inputs.json", json.dumps({**made_from, "counts": counts}, indent=1) + "\n")
    record_inputs(out / "inputs.json", made_from)
    log.info("inputs of %s lines in %s (%.1f min): %s", f"{len(m):,}", out, (time.perf_counter() - t0) / 60,
             json.dumps(counts))


def candidates(m, dated):
    """The lines the ranker scores, as a mask over lines.parquet: unmasked train lines with every kind at 0, outside
    the dated batch (every line of whose train weeks trains)."""
    return ((m.split.to_numpy() == "train") & ~m.kmask.to_numpy() & (m[I.KINDS].to_numpy() == 0).all(1)
            & (m.batch.to_numpy() != dated) & (m.coverage.to_numpy() != "reject"))


def rank_key(z):
    """The ranking score of logits [rows, OUTS]: the larger of the hold logit and the largest kind logit."""
    return np.maximum(z[:, I.OUTS.index("hold")], z[:, [I.OUTS.index(k) for k in I.KINDS]].max(1))


def ranker_logits(c, seqs, device):
    """The ranker's logits for each id array: fp32 weights under bf16 autocast, length-sorted batches."""
    import torch

    from .models import score_c
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return score_c(c, seqs, device, tokens_per_batch=RANK_TOKENS)


def trained_tokens(cfg):
    from .base import ranker
    return int(json.loads((ranker(cfg) / "input.json").read_text(encoding="utf-8"))["trained_tokens"])


def made_rank(cfg):
    from .base import ranker
    return {"inputs": fingerprint(folder(cfg) / "inputs.json"), "ranker": fingerprint(ranker(cfg)),
            "tokens": trained_tokens(cfg), "dated": cfg["periods"]["owner"], "rule": RANK_RULE,
            "code": source_hash("core", "inputs", "models", "trainset", "base")}


def rank(cfg, mf):
    """Scores the candidates with the ranker into rank.parquet (GPU)."""
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")   # batches of every shape, no fragments
    import pandas as pd
    import pyarrow as pa
    import torch

    from .base import ranker
    from .core import write_parquet
    from .models import limit_gpu_memory, load_c
    inp = folder(cfg)
    m = pd.read_parquet(inp / "lines.parquet", columns=["batch", "dossier", "id", "split", "kmask", "coverage"] + I.KINDS)
    rows = np.flatnonzero(candidates(m, mf["dated"]))
    ids, off = ragged(inp / f"c-{mf['tokens']}.npy")
    if len(off) != len(m) + 1:
        raise SystemExit(f"c-{mf['tokens']}.npy has {len(off) - 1:,} rows, lines.parquet {len(m):,}")
    order = np.argsort(np.diff(off)[rows], kind="stable")
    limit_gpu_memory(log)
    dev = torch.device("cuda")
    c = load_c(ranker(cfg), dev)
    log.info("ranking %s candidates (%s) at %d tokens", f"{len(rows):,}",
             ", ".join(f"{b} {n:,}" for b, n in m.batch.iloc[rows].value_counts(sort=False).items()), mf["tokens"])
    z = np.zeros((len(rows), len(I.OUTS)), np.float32)
    t0 = time.perf_counter()
    for i in range(0, len(order), RANK_CHUNK):
        idx = order[i:i + RANK_CHUNK]
        z[idx] = ranker_logits(c, [ids[off[r]:off[r + 1]] for r in rows[idx]], dev)
        done = min(len(order), i + RANK_CHUNK)
        s = time.perf_counter() - t0
        log.info("ranked %s of %s (%.0f s, %.0f lines a second; GPU peak %.1f GB allocated, %.1f GB reserved)",
                 f"{done:,}", f"{len(rows):,}", s, done / s, torch.cuda.max_memory_allocated() / 2**30,
                 torch.cuda.max_memory_reserved() / 2**30)
    if not np.isfinite(z).all():
        raise SystemExit("the ranker gave a logit that is not finite")
    out = m.iloc[rows][["batch", "dossier", "id"]].reset_index(drop=True)
    out.insert(0, "row", rows.astype(np.int64))
    for j, o in enumerate(I.OUTS):
        out[f"z_{o}"] = z[:, j]
    out["score"] = rank_key(z)
    write_parquet(pa.Table.from_pandas(out, preserve_index=False), rank_path(cfg))
    record_inputs(rank_path(cfg), mf)
    log.info("rank.parquet: %s lines in %.1f min", f"{len(out):,}", (time.perf_counter() - t0) / 60)


def hold_labels(cfg, m):
    """The hold label of each line of lines.parquet: its batch's hold.parquet for a Paper line, -1 (out of the hold
    loss) for a kinds-only or rejected line. Refuses a judged Paper line with no label, or a label of a line that is
    not one. Coverage changes the training mask without rewriting the source labels."""
    import pandas as pd
    hold = np.full(len(m), -1, np.int8)
    batch = m.batch.to_numpy()
    for b in cfg["train"]["owner"]:
        h = pd.read_parquet(hold_path(cfg, b), columns=["dossier", "id", "hold"])
        rows = np.flatnonzero(batch == b)
        have = pd.MultiIndex.from_arrays([h.dossier, h.id])
        if have.duplicated().any():
            raise SystemExit(f"{hold_path(cfg, b)} holds a line twice")
        pos = have.get_indexer(pd.MultiIndex.from_arrays([m.dossier.to_numpy()[rows], m.id.to_numpy()[rows]]))
        if (pos < 0).any() or len(h) != len(rows):
            raise SystemExit(f"{hold_path(cfg, b)}: {int((pos < 0).sum()):,} judged lines have no label, "
                             f"{len(h) - int((pos >= 0).sum()):,} labels name no judged line; run `uv run labels`")
        hold[rows] = h.hold.to_numpy()[pos]
    hold[m.coverage.to_numpy() == "reject"] = -1
    return hold


def select(m, hold, score, dated, paper, kinds_only, share, kinds_share, seed):
    """Why each line of lines.parquet trains ('' where it does not), by the rule of the module docstring. `score` is
    the ranking score of each candidate (NaN elsewhere)."""
    y, km = m[I.KINDS].to_numpy(), m.kmask.to_numpy()
    batch, train = m.batch.to_numpy(), m.split.to_numpy() == "train"
    nonzero = train & ~km & (y != 0).any(1)
    cand = candidates(m, dated)
    if np.isnan(score[cand]).any():
        raise SystemExit(f"{int(np.isnan(score[cand]).sum()):,} candidates have no ranking score")
    reason = np.full(len(m), "", object)
    reason[(batch == dated) & train] = "all"
    reason[(batch != dated) & nonzero] = "non-zero"
    reason[(batch != dated) & train & (hold == 1)] = "hold"

    def hard_and_random(b, part, rng):
        idx = np.flatnonzero(cand & (batch == b))
        nh = round(part * len(idx))
        hard = idx[np.argsort(-score[idx], kind="stable")[:nh]]
        rest = np.setdiff1d(idx, hard)
        rest = rest[rng.random(len(rest)) < part]
        reason[hard[reason[hard] == ""]] = "hard"
        reason[rest[reason[rest] == ""]] = "random"
        log.info("%s: %s candidates, %s hard (score >= %s), %s random", b, f"{len(idx):,}", f"{nh:,}",
                 f"{score[hard].min():.3f}" if nh else "-", f"{len(rest):,}")

    rng = np.random.default_rng(seed)
    for b in paper:
        if b != dated:
            hard_and_random(b, share, rng)
    for b in kinds_only:
        hard_and_random(b, kinds_share, np.random.default_rng(seed))
    reason[(reason != "") & km & (hold == -1)] = ""
    reason[m.coverage.to_numpy() == "reject"] = ""
    if (reason[~train] != "").any():
        raise SystemExit(f"{int((reason[~train] != '').sum()):,} picked lines are not in the train split")
    return reason


def made_pick(cfg):
    t = cfg["train"]
    return {"inputs": fingerprint(folder(cfg) / "inputs.json"), "rank": fingerprint(rank_path(cfg)),
            "holds": {b: fingerprint(hold_path(cfg, b)) for b in t["owner"]}, "dated": cfg["periods"]["owner"],
            "share": t["share"], "kinds_share": t["kinds_share"], "seed": t["seed"]}


def pick(cfg, mf):
    """The training lines into pick.parquet."""
    import pandas as pd
    import pyarrow as pa

    from .core import write_parquet
    t = cfg["train"]
    m = pd.read_parquet(folder(cfg) / "lines.parquet",
                        columns=["batch", "dossier", "id", "split", "kmask", "coverage"] + I.KINDS)
    r = pd.read_parquet(rank_path(cfg), columns=["row", "score"])
    score = np.full(len(m), np.nan)
    score[r.row.to_numpy()] = r.score.to_numpy()
    hold = hold_labels(cfg, m)
    reason = select(m, hold, score, mf["dated"], t["owner"], t["kinds_only"], t["share"], t["kinds_share"], t["seed"])
    rows = np.flatnonzero(reason != "")
    out = m.iloc[rows][["batch", "dossier", "id"]].reset_index(drop=True)
    out.insert(0, "row", rows.astype(np.int64))
    out["reason"] = reason[rows]
    out["hold"] = hold[rows]
    out["score"] = score[rows].astype(np.float32)
    write_parquet(pa.Table.from_pandas(out, preserve_index=False), pick_path(cfg))
    record_inputs(pick_path(cfg), mf)
    tab = out.groupby(["batch", "reason"], sort=False).hold.value_counts().unstack(fill_value=0)
    log.info("pick.parquet: %s training lines (hold = 1: %s)\n%s", f"{len(out):,}", f"{int((out.hold == 1).sum()):,}",
             tab.to_string())


def made(cfg):
    from .base import pinned
    from .export import links
    batches = list(cfg["train"]["owner"]) + list(cfg["train"]["kinds_only"])
    return {"owner": list(cfg["train"]["owner"]), "kinds_only": list(cfg["train"]["kinds_only"]),
            "tables": {b: fingerprint(table_path(cfg, b)) for b in batches},
            "manifests": {b: fingerprint(cfg.path(cfg["dossier"]["batches"]) / b / "manifest.json")
                          for b in cfg["train"]["owner"]},
            "allowlist": fingerprint(cfg.path(cfg["truth"]["allowlist"])), "canonical": links(cfg)[1],
            "raw_messages": fingerprint(cfg.path(cfg["dossier"]["messages"])),
            "c_base": pinned(cfg, "c")[:2], "c_tokens": lengths(cfg),
            "context": [I.CTX_LINES, I.CTX_SECONDS],
            "coverage_tokens": int(cfg["train"]["c"]["tokens"][0]),
            "code": source_hash("core", "inputs", "model_text", "trainset", "export", "truth", "base", "coverage")}


def body(cfg, args):
    out = folder(cfg)
    mf = made(cfg)
    if is_current(out / "inputs.json", mf) and not args.force:
        log.info("trainset: the inputs in %s are current", out)
    else:
        build(cfg, out, mf)
    mr = made_rank(cfg)
    if args.rank:
        if is_current(rank_path(cfg), mr) and not args.force:
            log.info("trainset: %s is current", rank_path(cfg))
        else:
            rank(cfg, mr)
    elif not is_current(rank_path(cfg), mr):
        raise SystemExit(f"{rank_path(cfg)} is missing or was made from other inputs or another ranker; run "
                         f"`uv run trainset --rank` (GPU)")
    missing = [b for b in cfg["train"]["owner"] if not hold_path(cfg, b).exists()]
    if missing:
        msg = f"the training lines wait on hold.parquet of {', '.join(missing)} (`uv run labels`)"
        if args.rank:
            log.info(msg)
            return
        raise SystemExit(msg)
    mp = made_pick(cfg)
    if is_current(pick_path(cfg), mp) and not args.force:
        log.info("trainset: %s is current", pick_path(cfg))
        return
    pick(cfg, mp)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rank", action="store_true", help="score the candidates with the ranker (GPU) first")
    ap.add_argument("--force", action="store_true", help="build and rank again even when current")
    run("trainset", body, ap)


if __name__ == "__main__":
    main()
