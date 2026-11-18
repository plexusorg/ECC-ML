"""uv run eval RUN [RUN ...]: the measures of one or more runs on the owner's validation and test weeks and on
Creayun's held-out lines, with intervals over resamples of the log files; each run after the first beside its
difference from the first.

A run is a folder of scores, named under [export] out (a system run) or [train] out (a C run), or given as a path (a
name found in both is refused):
- a system run, `export`'s folder of one C run: torch.parquet (dossier, id, period, c_<out> of every judged
  validation and test line) and thresholds.json (hold_recall, hold_1pct, hold_2pct), refused unless thresholds.json was
  set from that torch.parquet (the fingerprints both record). A line is held at a level when C's hold
  logit reaches the level's threshold, as ECC-Server decides. Its levels are diagnostic 1%, diagnostic 2%, and recall.
- a C run, a folder of `train c`: scores.parquet (dossier, id, period, c_<out> of every judged validation and test
  line) and, when it has them, creayun.parquet (dossier, id, c_<out> of Creayun's test_creayun lines). C alone holds a
  line when its hold logit reaches the score that holds each share of [eval] holds of the run's own validation lines.

The labels. A label set gives each judged line a hold label (1 hold, 0 not held, -1 unknown), its sitting and the
sitting's ruling and source. "forward" uses the frozen labels and ban timing from [eval] forward, with table names
mapped to batches by [eval] forward_batches. "new" reads hold.parquet in each [eval] batches folder and resolves ban
timing from the configured rulings and saved reads. It checks that the keys, hold labels and sitting fields still
match the saved labels. The split, kinds, masks and line order come from each batch's truth.parquet. Measure each
label set whose files exist, or name them with --labels. A scored line without a label counts only in the shares of
lines held and passed.

Measures per period (validation, test) and level, each a ratio of counts summed over the period's log files: recall of
the hold = 1 lines, of those in owner-ruled sittings and of those with each kind at 1; the share of lines held (and,
for a production run, scored by C); hold = 0 lines held per 1,000; the share of the marked hold = 0 lines held; active
clean players touched (clean sittings of 20 or more lines with 1 or more, and 5 or more, lines held); the owner's ban,
hold, nothing and not-ban sittings with a line held; and of his ban sittings with a ban line, those whose ban line is
held and those with a line held by it. Creayun's held-out lines (no hold label; every one a line its own filter
flagged), for runs that score them: the share of each kind's lines, of the lines with any kind and of the unmasked
all-zero lines over each hold level's threshold, and over each kind's own output at the score that flags [eval]
kind_share of the owner's validation lines. Beside them, per label set, the hold labels of the training period's lines
with each kind at 1.

Intervals: 95% over [eval] resamples resamples of the period's log files (Creayun: its dossiers, one server on one
day), drawn from [eval] seed, thresholds fixed. A difference between runs is paired on the same resamples; * marks one
whose interval excludes 0.

--validation reports the validation weeks only: every choice is made on them, and the test weeks are reported once,
after it. Writes <[eval] out>/<run>[+<run>...]/tables.md and eval.json. CPU only, under a minute.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .core import log, run, write_text
from .inputs import KINDS, OUTS
from .truth import table_path

PERIODS = ("val", "test")
RULES = {"ban": "ban", "hold": "hold", "nothing": "nothing", "not ban": "notban"}
CLEAN_LINES = 20        # a clean sitting of this many lines or more is an active clean player
SUPERVISION = ["sitting", "sitting_ruling", "sitting_source", "hold"]
LABELS = SUPERVISION + ["ban_here"]


# ---------------------------------------------------------------- runs

def thresholds(score, shares):
    """The score of the k-th highest line, k = ceil(share * lines): holding score >= it holds that share."""
    s = np.sort(np.asarray(score, np.float64))[::-1]
    return [float(s[max(1, int(np.ceil(h * len(s)))) - 1]) for h in shares]


def level(share):
    return f"{100 * share:g}%"


def check_scores(name, df, prefixes, period=True):
    cols = ["dossier", "id", "coverage"] + (["period"] if period else []) + [f"{p}_{o}" for p in prefixes for o in OUTS]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"{name}: the scores lack the columns {', '.join(missing)}")
    if df.duplicated(["dossier", "id"]).any():
        raise SystemExit(f"{name}: a line is scored twice")
    if period and not df.period.isin(PERIODS).all():
        raise SystemExit(f"{name}: a scored line's period is neither val nor test")
    if not df.coverage.isin(["cascade", "c", "reject"]).all():
        raise SystemExit(f"{name}: invalid coverage route")
    if not np.isfinite(df.loc[df.coverage != "reject", [f"{p}_{o}" for p in prefixes for o in OUTS]].to_numpy()).all():
        raise SystemExit(f"{name}: non-finite model score")


def system_run(name, scores, thr):
    """A production C-only run at validation diagnostic and recall thresholds."""
    check_scores(name, scores, ("c",))
    if thr.get("scoring", {}).get("mode") != "c_only" or not scores.coverage.isin(["c", "reject"]).all():
        raise SystemExit(f"{name}: production evaluation requires explicit C-only scores")
    return {"name": name, "kind": "c_only", "owner": scores.reset_index(drop=True), "creayun": None,
            "thresholds": {"1%": float(thr["hold_1pct"]), "2%": float(thr["hold_2pct"]),
                           "recall": float(thr["hold_recall"])}}


def c_run(name, scores, creayun, shares):
    """A run of C alone: held at a level when C's hold logit reaches the score that holds the level's share of the
    run's own validation lines."""
    check_scores(name, scores, ("c",))
    if creayun is not None:
        check_scores(name + " (Creayun)", creayun, ("c",), period=False)
    val = scores.c_hold.to_numpy()[(scores.period.to_numpy() == "val") & (scores.coverage.to_numpy() != "reject")]
    if not len(val):
        raise SystemExit(f"{name}: no validation line is scored")
    return {"name": name, "kind": "c", "owner": scores.reset_index(drop=True), "creayun": creayun,
            "thresholds": dict(zip(map(level, shares), thresholds(val, shares)))}


def held(r, df):
    """Return total blocks for each threshold. Report model holds and input rejects separately."""
    z = df.c_hold.to_numpy()
    routes = df.coverage.to_numpy()
    scored = routes != "reject"
    return {lv: (routes == "reject") | (scored & (z >= th)) for lv, th in r["thresholds"].items()}, None


def slot_thresholds(cfg, folder):
    """Use the shared production calibration guard for every evaluated export slot."""
    from .export import slot_thresholds as calibrated_thresholds
    thr = json.loads((folder / "thresholds.json").read_text(encoding="utf-8"))
    model = thr.get("models", {}).get("c", {}).get("folder")
    if not isinstance(model, str) or not model:
        raise SystemExit(f"{folder}: thresholds do not identify their C model")
    return calibrated_thresholds(cfg, {"c": Path(model), "out": folder, "models": folder / "models"})


def load_run(cfg, name, shares):
    import pandas as pd
    folder = Path(name)
    if not folder.is_dir():
        roots = (cfg.path(cfg["export"]["out"]), cfg.path(cfg["train"]["out"]))
        found = [root / name for root in roots if (root / name).is_dir()]
        if not found:
            raise SystemExit(f"{name}: no run of that name in {roots[0]} or {roots[1]}, and no such folder")
        if len(found) > 1:
            raise SystemExit(f"{name} is both {found[0]} (the system run) and {found[1]} (C alone); give the path")
        folder = found[0]
    if (folder / "torch.parquet").exists():
        if not (folder / "thresholds.json").exists():
            raise SystemExit(f"{folder}: torch.parquet without thresholds.json")
        thr = slot_thresholds(cfg, folder)
        r = system_run(folder.name, pd.read_parquet(folder / "torch.parquet"), thr)
        r["models"] = {"c": thr["models"]["c"]["folder"]}
        return r
    if (folder / "scores.parquet").exists():
        cre = folder / "creayun.parquet"
        r = c_run(folder.name, pd.read_parquet(folder / "scores.parquet"),
                  pd.read_parquet(cre) if cre.exists() else None, shares)
        r["models"] = {"c": folder.resolve().as_posix()}
        return r
    raise SystemExit(f"{folder}: neither torch.parquet (a system run) nor scores.parquet (a C run)")


# ---------------------------------------------------------------- labels

def truth_lines(path, batch):
    """The judged lines of a batch's truth table: dossier, id, batch, pos (the row in the table, the lines' order),
    split, the seven kinds and masked (left out of the kind loss: an unsure line no read marked, a spam raid's, or -1 on
    every kind)."""
    import pandas as pd
    df = pd.read_parquet(path, columns=["dossier", "id", "judged", "split", "unsure_masked", "left_out"] + KINDS)
    df["pos"] = np.arange(len(df))
    df["masked"] = df.unsure_masked.to_numpy() | df.left_out.to_numpy() | (df[KINDS].to_numpy() == -1).all(1)
    df = df[df.judged.to_numpy()].drop(columns=["judged", "unsure_masked", "left_out"])
    df.insert(2, "batch", batch)
    return df.reset_index(drop=True)


def label_set(name, labels, lines):
    """A label set: labels (dossier, id, batch, period, sitting, sitting_ruling, sitting_source, hold, ban_here) and
    lines (truth_lines of its batches, concatenated in batch order)."""
    for what, df in (("labels", labels), ("lines", lines)):
        if df.duplicated(["dossier", "id"]).any():
            raise SystemExit(f"labels {name}: a line is in its {what} twice")
    return {"name": name, "labels": labels.reset_index(drop=True), "lines": lines.reset_index(drop=True),
            "batches": list(dict.fromkeys(lines.batch))}


def forward_labels(cfg):
    import pandas as pd
    e = cfg["eval"]
    path = cfg.path(e["forward"])
    if not path.exists():
        return None
    names = e["forward_batches"]
    x = pd.read_parquet(path, columns=["dossier", "id", "table", "period"] + LABELS)
    unknown = sorted(set(x.table) - set(names))
    if unknown:
        raise SystemExit(f"{path}: the tables {', '.join(unknown)} are not in [eval] forward_batches")
    x["batch"] = x.table.map(names)
    lines = pd.concat([truth_lines(table_path(cfg, b), b) for b in dict.fromkeys(names.values())])
    return label_set("forward", x[["dossier", "id", "batch", "period"] + LABELS], lines)


def batch_labels(cfg):
    """Read the saved hold labels and resolve supported ban timing without changing those files."""
    import pandas as pd
    from .labels import build_label_cases, resolve_labels

    root = cfg.path(cfg["dossier"]["batches"])
    paths = {b: root / b / "hold.parquet" for b in cfg["eval"]["batches"]}
    have = [b for b, p in paths.items() if p.exists()]
    if not have:
        return None
    if len(have) < len(paths):
        raise SystemExit(f"hold.parquet is missing in {', '.join(sorted(set(paths) - set(have)))}")
    unknown = sorted(set(paths) - set(cfg["labels"]["batches"]))
    if unknown:
        raise SystemExit(f"[eval] batches outside [labels] batches: {', '.join(unknown)}")
    tables, cases, owner, _, where = build_label_cases(cfg)
    resolved, _, _ = resolve_labels(cfg, paths, tables, cases, owner, where)

    keys = ["dossier", "id"]
    frames = []
    for b, p in paths.items():
        saved = pd.read_parquet(p, columns=keys + SUPERVISION).set_index(keys)
        current = resolved[b].set_index(keys)
        if not saved.index.is_unique or not current.index.is_unique:
            raise SystemExit(f"{p}: saved and resolved label keys must be unique")
        if len(saved) != len(current) or not saved.index.isin(current.index).all():
            raise SystemExit(f"{p}: saved and resolved label keys differ")
        current = current.reindex(saved.index)
        for field in SUPERVISION:
            before, after = saved[field], current[field]
            equal = before.eq(after) | (before.isna() & after.isna())
            if not equal.fillna(False).all():
                raise SystemExit(f"{p}: saved and resolved {field} values differ")
        frames.append(saved.assign(batch=b, ban_here=current.ban_here).reset_index())
    lines = pd.concat([truth_lines(table_path(cfg, b), b) for b in paths])
    labels = pd.concat(frames)
    labels = labels.merge(lines[["dossier", "id", "split"]], on=["dossier", "id"], how="left", validate="1:1")
    if labels.split.isna().any():
        raise SystemExit(f"{int(labels.split.isna().sum())} lines of hold.parquet are no judged line of their batch")
    labels = labels.rename(columns={"split": "period"})
    return label_set("new", labels[["dossier", "id", "batch", "period"] + LABELS], lines)


def owner_lines(name, scores, ls, period):
    """Every judged line of the requested period, with its scores, kinds, order and labels. Refuses incomplete scores,
    including smoke runs, so every run uses the same lines and denominators."""
    expected = ls["lines"].loc[ls["lines"].split == period, ["dossier", "id"]]
    coverage = expected.merge(scores[["dossier", "id"]], on=["dossier", "id"], how="outer", validate="1:1",
                              indicator=True)
    missing = int((coverage._merge == "left_only").sum())
    extra = int((coverage._merge == "right_only").sum())
    if missing or extra:
        raise SystemExit(f"{name}: incomplete {period} scores for labels {ls['name']}: {missing:,} judged lines "
                         f"are not scored and {extra:,} scored lines are outside that period; score every judged "
                         "line before evaluation")
    u = scores[["dossier", "id", "period", "coverage"]].merge(ls["lines"].drop(columns="split"), on=["dossier", "id"], how="left",
                                                  validate="1:1")
    if u.pos.isna().any():
        raise SystemExit(f"{name}: {int(u.pos.isna().sum())} scored lines are no judged line of the batches of labels "
                         f"{ls['name']} ({', '.join(ls['batches'])})")
    u = u.merge(ls["labels"].drop(columns="batch").rename(columns={"period": "label_period"}), on=["dossier", "id"],
                how="left", validate="1:1")
    off = u.label_period.notna() & (u.label_period != u.period)
    if off.any():
        raise SystemExit(f"{name}: {int(off.sum())} scored lines are in another period than labels {ls['name']} give "
                         "them")
    return u.drop(columns="label_period")


# ---------------------------------------------------------------- measures

def owner_sittings(u):
    """The owner's ruled sittings among the lines: dossier, ruling, lines, banpos (the ban line's place, ban sittings)."""
    import pandas as pd
    o = u[(u.sitting_source == "owner") & u.sitting_ruling.isin(list(RULES))]
    g = o.groupby("sitting")
    return pd.DataFrame({"dossier": g.dossier.first(), "ruling": g.sitting_ruling.first(), "lines": g.size(),
                         "banpos": o[o.ban_here == 1].groupby("sitting").pos.min()})


def owner_measures(u, held, files, passed=None):
    """Numerator and denominator per log file of every measure, for the lines u held where `held`."""
    import pandas as pd
    f = pd.Categorical(u.dossier, categories=files).codes

    def lsum(x):
        return np.bincount(f, weights=np.asarray(x, float), minlength=len(files))

    firm = (u[KINDS].to_numpy() == 1).any(1)
    h = u.hold.to_numpy(float)
    own = u.sitting_source.to_numpy() == "owner"
    rejected = u.coverage.to_numpy() == "reject"
    model = held & ~rejected
    total = lsum(np.ones(len(u)))
    M = {"blocked_share": (lsum(held), total), "model_held_share": (lsum(model), total),
         "rejected_share": (lsum(rejected), total)}
    if passed is not None:
        M["passed_share"] = (lsum(passed), lsum(np.ones(len(u))))
    M |= {"recall_all": (lsum(held & (h == 1)), lsum(h == 1)),
          "recall_model_all": (lsum(model & (h == 1)), lsum(h == 1)),
          "recall_model_covered": (lsum(model & (h == 1)), lsum((h == 1) & ~rejected)),
          "recall_owner": (lsum(held & (h == 1) & own), lsum((h == 1) & own)),
          "per1000_h0": (1000 * lsum(held & (h == 0)), lsum(h == 0)),
          "let_through_held": (lsum(held & firm & (h == 0)), lsum(firm & (h == 0)))}
    for k in KINDS:
        pk = (h == 1) & (u[k].to_numpy() == 1)
        M[f"kind_{k}"] = (lsum(held & pk), lsum(pk))
        M[f"model_kind_{k}"] = (lsum(model & pk), lsum(pk & ~rejected))
    clean = u.sitting_ruling.to_numpy() == "clean"
    cl = u[clean].assign(held=held[clean])
    cl = cl.groupby("sitting").agg(n=("id", "size"), held=("held", "sum"), dossier=("dossier", "first"))
    cl = cl[cl.n >= CLEAN_LINES]
    fc = pd.Categorical(cl.dossier, categories=files).codes
    one = np.bincount(fc, minlength=len(files)).astype(float)
    M["clean20_any"] = (np.bincount(fc, weights=(cl.held >= 1).astype(float), minlength=len(files)), one)
    M["clean20_5"] = (np.bincount(fc, weights=(cl.held >= 5).astype(float), minlength=len(files)), one)
    s = owner_sittings(u)
    k = u.sitting.to_numpy()
    inown = pd.Series(k).isin(s.index).to_numpy()
    x = pd.DataFrame({"sitting": k[inown], "held": held[inown], "pos": u.pos.to_numpy()[inown]})
    x["banpos"] = x.sitting.map(s.banpos)
    anyh = x.groupby("sitting").held.any().reindex(s.index)
    byh = x[x.pos == x.banpos].groupby("sitting").held.any().reindex(s.index, fill_value=False)
    bob = x[x.pos <= x.banpos].groupby("sitting").held.any().reindex(s.index, fill_value=False)
    fs = pd.Categorical(s.dossier, categories=files).codes

    def ssum(v):
        return np.bincount(fs, weights=np.asarray(v, float), minlength=len(files))

    for rl, nm in RULES.items():
        r = (s.ruling == rl).to_numpy()
        M[f"{nm}_any"] = (ssum(r & anyh.to_numpy()), ssum(r))
    hb = (s.ruling == "ban").to_numpy() & s.banpos.notna().to_numpy()
    M["ban_by_held"] = (ssum(hb & byh.to_numpy()), ssum(hb))
    M["ban_by_or_before"] = (ssum(hb & bob.to_numpy()), ssum(hb))
    return M


def resamples(files, n, seed):
    """n resamples of the files with replacement, as weights (n, files)."""
    return np.random.default_rng(seed).multinomial(len(files), [1 / len(files)] * len(files), size=n).astype(float)


def summarise(M, W):
    """{measure: (point, resampled ratios, numerator, denominator)}."""
    out = {}
    for m, (num, den) in M.items():
        with np.errstate(invalid="ignore", divide="ignore"):
            boot = (W @ num) / (W @ den)
        out[m] = (num.sum() / den.sum() if den.sum() else np.nan, boot, float(num.sum()), float(den.sum()))
    return out


def ci(boot):
    if np.isnan(boot).all():
        return [np.nan, np.nan]
    return [float(np.nanpercentile(boot, 2.5)), float(np.nanpercentile(boot, 97.5))]


def difference(a, b):
    """a - b of two summaries, paired on their resamples: [point, low, high, mark]."""
    p = float(a[0] - b[0])
    lo, hi = ci(a[1] - b[1])
    return [p, lo, hi, "*" if (lo > 0 or hi < 0) else ""]


def evaluate_owner(runs, ls, periods, nb, seed):
    """Per period: the files, lines, labels and owner's sittings it holds, and {run: {level: {measure: summary}}}."""
    out = {}
    for p in periods:
        per, files = {}, None
        for r in runs:
            df = r["owner"][r["owner"].period.to_numpy() == p].reset_index(drop=True)
            u = owner_lines(r["name"], df, ls, p)
            fs = sorted(u.dossier.unique())
            if files is None:
                files, W, first = fs, resamples(fs, nb, seed), u
            elif fs != files:
                raise SystemExit(f"{r['name']} scores other {p} files than {runs[0]['name']}: runs are compared on the "
                                 "same files")
            hl, passed = held(r, df)
            per[r["name"]] = {lv: summarise(owner_measures(u, x, files, passed), W) for lv, x in hl.items()}
        s = owner_sittings(first)
        out[p] = {"files": len(files), "lines": len(first),
                  "hold": {str(h): int((first.hold == h).sum()) for h in (1, 0, -1)},
                  "unlabelled": int(first.hold.isna().sum()),
                  "sittings": {rl: int((s.ruling == rl).sum()) for rl in RULES},
                  "ban_lines": int(s.banpos.notna().sum()), "runs": per}
    return out


def creayun_lines(cfg):
    b = cfg["eval"]["creayun"]
    m = truth_lines(table_path(cfg, b), b)
    return m[m.split == "test_creayun"].reset_index(drop=True)


def evaluate_creayun(runs, m, shares, kind_share, nb, seed):
    """On Creayun's held-out lines m (truth_lines), for the runs that score them: {run: {measure: summary}}, and the
    thresholds, set on each run's owner's validation lines."""
    import pandas as pd
    y = m[KINDS].to_numpy()
    zero = ~m.masked.to_numpy() & (y == 0).all(1)
    groups = {**{k: y[:, j] == 1 for j, k in enumerate(KINDS)}, "any kind": (y == 1).any(1), "all-zero": zero}
    files = sorted(m.dossier.unique())
    f = pd.Categorical(m.dossier, categories=files).codes
    W = resamples(files, nb, seed)
    res, thr, coverage = {}, {}, {}
    for r in runs:
        if r["creayun"] is None:
            continue
        cre = m[["dossier", "id"]].merge(r["creayun"], on=["dossier", "id"], how="left", validate="1:1")
        if cre.c_hold.isna().any():
            raise SystemExit(f"{r['name']}: {int(cre.c_hold.isna().sum())} of Creayun's test_creayun lines are not "
                             "scored")
        val = r["owner"][(r["owner"].period == "val") & (r["owner"].coverage != "reject")]
        th = dict(zip((f"hold {level(s)}" for s in shares), thresholds(val.c_hold.to_numpy(), shares)))
        for k in KINDS:
            th[f"{k} {level(kind_share)}"] = thresholds(val[f"c_{k}"].to_numpy(), (kind_share,))[0]
        thr[r["name"]] = th
        coverage[r["name"]] = cre.coverage.value_counts().to_dict()
        M = {}
        for s in shares:
            rejected = cre.coverage.to_numpy() == "reject"
            model = ~rejected & (cre.c_hold.to_numpy() >= th[f"hold {level(s)}"])
            over = rejected | model
            for g, mask in groups.items():
                M[f"hold {level(s)}|{g}"] = over & mask, mask
                M[f"model hold {level(s)}|{g}"] = model & mask, mask & ~rejected
        covered = cre.coverage.to_numpy() != "reject"
        for k in KINDS:
            over = cre[f"c_{k}"].to_numpy() >= th[f"{k} {level(kind_share)}"]
            M[f"own|{k}"] = over & groups[k] & covered, groups[k] & covered
            M[f"own|all-zero:{k}"] = over & zero & covered, zero & covered
        M = {key: (np.bincount(f, weights=a.astype(float), minlength=len(files)),
                   np.bincount(f, weights=b.astype(float), minlength=len(files))) for key, (a, b) in M.items()}
        res[r["name"]] = summarise(M, W)
    return {"lines": len(m), "files": len(files), "groups": {g: int(v.sum()) for g, v in groups.items()},
            "thresholds": thr, "coverage": coverage, "runs": res}


def side_table(ls):
    """The training period's labelled lines with each kind at 1, and the unmasked all-zero ones, by hold label and
    batch."""
    x = ls["labels"][ls["labels"].period == "train"]
    j = x.drop(columns="batch").merge(ls["lines"].drop(columns="split"), on=["dossier", "id"], how="inner",
                                      validate="1:1")
    y = j[KINDS].to_numpy()
    sel = {**{k: y[:, i] == 1 for i, k in enumerate(KINDS)}, "any kind": (y == 1).any(1),
           "all-zero unmasked": ~j.masked.to_numpy() & (y == 0).all(1)}
    rows = {}
    for k, s in sel.items():
        g = j[s]
        rows[k] = {"lines": len(g), **{str(h): int((g.hold == h).sum()) for h in (1, 0, -1)},
                   "by_batch": {b: int((g.batch == b).sum()) for b in ls["batches"]}}
    return {"labelled": len(x), "without_line": len(x) - len(j), "rows": rows}


def evaluate(runs, sets, creayun, shares, kind_share, nb, seed, periods=PERIODS):
    """Everything eval measures: {runs, labels: {set: {period: ...}}, side: {set: ...}, creayun}, each measure a
    summary (point, resampled ratios, numerator, denominator)."""
    out = {"runs": {r["name"]: {"kind": r["kind"], "models": r["models"],
                                "thresholds": r["thresholds"]} for r in runs},
           "labels": {ls["name"]: evaluate_owner(runs, ls, periods, nb, seed) for ls in sets},
           "side": {ls["name"]: side_table(ls) for ls in sets}}
    out["creayun"] = evaluate_creayun(runs, creayun, shares, kind_share, nb, seed) if creayun is not None else None
    return out


# ---------------------------------------------------------------- the report

def one(s):
    return {"point": float(s[0]), "ci": ci(s[1]), "num": s[2], "den": s[3]}


def plain(out, base):
    """The JSON form: each summary as point, interval, numerator and denominator, and each later run's difference from
    base ([point, low, high, mark])."""
    j = {"runs": out["runs"], "base": base, "labels": {}, "side": out["side"]}
    for name, periods in out["labels"].items():
        j["labels"][name] = {}
        for p, d in periods.items():
            b = d["runs"][base]
            j["labels"][name][p] = {
                **{k: v for k, v in d.items() if k != "runs"},
                "runs": {r: {lv: {m: one(s) for m, s in ms.items()} for lv, ms in lvs.items()}
                         for r, lvs in d["runs"].items()},
                "difference": {r: {lv: {m: difference(s, b[lv][m]) for m, s in ms.items() if m in b[lv]}
                                   for lv, ms in lvs.items() if lv in b}
                               for r, lvs in d["runs"].items() if r != base}}
    c = out["creayun"]
    if c is not None:
        b = c["runs"].get(base)
        j["creayun"] = {**{k: v for k, v in c.items() if k != "runs"},
                        "runs": {r: {m: one(s) for m, s in ms.items()} for r, ms in c["runs"].items()},
                        "difference": {} if b is None else {r: {m: difference(s, b[m]) for m, s in ms.items()}
                                                            for r, ms in c["runs"].items() if r != base}}
    return j


LABEL = {"blocked_share": "lines blocked % (model hold or input rejection)", "model_held_share": "lines held by model %",
         "rejected_share": "lines rejected before model scoring %", "passed_share": "covered lines reaching C %",
         "recall_all": "blocked recall, hold = 1 (model or input rejection)",
         "recall_model_all": "model recall, all hold = 1 lines",
         "recall_model_covered": "model recall, covered hold = 1 lines",
         "recall_owner": "blocked recall, hold = 1 in owner-ruled sittings", "per1000_h0": "hold = 0 lines blocked per 1,000",
         "let_through_held": "marked lines with hold 0 held %",
         "clean20_any": f"clean {CLEAN_LINES}+-line sittings with 1+ held %",
         "clean20_5": f"clean {CLEAN_LINES}+-line sittings with 5+ held %",
         "ban_any": "owner's ban sittings with 1+ line held", "hold_any": "owner's hold sittings with 1+ line held",
         "nothing_any": "owner's nothing sittings with 1+ line held",
         "notban_any": "owner's not-ban sittings with 1+ line held", "ban_by_held": "ban sittings: the ban line held",
         "ban_by_or_before": "ban sittings: a line held by the ban line",
         **{f"kind_{k}": f"blocked recall, hold = 1 lines with {k} at 1" for k in KINDS},
         **{f"model_kind_{k}": f"model recall, covered hold = 1 lines with {k} at 1" for k in KINDS}}
COUNT = {"ban_any", "hold_any", "nothing_any", "notban_any", "ban_by_held", "ban_by_or_before"}
PER = {"per1000_h0"}


def cell(m, s):
    p, (lo, hi), num, den = s["point"], s["ci"], s["num"], s["den"]
    if m in PER:
        return f"{p:.1f} [{lo:.1f}-{hi:.1f}] ({num / 1000:.0f} of {den:.0f})"
    if m in COUNT:
        return f"{num:.0f}/{den:.0f} [{100 * lo:.0f}-{100 * hi:.0f}%]"
    return f"{100 * p:.1f} [{100 * lo:.1f}-{100 * hi:.1f}] ({num:.0f}/{den:.0f})"


def dcell(m, d, den):
    p, lo, hi, mark = d
    if m in PER:
        return f"{p:+.1f} [{lo:+.1f}, {hi:+.1f}]{mark}"
    if m in COUNT:
        return f"{p * den:+.0f} [{100 * lo:+.0f}, {100 * hi:+.0f} pp]{mark}"
    return f"{100 * p:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]{mark}"


def table(res, diff, base, header, rows):
    """A markdown table: one row per (key, label, measure kind), one column per run, then each later run minus base."""
    names = list(res)
    later = [r for r in names if r in diff]
    L = ["| " + " | ".join([header] + names + [f"{r} - {base}" for r in later]) + " |",
         "|" + "---|" * (1 + len(names) + len(later))]
    for key, label, m in rows:
        cells = [cell(m, res[r][key]) if key in res[r] else "" for r in names]
        dcells = [dcell(m, diff[r][key], res[r][key]["den"]) if key in diff[r] else "" for r in later]
        L.append("| " + " | ".join([label] + cells + dcells) + " |")
    return L


def report(j, periods):
    base, names = j["base"], list(j["runs"])
    L = [f"# eval: {', '.join(names)}", "",
         "Cells: the measure [95% interval over resamples of the period's log files, thresholds fixed] (lines or "
         f"sittings counted / the denominator). Differences are from {base}, paired on the same resamples; * = the "
         "interval excludes 0. Sitting counts show their interval as shares, their differences as counts with a pp "
         "interval.", "", "## Runs", ""]
    for r, x in j["runs"].items():
        th = ", ".join(f"{lv} {v:.4f}" for lv, v in x["thresholds"].items())
        m = x["models"]
        L.append(f"- {r}: C {m['c']}; hold thresholds {th}")
    for name, per in j["labels"].items():
        if "test" in per:
            gate = [f"{r} {s['num']:.0f} of {s['den']:.0f}" for r in names
                    if (s := per["test"]["runs"][r].get("2%", {}).get("recall_all"))]
            if gate:
                L += ["", f"Recall at 2% of the hold = 1 test lines, labels {name}: " + "; ".join(gate) + "."]
    for name, per in j["labels"].items():
        timing = ("Ban timing comes from the frozen benchmark." if name == "forward"
                  else "Ban timing comes from the current rulings and saved reads.")
        for p in periods:
            d = per[p]
            L += ["", f"## Labels {name}: the {'validation' if p == 'val' else 'test'} weeks", "",
                  timing, "",
                  f"{d['files']} log files, {d['lines']:,} judged lines (hold = 1 {d['hold']['1']:,}, 0 "
                  f"{d['hold']['0']:,}, -1 {d['hold']['-1']:,}, no label {d['unlabelled']:,}). The owner's sittings: "
                  + ", ".join(f"{rl} {n}" for rl, n in d["sittings"].items())
                  + f" (ban sittings with a ban line {d['ban_lines']})."]
            for lv in dict.fromkeys(lv for r in names for lv in d["runs"][r]):
                res = {r: d["runs"][r][lv] for r in names if lv in d["runs"][r]}
                diff = {r: x[lv] for r, x in d["difference"].items() if lv in x}
                ms = dict.fromkeys(m for r in res for m in res[r])
                L += ["", f"### At {lv}", ""] + table(res, diff, base, "measure", [(m, LABEL[m], m) for m in ms])
    c = j.get("creayun")
    if c and c["runs"] and "test" in periods:
        L += ["", "## Creayun's held-out lines", "",
              f"{c['lines']:,} test_creayun lines in {c['files']} dossiers (server-days). They have no hold label, each "
              "is a line Creayun's own filter flagged, and a line's context is the earlier flagged lines of its server "
              "and day only. Thresholds are each run's own, set on the owner's validation lines. Lines per group: "
              + ", ".join(f"{g} {n:,}" for g, n in c["groups"].items()) + "."]
        keys = next(iter(c["runs"].values()))
        for h in dict.fromkeys(k.split("|")[0] for k in keys if k.startswith("hold ")):
            L += ["", f"### Blocked at {h.split()[1]} (model hold or input rejection)", ""]
            L += table(c["runs"], c["difference"], base, "lines",
                       [(f"{h}|{g}", f"{g} at 1" if g != "all-zero" else "all-zero unmasked", "share")
                        for g in c["groups"]])
            L += ["", "Model holds among covered lines:", ""]
            L += table(c["runs"], c["difference"], base, "lines",
                       [(f"model {h}|{g}", g, "share") for g in c["groups"]])
        L += ["", "Coverage counts: " + "; ".join(f"{r}: {counts}" for r, counts in c["coverage"].items()), "",
              "### Each kind's own output on covered lines, at its validation threshold", ""]
        L += table(c["runs"], c["difference"], base, "lines",
                   [(f"own|{k}", f"{k} at 1", "share") for k in KINDS]
                   + [(f"own|all-zero:{k}", f"all-zero unmasked over the {k} threshold", "share") for k in KINDS])
        L += ["", "Thresholds (logits):", ""]
        L += [f"- {r}: " + ", ".join(f"{k} {v:.2f}" for k, v in th.items()) for r, th in c["thresholds"].items()]
    for name, s in j["side"].items():
        bs = list(next(iter(s["rows"].values()))["by_batch"])
        L += ["", f"## Labels {name}: the training period's lines by hold label", "",
              f"{s['labelled']:,} labelled training-period lines, {s['without_line']:,} of them not a judged line of "
              "the batches.", "",
              f"| lines | count | hold 1 | hold 0 | hold -1 | {' / '.join(bs)} |", "|---|---|---|---|---|---|"]
        for k, r in s["rows"].items():
            t = r["lines"] or 1
            L.append(f"| {k} | {r['lines']} | " + " | ".join(f"{r[h]} ({100 * r[h] / t:.1f}%)" for h in ("1", "0", "-1"))
                     + " | " + " / ".join(str(v) for v in r["by_batch"].values()) + " |")
    return "\n".join(L) + "\n"


def body(cfg, args):
    e = cfg["eval"]
    shares = [float(s) for s in e["holds"]]
    runs = [load_run(cfg, name, shares) for name in args.runs]
    if len({r["name"] for r in runs}) < len(runs):
        raise SystemExit("two runs have the same name")
    loaders = {"forward": forward_labels, "new": batch_labels}
    sets = []
    for name in args.labels or list(loaders):
        if name not in loaders:
            raise SystemExit(f"--labels: {name} is not a label set ({', '.join(loaders)})")
        ls = loaders[name](cfg)
        if ls is None:
            if args.labels:
                raise SystemExit(f"labels {name}: its files do not exist")
            log.info("labels %s: not measured, its files do not exist", name)
            continue
        sets.append(ls)
    if not sets:
        raise SystemExit("no label set's files exist")
    periods = ("val",) if args.validation else PERIODS
    creayun = None if args.validation or all(r["creayun"] is None for r in runs) else creayun_lines(cfg)
    out = evaluate(runs, sets, creayun, shares, float(e["kind_share"]), int(e["resamples"]), int(e["seed"]), periods)
    j = plain(out, runs[0]["name"])
    folder = cfg.path(e["out"]) / "+".join(r["name"] for r in runs)
    write_text(folder / "eval.json", json.dumps(j, indent=1))
    text = report(j, periods)
    write_text(folder / "tables.md", text)
    log.info("%s", text)
    log.info("eval: %s", folder / "tables.md")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run folders: names under [export] out or [train] out, or paths; the "
                                            "others are compared with the first")
    ap.add_argument("--labels", nargs="+", help="the label sets to measure (forward, new; default: each that exists)")
    ap.add_argument("--validation", action="store_true", help="the validation weeks only")
    run("eval", body, ap)
