"""uv run labels [BATCH...]: the judge's cases for the hold labels, and the sittings the owner ruled.

The owner's sittings. `[labels] rulings` is the ordered list of the owner's rulings files; the first that rules a
sitting gives its ruling and `by` (when it names no `by`, the next file with the same ruling that names one). An entry
is keyed `<dossier>|<name>` and lands on the sitting of each of its case's own lines (the player's lines, `>>`, of
the case he ruled: the ruling page or the rendered cases its `cases` names), every one of them if the lines are now
over more than one sitting; an entry with no case anywhere lands on the judged lines of its dossier whose speaker has
that name (`account`). An entry that lands on no judged line stops the run.
Several entries of one file on one sitting: the strongest ruling stands (STRENGTH), trusted if any of them is. A file
with `take = "trusted"` gives only its trusted sittings, one with `take = "unsure"` only the sittings it rules unsure;
any other file gives its rulings (block and warn are hold, no ban is not ban), and an entry of it that is unsure or
carries no ruling leaves the sitting to the next file.

The cases. One per sitting of a batch with a judged line marked (a kind at 1), except the owner's own sittings
(the sitting's account is one of `[labels] owner`), sittings whose judged lines are all spam raids (`left_out`), and
sittings any entry of a rulings file rules (a ruling other than unsure). A case shows each marked or disputed line of the
player with the lines around it (`data/v4/bundle.toml [judge.render]` before and after; a private message within its
own conversation), in log order with each line's time of day, `···` where lines are skipped, the player's lines `>>`,
each tagged by the tag rule (`[labels] tags`): "readers", the readers' kinds (1) and disputed kinds (-1, unless the
line is masked) with whom the line is aimed at; "none", no tags. A case longer than `max_chars` characters shows only the marked lines' windows,
then the narrower `cut_before` / `cut_after` window around them, then that many marked lines spread evenly over the
sitting (first and last kept) as fit. Beside them, the owner's own cases dated after `[labels] examples_until` that a
rulings file rules, as `[labels] cases` renders them, with id `owner|<case id>`: they measure the judge on the weeks
after its examples and are never labels.

The judge's examples are the owner's ruled cases `uv run bundle` packs (`[bundle] cases` with
`[bundle] rulings`) dated up to `[labels] examples_until`. The run stops when one is dated on or after
`[periods] test_from`, or when one's own lines are on a sitting that is a case.

The reads. Each of `[labels] reads` is a folder under `[labels] answers` of the judge's answers (ruling, by, sure),
keyed by the SHA-256 of the case text it read. A call's answers are a new file; files are never changed, and a later
answer to the same text stands. The run stops, naming them, while a case lacks an answer in any read.

The sitting rulings, one per sitting with a judged line marked and per sitting the owner ruled: the owner's own
sittings `own`; the owner's sittings his ruling (trusted, unsure, nothing, hold, ban, not ban); a sitting whose judged
lines are all spam raids `spam raid`; every other sitting the reads' ruling where they agree, else `split`. A ban's
`by` line is a valid explicit owner `by`, or the exact same valid, nonempty `by` from all selected reads when every
read is sure and rules ban. A read-derived `by` is marked `judge reads`, including on an owner-ruled ban. Select reads
on the owner's rendered cases (`[labels] cases`) whose own lines are on the sitting: `[labels] reads` on those that
are cases here (his cases after
`examples_until`), `[labels] by_reads` on the others (his examples). The run stops, naming them, while one of those
cases lacks an answer in a read of `by_reads`. A `by` must name a judged line of the sitting. Without supported timing,
`by` is absent. Separately, `censor_after` preserves the conservative boundary: a valid explicit owner `by`, else the
earliest valid selected read `by`, else the first line marked contact, sexual or romantic, else the first marked line.
An invalid explicit owner `by` falls back to a marked line. `censor_from` names its source; read-derived boundaries
also retain the case-text hashes in `censor_hashes` (`by_hashes` for supported read-derived timing). A censor boundary
is not a ban deadline.

The line labels, one per judged line: `hold` 1, 0 or -1 (left out of the hold loss) and `ban_here`. Unassessed lines
with a disputed or unsure kind are -1. All-zero unassessed lines and explicit `nothing` sittings are 0. `hold` and
`ban` sittings are 1 on lines with a kind at 1, -1 on lines whose only non-zero kinds are
-1, 0 otherwise, and in a ban sitting -1 on every line after `censor_after`. Only a supported `by` is `ban_here`;
every other sitting -1 on its lines with a non-zero kind, 0 otherwise; spam-raid lines (`left_out`) -1.

Writes `<work>/labels/cases.json`, `<work>/labels/owner.parquet` (one row per owner-ruled sitting: ruling, by, the
file it comes from, its cases, and the other sittings those cases' lines are on), `<work>/labels/sittings.parquet`
(every sitting ruling with its `by`, `censor_after` and their sources and case-text hashes) and each batch's
`hold.parquet` (dossier, id, sitting, sitting_ruling, sitting_source, hold, ban_here); the `hold.parquet` it replaces
is first copied to
`[labels] history`/hold-<the run's time>/<batch>/.
"""
import argparse
import hashlib
import json
import re
import shutil
import tomllib
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

from . import dossier as D
from .core import log, run, write_parquet, write_text
from .inputs import KINDS

TIER = {"nothing": "nothing", "block": "hold", "warn": "hold", "hold": "hold", "ban": "ban", "no ban": "not ban",
        "not ban": "not ban"}
# Which ruling stands when several entries of one file land on one sitting (a later one only when stronger).
STRENGTH = {"nothing": 0, "block": 1, "no ban": 2, "not ban": 2, "warn": 3, "hold": 3, "ban": 4}
SERIOUS = ["ask_contact", "give_contact", "sexual", "romantic"]
TAGS = ("readers", "none")
TAKE = ("rulings", "trusted", "unsure")
READ_RULINGS = ("nothing", "hold", "ban")
PAGE_CASES = re.compile(r"const CASES = (\[.*?\]);\s*\n", re.S)


def account(label):
    """The name a ruling's key holds for a speaker label: the part after '=' when the shown name hides the account,
    else the last word, tags in brackets and a leading '~' removed, lowercased. Also the key of a private message's
    two players."""
    label = re.sub(r"\s*\(replying to .*\)$", "", label)
    if "=" in label:
        label = label.rsplit("=", 1)[1]
    label = re.sub(r"\[[^\]]*\]", " ", label).split()
    return (label[-1] if label else "").lstrip("~").lower()


def hms(sec):
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


# ---------------------------------------------------------------- settings

def render_settings(cfg):
    """The case rendering the live judge is shown: data/v4/bundle.toml [judge.render]."""
    s = tomllib.loads((cfg.path(cfg["bundle"]["private"]) / "bundle.toml").read_text(encoding="utf-8"))
    r = dict(s["judge"]["render"])
    names = r.pop("kinds")
    if list(names) != KINDS:
        raise SystemExit(f"data/v4/bundle.toml [judge.render.kinds] must name {KINDS} in that order")
    return dict(r, names=names)


def sources(cfg):
    """[labels] rulings with their paths resolved."""
    out = []
    for s in cfg["labels"]["rulings"]:
        take = s.get("take", "rulings")
        if take not in TAKE:
            raise SystemExit(f"[labels] rulings: take must be one of {TAKE}, not {take!r}")
        out.append(dict(file=cfg.path(s["file"]), take=take, cases=[cfg.path(c) for c in s["cases"]]))
    return out


def load_tables(cfg, batches):
    from .truth import table_path
    return {b: pd.read_parquet(table_path(cfg, b)) for b in batches}


# ---------------------------------------------------------------- the owner's rulings

def read_rulings(path):
    """{case id: dict(ruling, by, trusted)} of a rulings file: a page's saved file ({"rulings": {id: {ruling, by,
    trusted}}}), final.json ({"rulings": {id: ruling}, "by": {id: line}}) or prior.json ({id: ruling})."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    entries, by = (data["rulings"], data.get("by", {})) if "rulings" in data else (data, {})
    out = {}
    for cid, e in entries.items():
        e = {"ruling": e} if isinstance(e, str) else e
        ruling = e.get("ruling")
        if ruling is not None and ruling != "unsure" and ruling not in TIER:
            raise SystemExit(f"{Path(path).name}: {cid}: unknown ruling {ruling!r}")
        out[cid] = dict(ruling=ruling, by=e.get("by") or by.get(cid) or None, trusted=bool(e.get("trusted")))
    return out


def case_lines(path):
    """{case id: [own line ids]} of a ruling page (its CASES, the lines marked `mine`) or of a rendered cases file
    ({"cases": [{id, text}]}, the lines shown `>> `)."""
    path = Path(path)
    if path.suffix == ".html":
        m = PAGE_CASES.search(path.read_text(encoding="utf-8"))
        if not m:
            raise SystemExit(f"{path}: no CASES")
        return {c["id"]: [ln["id"] for ln in c["lines"] if ln["mine"]] for c in json.loads(m.group(1))}
    doc = json.loads(path.read_text(encoding="utf-8"))
    return {c["id"]: [ln[3:].split(" ", 1)[0] for ln in c["text"].split("\n") if ln.startswith(">> ")]
            for c in doc["cases"]}


class Lines:
    """Where the judged lines of the batches are: (dossier, line id) and (dossier, speaker's name) -> (batch, sitting).
    A dossier is one log file named by its stem; one stem in two batches (two log files of one name from different
    logs folders) stops the run."""

    def __init__(self, tables):
        seen = defaultdict(list)
        for b, t in tables.items():
            for d in t["dossier"].unique():
                seen[d].append(b)
        twice = {d: bs for d, bs in seen.items() if len(bs) > 1}
        if twice:
            raise SystemExit(f"{len(twice)} log file names are in more than one batch: "
                             + "; ".join(f"{d} ({', '.join(bs)})" for d, bs in sorted(twice.items())[:20]))
        self.by_id, self.by_name = {}, defaultdict(set)
        for b, t in tables.items():
            if t.duplicated(["dossier", "id"]).any():
                raise SystemExit(f"{b}: a (dossier, id) is in its table twice")
            j = t[t["judged"] & t["sitting"].notna()]
            for d, i, s, sp in zip(j["dossier"], j["id"], j["sitting"], j["speaker"]):
                self.by_id[(d, i)] = (b, s)
                self.by_name[(d, account(sp))].add((b, s))

    def sittings(self, cid, own):
        """The sittings an entry lands on, and how: by its case's own lines, else by the name in its key."""
        dossier, _, name = cid.partition("|")
        if own is not None:
            return {self.by_id[(dossier, i)] for i in own if (dossier, i) in self.by_id}, "case lines"
        return set(self.by_name.get((dossier, name), ())), "name"


def owner_rulings(srcs, where):
    """The owner's ruled sittings, landed on the judged lines `where` (a Lines) knows. Returns (DataFrame, one row per
    sitting a rulings file decides: batch, sitting, ruling, owner_raw, by, source, by_source, cases, also_on; the set of
    (batch, sitting) any entry rules; a report dict). Stops when an entry lands on no judged line or a rulings file is
    missing."""
    missing = [str(s["file"]) for s in srcs if not s["file"].exists()]
    if missing:
        raise SystemExit("[labels] rulings: missing " + ", ".join(missing))
    lines_cache = {}
    decided, ruled = {}, set()
    unmapped, spans, how_used = [], {}, Counter()
    for s in srcs:
        own_of = {}
        for f in reversed(s["cases"]):                     # the first file of `cases` that holds a case gives its lines
            if f not in lines_cache:
                lines_cache[f] = case_lines(f)
            own_of.update(lines_cache[f])
        here = {}
        for cid, e in read_rulings(s["file"]).items():
            sits, how = where.sittings(cid, own_of.get(cid))
            how_used[(s["file"].name, how)] += 1
            if not sits:
                unmapped.append(f"{s['file'].name}: {cid} ({how})")
                continue
            if len(sits) > 1:
                spans[(s["file"].name, cid)] = sorted(x[1] for x in sits)
            if e["ruling"] not in (None, "unsure"):
                ruled |= sits
            for key in sits:
                rec = dict(e, cases=[cid], also_on=sorted(x[1] for x in sits if x != key))
                if key in here:
                    a = here[key]
                    rank = lambda r: -1 if r in (None, "unsure") else STRENGTH[r]  # noqa: E731
                    win = rec if rank(rec["ruling"]) > rank(a["ruling"]) else a
                    here[key] = dict(win, trusted=a["trusted"] or rec["trusted"], cases=a["cases"] + rec["cases"],
                                     also_on=sorted(set(a["also_on"]) | set(rec["also_on"])))
                else:
                    here[key] = rec
        for key, rec in here.items():
            if key in decided:
                d = decided[key]
                if (d["by"] is None and s["take"] == "rulings" and rec["by"]
                        and TIER.get(rec["ruling"]) == d["ruling"]):
                    d.update(by=rec["by"], by_source=s["file"].name)
                continue
            if s["take"] != "rulings":
                if rec["trusted"] if s["take"] == "trusted" else rec["ruling"] == "unsure":
                    decided[key] = dict(ruling=s["take"], owner_raw=rec["ruling"], by=None)
            elif rec["ruling"] not in (None, "unsure"):
                decided[key] = dict(ruling=TIER[rec["ruling"]], owner_raw=rec["ruling"], by=rec["by"])
            if key in decided:
                decided[key].update(source=s["file"].name, by_source=s["file"].name if decided[key]["by"] else None,
                                    cases=", ".join(rec["cases"]), also_on=", ".join(rec["also_on"]))
    if unmapped:
        raise SystemExit(f"{len(unmapped)} owner rulings land on no judged line: " + "; ".join(unmapped))
    rows = [dict(batch=b, sitting=sid, **v) for (b, sid), v in sorted(decided.items())]
    cols = ["batch", "sitting", "ruling", "owner_raw", "by", "source", "by_source", "cases", "also_on"]
    spans = [dict(source=f, case=cid, sittings=v) for (f, cid), v in spans.items()]
    return pd.DataFrame(rows, columns=cols), ruled, dict(spans=spans, used=dict(how_used))


def first_rulings(srcs):
    """{case id: ruling (nothing, hold, ban, not ban)}: per case id, the first rulings file in order that gives it a
    ruling; none for a case a `take = "trusted"` or `"unsure"` file before it marks so."""
    out, flagged = {}, set()
    for s in srcs:
        for cid, e in read_rulings(s["file"]).items():
            if cid in out or cid in flagged:
                continue
            if s["take"] != "rulings":
                if e["trusted"] if s["take"] == "trusted" else e["ruling"] == "unsure":
                    flagged.add(cid)
            elif e["ruling"] not in (None, "unsure"):
                out[cid] = TIER[e["ruling"]]
    return out


# ---------------------------------------------------------------- the cases

def windows(n_rows, positions, pair, talks, before, after):
    keep = set()
    for p in positions:
        if pair[p]:
            talk = talks[pair[p]]
            i = talk.index(p)
            keep.update(talk[max(0, i - before):i + after + 1])
        else:
            keep.update(range(max(0, p - before), min(n_rows, p + after + 1)))
    return keep


def render_text(player, dossier, lines, names):
    """A case as the judge reads it: a header line, then one line per shown line."""
    rows = []
    for ln in lines:
        if ln["gap"]:
            rows.append("···")
        ch = ln["ch"] + (" to " + ln["to"] if ln["to"] else "")
        s = f"{'>> ' if ln['mine'] else ''}{ln['id']} {ln['t']} {ch} {ln['who']}: {ln['text']}"
        if ln["n"] > 1:
            s += f" (sent {ln['n']} times)"
        aim = " to " + ln["aim"] if ln["aim"] and (ln["kinds"] or ln["maybe"]) else ""
        for k in ln["kinds"]:
            s += f" [{names[k]}{aim}]"
        for k in ln["maybe"]:
            s += f" [maybe: {names[k]}{aim}]"
        rows.append(s)
    return "player: " + player + " · log " + dossier[:10] + "\n" + "\n".join(rows)


def render(d, keep, sid, player, dossier, tags, names):
    lines, last = [], None
    for p in sorted(keep):
        r = d[p]
        mine = r["sitting"] == sid and r["judged"]
        tagged = mine and tags == "readers"
        lines.append(dict(
            id=r["id"], t=hms(int(r["time"]) % 86400), ch=r["channel"], to=r["partner"] or "", who=r["speaker"],
            text=r["text"], n=int(r["sent"]), gap=last is not None and p != last + 1, mine=bool(mine),
            kinds=[k for k in KINDS if tagged and r[k] == 1],
            maybe=[k for k in KINDS if tagged and r[k] == -1 and not r["masked"]],
            aim=r["aim"] if tagged else ""))
        last = p
    return lines, render_text(player, dossier, lines, names)


def cap(d, sid, flagged, marked, pair, talks, player, dossier, tags, rs):
    """The case within rs["max_chars"]: every flagged line's window; else the marked lines' windows; else the cut
    window around them; else the most marked lines, spread evenly, whose cut windows fit."""
    full, cut = (rs["before"], rs["after"]), (rs["cut_before"], rs["cut_after"])
    for pos, (b, a), how in ((flagged, full, None), (marked, full, "marked lines only"),
                             (marked, cut, f"marked lines, window {cut[0]} before {cut[1]} after")):
        lines, text = render(d, windows(len(d), pos, pair, talks, b, a), sid, player, dossier, tags, rs["names"])
        if len(text) <= rs["max_chars"]:
            return lines, text, how
    n = len(marked)
    lo, hi, best = 1, n, None
    while lo <= hi:
        k = (lo + hi) // 2
        pick = sorted({marked[round(i * (n - 1) / max(1, k - 1))] for i in range(k)}) if k > 1 else [marked[0]]
        lines, text = render(d, windows(len(d), pick, pair, talks, *cut), sid, player, dossier, tags, rs["names"])
        if len(text) <= rs["max_chars"]:
            best, lo = (lines, text, k), k + 1
        else:
            hi = k - 1
    if best is None:
        raise SystemExit(f"{sid}: one marked line does not fit in {rs['max_chars']} characters")
    return best[0], best[1], f"{best[2]} of {n} marked lines spread evenly, window {cut[0]} before {cut[1]} after"


def table_cases(batch, t, ruled, owner_names, tags, rs):
    """The cases of one batch table, and its counts."""
    t = t.reset_index(drop=True)
    t["partner"] = t["partner"].fillna("").astype(str)
    t["sitting"] = t["sitting"].astype(object).where(t["sitting"].notna(), None)
    t["masked"] = t["unsure_masked"] | t["left_out"]
    aim = t["aimed_at"].fillna("").astype(str).str.strip()
    t["aim"] = aim.where(~aim.str.lower().isin(["nan", "none"]), "")
    t["old"] = t["speaker"].map(account)
    j = t[t["judged"] & t["sitting"].notna()]
    marked_sit = set(j.loc[(j[KINDS] == 1).any(axis=1), "sitting"])
    own = {s for s in marked_sit if s.split("|", 1)[1] in owner_names}
    spam = {s for s, g in j[j["sitting"].isin(marked_sit)].groupby("sitting") if g["left_out"].all()}
    owner = {s for s in marked_sit if (batch, s) in ruled}
    todo = sorted(marked_sit - own - spam - owner)
    want = defaultdict(list)
    for s in todo:
        want[s.split("|", 1)[0]].append(s)
    cols = ["id", "time", "channel", "partner", "speaker", "text", "sent", "judged", "sitting", "masked", "aim",
            "old"] + KINDS
    cases = []
    for dossier, g in t[t["dossier"].isin(want)].groupby("dossier", sort=True):
        d = g[cols].to_dict("records")
        pair = [frozenset((r["old"], account(r["partner"] or ""))) if r["channel"] == "msg" else None for r in d]
        talks = defaultdict(list)
        for p, key in enumerate(pair):
            if key:
                talks[key].append(p)
        for sid in want[dossier]:
            mine = [p for p, r in enumerate(d) if r["sitting"] == sid and r["judged"]]
            flagged = [p for p in mine if any(d[p][k] == 1 or (d[p][k] == -1 and not d[p]["masked"]) for k in KINDS)]
            marked = [p for p in mine if any(d[p][k] == 1 for k in KINDS)]
            player = Counter(d[p]["speaker"] for p in mine).most_common(1)[0][0]
            lines, text, how = cap(d, sid, flagged, marked, pair, talks, player, dossier, tags, rs)
            counts = {k: sum(1 for p in marked if d[p][k] == 1) for k in KINDS}
            cases.append(dict(
                id=sid, origin="table", batch=batch, dossier=dossier, player=player, counts=counts, total=len(mine),
                marked=len(marked), lines=lines, text=text, mine_ids=[ln["id"] for ln in lines if ln["mine"]],
                capped=how, kind="contact, sexual or romantic" if any(counts[k] for k in SERIOUS) else "other"))
    rep = dict(sittings_marked=len(marked_sit), owners_own=len(own), spam=len(spam), owner_ruled=len(owner),
               cases=len(cases))
    return cases, rep


def owner_cases(srcs, rendered, examples_until):
    """The owner's ruled cases dated after examples_until, as `rendered` shows them: measurement only."""
    doc = json.loads(Path(rendered).read_text(encoding="utf-8"))
    tier = first_rulings(srcs)
    out = []
    for c in doc["cases"]:
        if c["id"][:10] > examples_until and c["id"] in tier:
            out.append(dict(id="owner|" + c["id"], origin="owner", owner_case=c["id"], owner=tier[c["id"]],
                            dossier=c["id"].split("|", 1)[0], text=c["text"]))
    return out


def build(cfg, tables, srcs, tags, batches=None):
    """(cases, owner sittings DataFrame, report, Lines) over the batch tables {batch: DataFrame}; the rulings land on any
    of them, and the table cases are built for `batches` (all by default)."""
    if tags not in TAGS:
        raise SystemExit(f"[labels] tags must be one of {TAGS}, not {tags!r}")
    rs = render_settings(cfg)
    lab = cfg["labels"]
    where = Lines(tables)
    owner, ruled, rep = owner_rulings(srcs, where)
    cases, rep["batches"] = [], {}
    for b in batches or list(tables):
        c, rep["batches"][b] = table_cases(b, tables[b], ruled, lab["owner"], tags, rs)
        cases += c
    cases += owner_cases(srcs, cfg.path(lab["cases"]), str(lab["examples_until"]))
    ids = [c["id"] for c in cases]
    if len(set(ids)) != len(ids):
        raise SystemExit("two cases with one id: " + ", ".join(k for k, v in Counter(ids).items() if v > 1))
    return cases, owner, rep, where


# ---------------------------------------------------------------- the examples

def examples(cfg, where, cases):
    """The judge's examples: the owner's ruled cases `uv run bundle` packs, dated up to [labels] examples_until, as
    dicts (id, ruling, date, text). Stops when one is dated on or after [periods] test_from or its own lines are on a
    sitting that is one of `cases`."""
    from .bundle import judge_examples
    until, test_from = str(cfg["labels"]["examples_until"]), cfg["periods"]["test_from"].isoformat()
    ex = [e for e in judge_examples(cfg)[1] if e["date"] <= until]
    late = [e["id"] for e in ex if e["date"] >= test_from]
    if late:
        raise SystemExit(f"{len(late)} examples are dated on or after [periods] test_from {test_from} "
                         f"([labels] examples_until {until}): " + ", ".join(late))
    judged = {(c["batch"], c["id"]) for c in cases if c["origin"] == "table"}
    own = case_sittings(where, {e["id"]: e["text"] for e in ex})
    both = [f"{cid} ({', '.join(s for _, s in sorted(own[cid] & judged))})" for cid in own if own[cid] & judged]
    if both:
        raise SystemExit(f"{len(both)} examples are on a sitting that is also a case: " + "; ".join(both))
    return ex


def case_sittings(where, texts):
    """{case id: {(batch, sitting)}} of rendered owner cases {id: text}: the sittings of their own (`>>`) lines."""
    out = {}
    for cid, text in texts.items():
        dossier = cid.split("|", 1)[0]
        own = [ln[3:].split(" ", 1)[0] for ln in text.split("\n") if ln.startswith(">> ")]
        out[cid] = {where.by_id[(dossier, i)] for i in own if (dossier, i) in where.by_id}
    return out


def owner_cases_on(where, rendered):
    """{(batch, sitting): [(id, text) of the owner's rendered cases (a cases file) whose own lines are on it]}."""
    texts = {c["id"]: c["text"] for c in json.loads(Path(rendered).read_text(encoding="utf-8"))["cases"]}
    on = defaultdict(list)
    for cid, sits in case_sittings(where, texts).items():
        for s in sits:
            on[s].append((cid, texts[cid]))
    return dict(on)


# ---------------------------------------------------------------- the reads

def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def save_answers(folder, answers, **meta):
    """Adds a file of answers (dicts: hash, case, ruling, by, sure) to a read's folder; never replaces one."""
    bad = [a.get("case") for a in answers if a["ruling"] not in READ_RULINGS]
    if bad:
        raise SystemExit(f"answers with a ruling not in {READ_RULINGS}: {', '.join(map(str, bad))}")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{datetime.now():%Y%m%d-%H%M%S-%f}.json"
    pending = path.with_suffix(".part")
    with open(pending, "x", encoding="utf-8") as f:
        json.dump(dict(meta, answers=answers), f, ensure_ascii=False, indent=1)
    pending.rename(path)
    return path


def load_reads(root, names, lineage=None):
    """{read: {case text hash: answer}} from the answer files of each read's folder under root, in file-name (time)
    order: a later answer to a text stands."""
    out = {}
    for name in names:
        got = {}
        if lineage is not None:
            manifest = Path(root) / name / "lineage.json"
            if not manifest.exists() or json.loads(manifest.read_text(encoding="utf-8")) != {"lineage": lineage}:
                raise SystemExit(f"{manifest} does not record the current judge inputs; run "
                                 "`uv run python -m ecc.judge_reads`")
        for f in sorted((Path(root) / name).glob("*.json")):
            if f.name == "lineage.json":
                continue
            doc = json.loads(f.read_text(encoding="utf-8"))
            if lineage is not None and doc.get("lineage") != lineage:
                raise SystemExit(f"{f} is not of the current judge inputs")
            for a in doc["answers"]:
                got[a["hash"]] = a
        out[name] = got
    return out


def unread(cases, reads):
    """The ids of the cases some read has no answer for."""
    return [c["id"] for c in cases if any(text_hash(c["text"]) not in r for r in reads.values())]


def ban_reads(text, label, reads, by_reads):
    """The answers selected for timing an owner ban with no explicit `by`, from one of his rendered cases:
    `reads` when the text is a case here (its hash in `label`), else `by_reads`."""
    h = text_hash(text)
    return [r[h] for r in (reads if h in label else by_reads).values()]


def unread_by(tables, cases, owner, by_reads, owner_cases_on, owner_names):
    """'<sitting>: <case id> (<reads>)' for each of the owner's cases on an owner ban of `tables` with no `by` that is
    not a case here and that a read of `by_reads` has no answer for."""
    label = {text_hash(c["text"]) for c in cases}
    out = []
    for r in owner.itertuples():
        if (r.batch not in tables or r.ruling != "ban" or (isinstance(r.by, str) and r.by)
                or r.sitting.split("|", 1)[1] in owner_names):
            continue
        for cid, text in owner_cases_on.get((r.batch, r.sitting), []):
            h = text_hash(text)
            lack = [x for x, got in by_reads.items() if h not in got] if h not in label else []
            if lack:
                out.append(f"{r.sitting}: {cid} ({', '.join(lack)})")
    return out


# ---------------------------------------------------------------- sitting rulings and line labels

def judged_lines(t):
    """The judged lines of a batch table, with `pos` their row in the table (log order)."""
    t = t.reset_index(drop=True)
    j = t[t["judged"]].copy()
    j["pos"] = j.index
    j["sitting"] = j["sitting"].astype(object).where(j["sitting"].notna(), None)
    return j.reset_index(drop=True)


def sitting_rulings(tables, cases, owner, reads, by_reads, owner_cases_on, owner_names):
    """One row per sitting with a judged line marked and per sitting the owner ruled: batch, sitting, ruling, source
    (owner, judge, none), supported by/by_from and conservative censor_after/censor_from, with their read case hashes.
    `reads` are the label reads {read: {hash: answer}} (every case read by each),
    `by_reads` the reads whose `by` on the owner's cases that are not cases here (unread_by: every one of them read
    by each) may supply timing for an owner ban with no `by`; `owner_cases_on` maps (batch, sitting) to its owner cases.
    Returns (rows DataFrame, notes)."""
    case_of = {(c["batch"], c["id"]): c for c in cases if c["origin"] == "table"}
    label = {text_hash(c["text"]) for c in cases}
    owner_of = {(r.batch, r.sitting): r for r in owner.itertuples()}
    rows, notes = [], []

    def earliest(g, ids, sid, what):
        ids = [i for i in ids if i]
        pos = g[g["id"].isin(ids)].set_index("id")["pos"]
        outside = sorted(set(ids) - set(pos.index))
        if outside:
            notes.append(f"{sid}: {what} by {', '.join(outside)} is not a judged line of the sitting, not used")
        return None if pos.empty else pos.idxmin()

    for b, t in tables.items():
        j = judged_lines(t)
        marked = set(j.loc[(j[KINDS] == 1).any(axis=1), "sitting"]) - {None}
        sits = marked | {s for (bb, s) in owner_of if bb == b}
        groups = {s: g for s, g in j[j["sitting"].isin(sits)].groupby("sitting")}
        for sid in sorted(sits):
            g = groups[sid]
            r = dict(batch=b, sitting=sid, ruling=None, source="judge", by=None, by_from=None, by_hashes=[],
                     censor_after=None, censor_from=None, censor_hashes=[])
            got = []
            if sid.split("|", 1)[1] in owner_names:
                r.update(ruling="own", source="none")
            elif (b, sid) in owner_of:
                o = owner_of[(b, sid)]
                r.update(ruling=o.ruling, source="owner")
                if o.ruling == "ban":
                    if isinstance(o.by, str) and o.by:
                        r.update(censor_after=o.by, censor_from="owner")
                        if o.by in set(g["id"]):
                            r.update(by=o.by, by_from="owner")
                    else:
                        got = [a for _, text in owner_cases_on.get((b, sid), [])
                               for a in ban_reads(text, label, reads, by_reads)]
                        r["censor_after"] = earliest(g, [a["by"] for a in got], sid, "the owner's case reads'")
                        r["censor_from"] = "judge reads" if r["censor_after"] else None
            elif g["left_out"].all():
                r.update(ruling="spam raid", source="none")
            else:
                c = case_of[(b, sid)]
                got = [reads[x][text_hash(c["text"])] for x in reads]
                rulings = {a["ruling"] for a in got}
                r["ruling"] = got[0]["ruling"] if len(rulings) == 1 and all(a["sure"] for a in got) else "split"
                if r["ruling"] == "ban":
                    r["censor_after"] = earliest(g, [a["by"] for a in got], sid, "the reads'")
                    r["censor_from"] = "judge reads" if r["censor_after"] else None
            if r["ruling"] == "ban" and got:
                hashes = sorted({a["hash"] for a in got})
                if r["censor_after"] is not None:
                    r["censor_hashes"] = hashes
                by = got[0]["by"]
                if (isinstance(by, str) and by and by in set(g["id"])
                        and all(a["sure"] is True and a["ruling"] == "ban" and a["by"] == by for a in got)):
                    r.update(by=by, by_from="judge reads", by_hashes=hashes)
            if r["censor_after"] is not None and r["censor_after"] not in set(g["id"]):
                notes.append(f"{sid}: by {r['censor_after']} ({r['censor_from']}) "
                             "is not a judged line of the sitting, not used")
                r.update(censor_after=None, censor_from=None, censor_hashes=[])
            if r["ruling"] == "ban" and r["censor_after"] is None:
                serious = g[(g[SERIOUS] == 1).any(axis=1)]
                pick = serious if len(serious) else g[(g[KINDS] == 1).any(axis=1)]
                if len(pick):
                    r["censor_after"] = pick.sort_values("pos")["id"].iloc[0]
                    r["censor_from"] = "first contact, sexual or romantic line" if len(serious) else "first marked line"
                else:
                    notes.append(f"{sid}: a ban with no `by` and no marked line; no line is ban_here")
            rows.append(r)
    cols = ["batch", "sitting", "ruling", "source", "by", "by_from", "by_hashes",
            "censor_after", "censor_from", "censor_hashes"]
    return pd.DataFrame(rows, columns=cols), notes


def line_labels(t, sits):
    """The labels of a batch table's judged lines, in table order: dossier, id, sitting, sitting_ruling,
    sitting_source, hold, ban_here; `sits` its rows of sitting_rulings. A disputed or unsure line without a ruling is
    unknown, even when no firm mark made its sitting a judge case. An explicit nothing ruling clears it.
    A ban's censor_after masks later hold targets; only its supported by marks ban_here."""
    j = judged_lines(t)
    n = len(j)
    firm = (j[KINDS] == 1).any(axis=1).to_numpy()
    flag = (j[KINDS] != 0).any(axis=1).to_numpy()
    maybe = ~firm & flag
    hold = np.where(maybe, -1, 0).astype(np.int8)
    ban_here = np.zeros(n, dtype=np.int8)
    ruling = np.full(n, "clean", dtype=object)
    source = np.full(n, "none", dtype=object)
    at = {s: g.index.to_numpy() for s, g in j[j["sitting"].isin(set(sits["sitting"]))].groupby("sitting")}
    for r in sits.itertuples():
        idx = at[r.sitting]
        ruling[idx], source[idx] = r.ruling, r.source
        if r.ruling in ("hold", "ban"):
            hold[idx] = np.where(firm[idx], 1, np.where(maybe[idx], -1, 0))
            if r.ruling == "ban":
                ids = j["id"].to_numpy()[idx]
                if isinstance(r.censor_after, str):
                    k = int(np.flatnonzero(ids == r.censor_after)[0])
                    hold[idx[k + 1:]] = -1
                if isinstance(r.by, str):
                    ban_here[idx[int(np.flatnonzero(ids == r.by)[0])]] = 1
        elif r.ruling == "nothing":
            hold[idx] = 0
        else:
            hold[idx] = np.where(flag[idx], -1, 0)
    hold[j["left_out"].to_numpy()] = -1
    out = j[["dossier", "id", "sitting"]].copy()
    out["sitting_ruling"], out["sitting_source"], out["hold"], out["ban_here"] = ruling, source, hold, ban_here
    return out


def label_lines(tables, cases, owner, reads, by_reads, owner_cases_on, owner_names):
    """({batch: line labels}, sitting rulings, notes). Stops while a case lacks an answer in any read, or an owner case
    that places the `by` of an owner ban lacks one in a read of `by_reads`."""
    missing = unread_by(tables, cases, owner, by_reads, owner_cases_on, owner_names)
    if missing:
        raise SystemExit(f"a read of [labels] by_reads ({', '.join(by_reads)}) has no answer for {len(missing)} of the "
                         "owner's cases on his bans that name no `by`: " + "; ".join(missing))
    missing = unread(cases, reads)
    if missing:
        log.debug("cases without an answer in every read: %s", ", ".join(missing))
        raise SystemExit(f"{len(missing)} cases lack an answer in a read of [labels] reads ({', '.join(reads)}): "
                         + ", ".join(missing[:20]) + (", ... (all of them in the log)" if len(missing) > 20 else ""))
    sits, notes = sitting_rulings(tables, cases, owner, reads, by_reads, owner_cases_on, owner_names)
    out = {b: line_labels(t, sits[sits["batch"] == b]) for b, t in tables.items()}
    return out, sits, notes


def build_label_cases(cfg):
    """Load all configured tables and build the cases used to record judge input lineage."""
    lab = cfg["labels"]
    tables = load_tables(cfg, lab["batches"])
    cases, owner, rep, where = build(cfg, tables, sources(cfg), lab["tags"])
    return tables, cases, owner, rep, where


def resolve_labels(cfg, batches, tables, cases, owner, where):
    """Resolve line labels from the configured rulings and saved reads without writing any files."""
    from .judge_reads import digest, snapshot

    lab = cfg["labels"]
    ex = examples(cfg, where, cases)
    log.info("%d examples dated up to %s: %s", len(ex), lab["examples_until"], dict(Counter(e["ruling"] for e in ex)))
    root = cfg.path(lab["answers"])
    lineage = digest(snapshot(cfg, cases, ex))
    return label_lines({b: tables[b] for b in batches}, cases, owner,
                       load_reads(root, lab["reads"], lineage=lineage), load_reads(root, lab["by_reads"]),
                       owner_cases_on(where, cfg.path(lab["cases"])), lab["owner"])


def write_hold(path, df, keep):
    """Writes a batch's hold.parquet; the one it replaces is first copied into the folder `keep`."""
    path = Path(path)
    if path.exists():
        keep = Path(keep)
        if (keep / path.name).exists():
            raise SystemExit(f"{keep / path.name} exists; the earlier {path} is not copied over it")
        keep.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, keep / path.name)
        log.info("the earlier %s is in %s", path, keep)
    write_parquet(pa.Table.from_pandas(df, preserve_index=False), path)


# ---------------------------------------------------------------- the command

def body(cfg, args):
    lab = cfg["labels"]
    batches = args.batch or lab["batches"]
    unknown = sorted(set(batches) - set(lab["batches"]))
    if unknown:
        raise SystemExit(f"not in [labels] batches: {', '.join(unknown)}")
    tables, cases, owner, rep, where = build_label_cases(cfg)
    out = cfg.work / "labels"
    write_text(out / "cases.json", json.dumps(cases, ensure_ascii=False, indent=1))
    write_parquet(pa.Table.from_pandas(owner, preserve_index=False), out / "owner.parquet")
    for b, r in rep["batches"].items():
        log.info("%s: %d sittings with a marked line; the owner's own %d, spam raids %d, owner-ruled %d; %d cases", b,
                 r["sittings_marked"], r["owners_own"], r["spam"], r["owner_ruled"], r["cases"])
    n = sorted(len(c["text"]) for c in cases if c["origin"] == "table")
    log.info("%d cases (%d of the owner's after %s); characters median %d, max %d; capped %d", len(cases),
             sum(c["origin"] == "owner" for c in cases), lab["examples_until"], n[len(n) // 2] if n else 0,
             n[-1] if n else 0, sum(1 for c in cases if c.get("capped")))
    log.info("owner's sittings: %s; by source: %s", dict(Counter(owner["ruling"])), dict(Counter(owner["source"])))
    log.info("entries landed by: %s", ", ".join(f"{f} {how} {k}" for (f, how), k in rep["used"].items()))
    for s in rep["spans"]:
        log.info("%s: %s is on %d sittings: %s", s["source"], s["case"], len(s["sittings"]), ", ".join(s["sittings"]))
    log.info("-> %s", out)

    hold, sits, notes = resolve_labels(cfg, batches, tables, cases, owner, where)
    for note in notes:
        log.debug("%s", note)
    write_parquet(pa.Table.from_pandas(sits, preserve_index=False), out / "sittings.parquet")
    log.info("sittings: %s; %d notes on `by` lines in the log", dict(Counter(sits["source"] + " " + sits["ruling"])),
             len(notes))
    log.info("ban lines from: %s", dict(Counter(sits.loc[sits["ruling"] == "ban", "by_from"].fillna("none"))))
    keep = cfg.path(lab["history"]) / f"hold-{datetime.now():%Y-%m-%d-%H%M%S}"
    for b, df in hold.items():
        write_hold(D.batch_dir(cfg, b) / "hold.parquet", df, keep / b)
        log.info("%s: %d judged lines, hold 1 / 0 / -1 = %d / %d / %d, ban_here %d", b, len(df),
                 *(int((df["hold"] == v).sum()) for v in (1, 0, -1)), int(df["ban_here"].sum()))
    log.info("wrote hold.parquet of %s; run `uv run backup`", ", ".join(hold))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("batch", nargs="*", help="batches to label (default: [labels] batches)")
    run("labels", body, p)
