"""uv run truth BATCH...: the line table of a reader batch, from its reads. Write under [truth] tables/BATCH, or a new
--out folder. Keep an identical existing table and refuse to replace a different one.

One row per player line (chat, private messages, mail, /me, nicknames, guild, Discord relays, signs, books, item names,
tags, lines typed while muted) with the seven kinds, the account its speaker resolved to, the sitting
(`<dossier>|<account>`), up to 5 earlier lines of its conversation within 5 minutes, and the split. A Paper row also
has its run (`dossier.runs`: the server uptime its log file belongs to; rows are in run order, each run's files in
order) with `time` in seconds from the midnight of the run's first day, and what the plugin sends for the line: the
raw text, the raw shown name (chat and Discord lines) and the copies folded into it (id, time, raw text, raw shown
name). A Creayun row's run is its dossier; its rows are in dossier order, each dossier's parts by part number.

The split ([periods]): a file of the owner batch is train before val_from, val before test_from and test from then
on, by the date of the file; every line of any other Paper batch is train. A Creayun dossier is test_creayun when a
hash of its name falls in [truth] test_share, else train.

Labels per kind, from reads a and b ([truth] reads): 1 when both marked the line, -1 when one did, 0 when neither. On a
part the third read ([truth] third) answered, 1 with two of three, -1 with one, 0 with none. A line a read listed as
unsure and no read marked is -1 on every kind (where the third read answered, only if it listed the line unsure too).
off_server is then 0 on a line whose meaning-normalized links are all on the allowlist (off_server_raw keeps the
value before). Normalize exactly once, as the model does. A line
holding a text of the left-out list is -1 on every kind. disputed.json lists the parts, and the lines in them, that a
third read should cover: every line still -1 on some kind outside the parts it answered, left-out lines excepted.

A Paper batch is built once reads a and b have answered every part; its dossiers are rebuilt from the logs for each
line's account, and the build stops unless every part is byte-identical to the part the reads read. A Creayun batch
takes the parts both reads have answered; its rows are the lines of each part's body, labelled from that part's answers
alone, with context from the part's own lines. Creayun has no account behind a name: the account is the shown name.
"""
import argparse
import collections
import hashlib
import json
import os
import uuid
from datetime import date
import re
from pathlib import Path

from . import dossier as D
from . import inputs as I
from .core import log, run, write_text
from .model_text import normalize

KINDS = ["ask_contact", "give_contact", "ask_personal", "give_personal", "off_server", "sexual", "romantic"]
PLAYER = {"chat", "guild", "msg", "mail", "me", "nick", "sign", "book", "item", "tag", "muted", "discord"}
PUBLIC = {"chat", "guild", "me", "discord"}
EVENT = re.compile(r"^([LR]\d+) (\d\d):(\d\d):(\d\d) (\S+) (.*)$")
SPEAKER = re.compile(r"^(.+?): (.*)$")
REPEAT = re.compile(r" \[sent (\d+) times until \d\d:\d\d:\d\d\]$")
CTX_LINES, CTX_SECONDS = 5, 300


def order_key(lid):
    return int(lid[1:]) if lid[0] == "L" else 10 ** 9


# ---------------------------------------------------------------- inputs

def table_path(cfg, batch):
    """The selected version's table. Reader batches remain the source of its evidence."""
    return cfg.path(cfg["truth"]["tables"]) / batch / "truth.parquet"


def left_out(cfg):
    lines = cfg.path(cfg["truth"]["left_out"]).read_text(encoding="utf-8").splitlines()
    return [x.strip().lower() for x in lines if x.strip() and not x.startswith("#")]


def allowlist(cfg):
    allow = []
    for line in cfg.path(cfg["truth"]["allowlist"]).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip().lower()
        if line:
            host, _, prefix = line.partition("/")
            allow.append((host, "/" + prefix if prefix else ""))
    return allow


def all_links_allowed(text, allow):
    links = I.LINK.findall(normalize(text))
    return bool(links) and all(I.allowed(h, p, allow) for h, p in links)


def answer(folder, stem):
    """(marks {id: kinds}, aims {id: [targets]}, unsure ids) of one answer file, or None when it is missing or not
    complete JSON (a file a running read is still writing)."""
    f = folder / f"{stem}.json"
    if not f.exists():
        return None
    try:
        o = json.loads(f.read_text(encoding="utf-8"))
        marks, aims = collections.defaultdict(set), collections.defaultdict(list)
        for e in o["lines"]:
            if e["kind"] in KINDS:
                marks[e["id"]].add(e["kind"])
                aims[e["id"]].append(str(e.get("target", "")).strip())
        return marks, aims, {e["id"] for e in o["unsure"]}
    except (ValueError, KeyError, TypeError):
        return None


def load_read(folder, stems):
    """One read over a batch: ({(dossier, id): kinds}, {(dossier, id)} listed unsure, {(dossier, id): [aims]},
    stems answered). A line in two parts' overlap collects the marks of both."""
    marks, unsure, aims, answered = collections.defaultdict(set), set(), collections.defaultdict(list), set()
    for s in stems:
        a = answer(folder, s)
        if a is None:
            continue
        answered.add(s)
        d = D.dossier_of(s)
        for lid, ks in a[0].items():
            marks[(d, lid)] |= ks
        for lid, xs in a[1].items():
            aims[(d, lid)] += xs
        unsure |= {(d, lid) for lid in a[2]}
    return marks, unsure, aims, answered


def aimed(answers):
    """The most common answer (case folded); a tie goes to a name over 'everyone' over 'unknown', then alphabetically."""
    if not answers:
        return None
    c = collections.Counter(a.lower() for a in answers)
    first = {}
    for a in answers:
        first.setdefault(a.lower(), a)
    return first[min(c, key=lambda k: (-c[k], 2 if k in ("", "unknown") else 1 if k == "everyone" else 0, k))]


def label(row, ma, mb, mc, unsure_by, text, allow, skip):
    """Adds the kinds and the masks to `row`; mc is None where the third read did not answer."""
    for k in KINDS:
        if mc is not None:
            n = (k in ma) + (k in mb) + (k in mc)
            row[k] = 1 if n >= 2 else -1 if n == 1 else 0
        else:
            row[k] = 1 if (k in ma and k in mb) else -1 if (k in ma or k in mb) else 0
    row["unsure_masked"] = False
    if not (ma or mb or mc) and unsure_by and (mc is None or "c" in unsure_by):
        row["unsure_masked"] = True
        for k in KINDS:
            row[k] = -1
    row["off_server_raw"] = row["off_server"]
    row["allowlisted"] = all_links_allowed(text, allow)
    if row["allowlisted"]:
        row["off_server"] = 0
    row["left_out"] = any(x in text.lower() for x in skip)
    if row["left_out"]:
        for k in KINDS:
            row[k] = -1
    row["read_a"], row["read_b"] = sorted(ma), sorted(mb)
    row["read_c"] = sorted(mc) if mc is not None else None
    row["unsure_by"] = unsure_by


def parse_line(line):
    """(id, second, channel, partner, speaker, text, sent) of a player line, or None for any other line."""
    m = EVENT.match(line)
    if not m:
        return None
    base, _, partner = m.group(5).partition(">")
    sm = SPEAKER.match(m.group(6))
    if base not in PLAYER or not sm:
        return None
    text, sent = sm.group(2), 1
    rm = REPEAT.search(text)
    if rm:
        text, sent = text[:rm.start()], int(rm.group(1))
    sec = int(m.group(2)) * 3600 + int(m.group(3)) * 60 + int(m.group(4))
    return m.group(1), sec, base, partner, sm.group(1), text, sent


def context(lines, prev, sec, name=lambda x: x):
    return [{"speaker": name(c[4]), "text": c[5]} for c in
            (lines[j] for j in prev[-CTX_LINES:]) if sec - c[1] <= CTX_SECONDS]


def disputed(df, name, home):
    """The parts a third read needs: lines still -1 on some kind outside the parts it answered, left-out lines
    excepted. home(dossier, id) is the part whose body holds a line."""
    open_ = df[(df[KINDS] == -1).any(axis=1) & df.read_c.isna() & ~df.left_out]
    parts = collections.defaultdict(list)
    for d, lid in zip(open_.dossier, open_.id):
        parts[home(d, lid)].append(lid)
    return dict(batch=name, lines=len(open_), one_read_lines=int((~open_.unsure_masked).sum()),
                unsure_lines=int(open_.unsure_masked.sum()), parts=len(parts),
                by_part={s: sorted(v, key=order_key) for s, v in sorted(parts.items())})


def periods(cfg, name, manifest):
    """{dossier: split} of a Paper batch ([periods]): the owner batch's files by their date (train before val_from,
    val before test_from, test from then on); every file of any other batch is train."""
    p = cfg["periods"]
    if not isinstance(p["val_from"], date) or not isinstance(p["test_from"], date) or p["val_from"] >= p["test_from"]:
        raise SystemExit("[periods] needs the dates val_from before test_from")
    if name != p["owner"]:
        return {m["dossier"]: "train" for m in manifest["dossiers"]}
    out = {}
    for m in manifest["dossiers"]:
        day = date.fromisoformat(m["day"])
        out[m["dossier"]] = "test" if day >= p["test_from"] else "val" if day >= p["val_from"] else "train"
    return out


# ---------------------------------------------------------------- Paper batches

def paper_table(cfg, folder, manifest, stems):
    import pandas as pd
    t = cfg["truth"]
    reads = {}
    for r in t["reads"]:
        rd = load_read(folder / "reads" / r, stems)
        missing = len(stems) - len(rd[3])
        if missing:
            raise SystemExit(f"read {r} of {folder.name} has no answer for {missing} of {len(stems)} parts; the table "
                             f"is built once reads {' and '.join(t['reads'])} have answered every part")
        reads[r] = rd
    C = load_read(folder / "reads" / t["third"], stems) if (folder / "reads" / t["third"]).exists() else None
    if C is not None:
        log.info("third read %s: %d of %d parts answered", t["third"], len(C[3]), len(stems))
    A, B = reads[t["reads"][0]], reads[t["reads"][1]]

    # each line's account and what the plugin sends for it, from the dossiers rebuilt from the logs; the rebuild must
    # be the parts the reads read, byte for byte
    accounts, raws, carried, n_parts = {}, {}, [], 0
    for m, d in D.build_batch(cfg, manifest, folder):
        if d is None:
            raise SystemExit(f"{m['dossier']} builds to no lines")
        for name, text in D.built_parts(d).items():
            n_parts += 1
            p = folder / "parts" / name
            if not p.exists():
                raise SystemExit(f"{name}: the rebuild has a part the batch lacks")
            if p.read_bytes() != D.part_bytes(text):
                raise SystemExit(f"{name}: the rebuilt part differs from the part the reads read; `uv run dossier "
                                 f"{folder.name} --rewrite NAME` writes the rebuilt parts")
        accounts.update({(d["stem"], lid): acct for lid, acct in zip(d["ids"], d["accounts"])})
        raws.update({(d["stem"], lid): r for lid, r in zip(d["ids"], d["raw"])})
        carried += d["carried"]
    for stem, rid, copy in carried:         # a copy of a line of the file before in the run
        if (stem, rid) in raws:
            raws[(stem, rid)]["copies"].append(copy)
    if n_parts != len(stems):
        raise SystemExit(f"the rebuild has {n_parts} parts, the batch {len(stems)}")
    log.info("parts: %d rebuilt from the logs, byte-identical to the parts the reads read", n_parts)

    # each dossier's run (one server uptime over one or more log files) and its first second in the run's time, which
    # counts from the midnight of the run's first day
    logs = cfg.path(cfg["dossier"]["logs"])
    run_of, order, offset = {}, {}, {}
    for folder_name in sorted({m["source"].split("/")[0] for m in manifest["dossiers"]}):
        for k, (f, r) in enumerate(D.runs(logs / folder_name).items()):
            run_of[f"{folder_name}/{f}"], order[f"{folder_name}/{f}"] = r[:2], (folder_name, k)
    for m in manifest["dossiers"]:
        run, first_day = run_of[m["source"]]
        offset[m["dossier"]] = (run, (date.fromisoformat(m["day"]) - date.fromisoformat(first_day)).days * 86400)
    rank = {m["dossier"]: order[m["source"]] for m in manifest["dossiers"]}

    # the lines, each id once, in part order; the part whose body holds each
    by = collections.defaultdict(list)
    for s in stems:
        by[D.dossier_of(s)].append((int(D.PART.match(s).group(2)), s))
    dossiers, in_parts, home = {}, collections.defaultdict(set), {}
    for d, ps in sorted(by.items(), key=lambda x: rank[x[0]]):      # each run's files in order
        seen, rows, day, last = set(), [], 0, None
        for _, s in sorted(ps):
            body = (folder / "parts" / f"{s}.txt").read_text(encoding="utf-8").splitlines()
            marker = next((i for i, ln in enumerate(body) if ln.startswith("--- part ")), -1)
            for i, line in enumerate(body):
                m = EVENT.match(line)
                if not m:
                    continue
                lid = m.group(1)
                in_parts[(d, lid)].add(s)
                if i > marker:
                    home[(d, lid)] = s
                if lid in seen:
                    continue
                seen.add(lid)
                sec = int(m.group(2)) * 3600 + int(m.group(3)) * 60 + int(m.group(4))
                if last is not None and sec + day < last - 6 * 3600:   # past midnight
                    day += 86400
                last = sec = sec + day
                if parse_line(line):        # a player line: its fields as the dossier built them
                    r = raws[(d, lid)]
                    base, _, partner = r["channel"].partition(">")
                    rows.append((lid, sec, base, partner, r["name"], r["text"], r["sent"]))
        dossiers[d] = rows

    side = periods(cfg, folder.name, manifest)
    allow, skip = allowlist(cfg), left_out(cfg)
    out = []
    for d, lines in dossiers.items():
        public, private = [], collections.defaultdict(list)
        for i, (lid, sec, chan, partner, speaker, text, sent) in enumerate(lines):
            key = (d, lid)
            acct, how = accounts[key]
            if chan == "msg":
                pk = frozenset((acct.lower(), partner.lower()))
                prev = private[pk]
            else:
                prev = public
            ma, mb = A[0].get(key, set()), B[0].get(key, set())
            covered = C is not None and bool(in_parts[key] & C[3])
            mc = C[0].get(key, set()) if covered else None
            run, base = offset[d]
            raw = raws[key]
            row = {"dossier": d, "run": run, "id": lid, "time": base + sec, "channel": chan, "partner": partner,
                   "speaker": speaker, "account": acct, "account_how": how,
                   "sitting": f"{d}|{acct.lower()}" if acct else None, "text": text, "sent": sent,
                   "raw_text": raw["raw"], "raw_shown": raw["shown"],
                   "copies": [{"id": c, "time": base + sec + (cs - sec) % 86400, "raw_text": ct, "raw_shown": cn}
                              for c, cs, ct, cn in raw["copies"]],
                   "judged": chan != "discord", "train": True, "context": context(lines, prev, sec)}
            unsure_by = "".join(x for x, r in (("a", A), ("b", B), ("c", C)) if r is not None and key in r[1])
            label(row, ma, mb, mc, unsure_by, text, allow, skip)
            row["aimed_at"] = aimed(A[2].get(key, []) + B[2].get(key, []) + (C[2].get(key, []) if covered else [])) \
                if (ma or mb or mc) else None
            row["split"] = side[d]
            out.append(row)
            if chan == "msg":
                private[pk].append(i)
            elif chan in PUBLIC:
                public.append(i)
    df = pd.DataFrame(out)
    return df, disputed(df, folder.name, lambda d, lid: home[(d, lid)])


# ---------------------------------------------------------------- Creayun batches

def creayun_table(cfg, folder, manifest, stems):
    import pandas as pd
    t = cfg["truth"]
    ra_dir, rb_dir = (folder / "reads" / r for r in t["reads"])
    reads = {}
    for s in stems:
        a, b = answer(ra_dir, s), answer(rb_dir, s)
        if a is not None and b is not None:
            reads[s] = (a, b)
    if not reads:
        raise SystemExit("no part has both answers")
    log.info("parts answered by both reads: %d of %d", len(reads), len(stems))

    # the dossiers of the answered parts, rebuilt from the message table, must be the parts the reads read
    units = {D.dossier_of(s) for s in reads}
    entries = {m["dossier"]: m for m in manifest["dossiers"]}
    built = D.creayun_dossiers(cfg, {(entries[u]["server"], entries[u]["day"]) for u in units})
    missing = units - set(built)
    if missing:
        raise SystemExit(f"{len(missing)} dossiers have no rows in the message table")
    n = 0
    for u in sorted(units):
        d = built[u]
        if d["rows"] != entries[u]["rows"]:
            raise SystemExit(f"{u}: the rebuilt rows differ from the manifest")
        parts = D.built_parts(d)
        if list(parts) != [p["file"] for p in entries[u]["parts"]]:
            raise SystemExit(f"{u}: the rebuilt parts differ from the manifest's")
        for name, text in parts.items():
            if (folder / "parts" / name).read_bytes() != D.part_bytes(text):
                raise SystemExit(f"{name}: the rebuilt part differs from the part the reads read")
            n += 1
    log.info("parts: %d dossiers (%d parts) rebuilt from the message table are byte-identical to the parts the reads "
             "read", len(units), n)

    allow, skip = allowlist(cfg), left_out(cfg)
    rows = []
    for s in sorted(reads, key=lambda s: (D.dossier_of(s), int(D.PART.match(s).group(2)))):     # part2 before part10
        d = D.dossier_of(s)
        server, day = entries[d]["server"], entries[d]["day"]
        body = (folder / "parts" / f"{s}.txt").read_text(encoding="utf-8").splitlines()
        marker = next((i for i, ln in enumerate(body) if ln.startswith("--- part ")), -1)
        lines = []
        for i, line in enumerate(body):
            if EVENT.match(line):
                x = parse_line(line)
                if x is None:
                    raise SystemExit(f"{s}: line {line.split(' ', 1)[0]} is not a player line")
                lines.append(x + (i > marker,))
        (ma_all, aim_a, un_a), (mb_all, aim_b, un_b) = reads[s]
        side = "test_creayun" if int(hashlib.sha1(d.encode()).hexdigest(), 16) % 10000 < t["test_share"] * 10000 \
            else "train"
        public, private = [], collections.defaultdict(list)
        for i, (lid, sec, chan, partner, speaker, text, sent, in_body) in enumerate(lines):
            acct = speaker.lower()
            if chan == "msg":
                pk = frozenset((acct, partner.lower()))
                prev = private[pk]
            else:
                prev = public
            if in_body:
                ma, mb = ma_all.get(lid, set()), mb_all.get(lid, set())
                row = {"dossier": d, "run": d, "id": lid, "time": sec, "channel": chan, "partner": partner,
                       "speaker": speaker, "account": acct, "account_how": "shown", "sitting": f"{d}|{acct}",
                       "text": text, "sent": sent, "judged": True, "train": True,
                       "context": context(lines, prev, sec, str)}
                unsure_by = "".join(x for x, u in (("a", un_a), ("b", un_b)) if lid in u)
                label(row, ma, mb, None, unsure_by, text, allow, skip)
                row["aimed_at"] = aimed(aim_a.get(lid, []) + aim_b.get(lid, [])) if (ma or mb) else None
                row["split"] = side
                row["server"], row["day"], row["row"], row["part"] = server, day, int(lid[1:]), s
                rows.append(row)
            if chan == "msg":
                private[pk].append(i)
            else:
                public.append(i)
    df = pd.DataFrame(rows)
    if df.duplicated(["dossier", "id"]).any():
        raise SystemExit("a line is in the body of two parts")
    part_of = dict(zip(zip(df.dossier, df.id), df.part))
    return df, disputed(df, folder.name, lambda d, lid: part_of[(d, lid)])


# ---------------------------------------------------------------- the command

def write_table(df, path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    schema = pa.Schema.from_pandas(df, preserve_index=False)
    schema = schema.set(schema.get_field_index("read_c"), pa.field("read_c", pa.list_(pa.string())))
    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
    if path.exists():
        if pq.read_table(path).equals(table, check_metadata=False):
            log.info("%s is unchanged; kept its existing bytes", path)
            return
        raise SystemExit(f"refuse to replace {path}; use --out with a new table folder")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    pq.write_table(table, tmp)
    try:
        os.link(tmp, path)  # create the complete file only if another writer has not created it
    except FileExistsError as error:
        raise SystemExit(f"refuse to replace {path}; use --out with a new table folder") from error
    finally:
        tmp.unlink()


def body(cfg, args):
    for name in args.batch:
        folder = D.batch_dir(cfg, name)
        manifest = D.load_manifest(folder)
        if manifest is None:
            raise SystemExit(f"{folder} has no manifest.json")
        stems = [p.stem for p in D.part_files(folder)]
        build = creayun_table if manifest.get("kind") == "creayun" else paper_table
        df, dis = build(cfg, folder, manifest, stems)
        out = args.out / name if args.out else table_path(cfg, name).parent
        disputed_text = json.dumps(dis, indent=1)
        disputed_path = out / "disputed.json"
        if disputed_path.exists() and disputed_path.read_text(encoding="utf-8") != disputed_text:
            raise SystemExit(f"refuse to replace {disputed_path}; use --out with a new table folder")
        write_table(df, out / "truth.parquet")
        if not disputed_path.exists():
            write_text(disputed_path, disputed_text)
        t = df[df.judged]
        log.info("%s: %d rows, %d dossiers, %d sittings, judged %d; with a kind at 1: %d, at -1: %d; disputed: %d "
                 "parts, %d lines -> %s", name, len(df), df.dossier.nunique(), df.sitting.nunique(), len(t),
                 int((t[KINDS] == 1).any(axis=1).sum()), int((t[KINDS] == -1).any(axis=1).sum()), dis["parts"],
                 dis["lines"], out)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("batch", nargs="+", help="batches: folders under [dossier] batches")
    p.add_argument("--out", type=Path, help="write each table to a new OUT/<batch>/ instead of [truth] tables")
    run("truth", body, p)
