"""uv run merge: one label per message in data/labels.parquet, and player roles in data/roles.jsonl.

Precedence for a row: data/corrections.jsonl > unsure regrade > regrade set > first pass (label.REGRADES). For a
window labelled twice in one set, the later-written output wins. A row shared by overlapping windows keeps its first window's label. Every line
of a labelled window that is not flagged is ok; rows outside every window are ok by construction (not listed).
Roles naming a player with no ban line in that window are dropped.
"""
import json

import pyarrow as pa
import pyarrow.parquet as pq

from .chat import load_windows
from .core import log, read_jsonl, run, save_state, write_parquet, write_text
from .label import label_sets, read_outputs, scan


def body(cfg, args):
    data = cfg.data
    label = data / "label"
    windows = load_windows(data / "windows.jsonl")
    sets, _, _ = scan(cfg, windows)
    for name, (index, valid, invalid) in sets.items():
        if invalid:
            log.warning("%s: %d of %d not labelled yet; their windows are left out", name, len(invalid), len(index))
    out_dirs = [label / out_dir for _, _, _, out_dir in label_sets(cfg)]
    first = read_outputs(label / "out", sets["shards"][1])
    regrade = {}    # each regrade set replaces the windows it holds
    for name, _, _, out_dir in label_sets(cfg)[1:]:
        regrade.update(read_outputs(label / out_dir, sets[name][1]))
    cpath = data / "corrections.jsonl"
    corrections = {c["row"]: c for c in read_jsonl(cpath)}

    same = diff = 0
    for wid, versions in first.items():
        if len(versions) == 2:
            a = {f["row"]: f["label"] for f in versions[0].get("flags", [])}
            b = {f["row"]: f["label"] for f in versions[1].get("flags", [])}
            for r, _, _ in windows[wid]["lines"]:
                if (a.get(r, "ok") == "ban") == (b.get(r, "ok") == "ban"):
                    same += 1
                else:
                    diff += 1
    log.info("double-labelled windows: %d; line ban/not-ban agreement %.1f%% (%d lines differ)",
             sum(1 for v in first.values() if len(v) == 2), 100 * same / max(same + diff, 1), diff)

    rows, labels, rules, wins, roles_out, seen = [], [], [], [], [], set()
    for wid, w in windows.items():
        chosen = regrade.get(wid, first.get(wid))
        if not chosen:
            continue
        o = chosen[-1]
        flags = {f["row"]: f for f in o.get("flags", [])}
        for r, _, _ in w["lines"]:
            if r in seen:
                continue
            seen.add(r)
            f = flags.get(r)
            rows.append(r)
            labels.append(f["label"] if f else "ok")
            rules.append(f.get("rule") if f else None)
            wins.append(wid)
        banned = {s for r, s, _ in w["lines"] if flags.get(r, {}).get("label") == "ban"}
        for player, role in (o.get("roles") or {}).items():
            if player in banned:
                roles_out.append({"window": wid, "player": player, "role": role})
    applied = 0
    for i, r in enumerate(rows):
        if r in corrections:
            labels[i] = corrections[r]["label"]
            rules[i] = corrections[r].get("rule")
            applied += 1
    unknown = len(corrections) - applied
    if unknown:
        log.warning("%d corrections name rows outside every labelled window; they are ignored", unknown)
    table = pa.table({"row": rows, "label": labels, "rule": rules, "window": wins})
    out = data / "labels.parquet"
    unchanged = out.exists() and pq.read_table(out).equals(table)
    if not unchanged:   # an unchanged file keeps its fingerprint, so what was made from it stays current
        write_parquet(table, out)
    write_text(data / "roles.jsonl", "".join(json.dumps(r) + "\n" for r in roles_out))
    counts = {k: labels.count(k) for k in ("ban", "amb", "ok")}
    log.info("labelled rows in windows: %s (ban %s, amb %s, ok %s); regraded windows %d; corrections %d; roles %d%s",
             f"{len(rows):,}", f"{counts['ban']:,}", f"{counts['amb']:,}", f"{counts['ok']:,}", len(regrade), applied,
             len(roles_out), "; labels.parquet unchanged" if unchanged else "")
    save_state(cfg, "merge", [data / "windows.jsonl", *out_dirs, cpath],
               rows=len(rows), counts=counts, corrections=applied)


def main():
    run("merge", body)
