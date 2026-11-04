"""uv run python -m ecc.leakage [--drop] FILE...: checks written case files (the [trainset] extra files) against the
eval data before they are trained on.

A line leaks when its normalised text (NFKC, lower case, letters and digits only) equals an eval line's, or when its
token set shares at least MIN_SHARED tokens and at least NEAR of the union with an eval line's (Jaccard). The shared
minimum keeps common short phrases ("how old r u anyway" against "how old r u") from counting. Eval lines are every
line of the [eval.sets] case files and every val.parquet text. Exact matches of NEAR_CHARS characters or fewer ("ok",
"why", "lol") are counted but not offenders: such replies are in every chat. Speakers whose name (case-insensitive) is
a player in messages.parquet are listed too. --drop rewrites each file without its offending lines.
"""
import argparse
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from .chat import load_cases
from .core import Config

MIN_SHARED = 5
NEAR = 0.6
NEAR_CHARS = 15
COMMON = 3000   # tokens in more eval lines than this do not pick candidates; Jaccard still counts them


def norm(text):
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).lower()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drop", action="store_true", help="rewrite the files without their offending lines")
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()
    cfg = Config()
    eval_lines = [(c["text"], "case") for c in
                  load_cases([cfg.path(f) for s in cfg["eval"]["sets"].values() for f in s])]
    eval_lines += [(t, "val") for t in pq.read_table(cfg.work / "training" / "val.parquet", columns=["text"]).column("text")
                   .to_pylist()]
    exact = {}
    sets = []
    postings = defaultdict(list)
    for t, origin in eval_lines:
        n = norm(t)
        exact.setdefault(n, f"{origin}: {t}")
        toks = set(n.split())
        for tok in toks:
            postings[tok].append(len(sets))
        sets.append((toks, f"{origin}: {t}"))

    def offence(text):
        n = norm(text)
        if n in exact:
            return ("exact", exact[n]) if len(n) > NEAR_CHARS else ("short", exact[n])
        toks = set(n.split())
        if len(toks) < MIN_SHARED:
            return None
        cand = {j for tok in toks if len(postings[tok]) <= COMMON for j in postings[tok]}
        cand = [j for j in cand if len(toks & sets[j][0]) >= MIN_SHARED]
        best = max(cand, default=None, key=lambda j: len(toks & sets[j][0]) / len(toks | sets[j][0]))
        if best is not None:
            other, t = sets[best]
            jaccard = len(toks & other) / len(toks | other)
            if jaccard >= NEAR:
                return "near", f"{jaccard:.2f} {t}"
        return None

    speakers = {s.lower() for s in pq.read_table(cfg.data / "messages.parquet", columns=["speaker"])
                .column("speaker").unique().to_pylist()}
    for f in args.files:
        path = Path(f)
        items = load_cases([path])
        counts = Counter()
        bad = set()
        for it in items:
            hit = offence(it["text"])
            if hit:
                counts[hit[0]] += 1
                if hit[0] != "short":
                    bad.add(it["text"])
                    print(f"  {hit[0]}: {it['conv']} {it['label']}|{it['speaker']}|{it['text']}  ~  {hit[1]}")
        real = sorted({it["speaker"] for it in items if it["speaker"].lower() in speakers})
        print(f"{path.name}: {len(items):,} lines; exact {counts['exact']}, near {counts['near']}, short exact "
              f"{counts['short']} (not offenders); speakers who are real players: {', '.join(real) or 'none'}")
        if args.drop and bad:
            kept = [raw for raw in path.read_text(encoding="utf-8").splitlines()
                    if not (raw.count("|") >= 2 and raw.split("|", 2)[2].strip() in bad)]
            path.write_text("\n".join(kept) + "\n", encoding="utf-8")
            print(f"{path.name}: dropped {len(bad)} offending texts")


if __name__ == "__main__":
    main()
