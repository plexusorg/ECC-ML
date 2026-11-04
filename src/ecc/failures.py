"""<work>/eval/<model>/failures.md: where a model fails, rebuilt with summary.md by every `uv run eval`.

Every listed message shows the last lines of the window the model saw, each with its label and P(ban), the message's
P(ban) in context and, for a classifier, alone (scored with no earlier lines): a message that is high only in context
was flagged for its conversation, one that is high alone for its own words. Labels of messages.parquet rows show
labels.parquet's rule.

- Case sets: the innocent players that set the zero-innocent bar, every offender under it, and at the operating cut
  the bans it misses and the other messages it flags.
- val: the highest-scoring ok rows and the lowest-scoring bans.
- tf: players are offenders when labels.parquet labels one of their rows ban (rows outside every window count as ok);
  at the operating cut, the offenders it misses and the other players it triggers.

The operating cut is the margin at val FPR 1%, where the fast check's trigger sits (docs/spec.md). Without current
val scores, the lists that need it are left out.
"""
from collections import defaultdict

import pyarrow.parquet as pq

from .evaluate import score

SHOW = 15        # messages per list
CASE_CONTEXT = 6  # earlier lines shown for a case-set message
LOG_CONTEXT = 4   # for a val or tf message
TEXT = 160
LABEL = 24       # fits "ban (stated-minor-flirt)"


def plural(n, noun):
    return f"{n} {noun}" + ("" if n == 1 else "s")


def cut_text(s):
    s = " ".join(s.split())
    return s if len(s) <= TEXT else s[:TEXT - 1] + "…"


def message_labels(cfg):
    """{messages.parquet row: (label, rule)} of labels.parquet."""
    t = pq.read_table(cfg.data / "labels.parquet", columns=["row", "label", "rule"]).to_pydict()
    return {r: (l, rule) for r, l, rule in zip(t["row"], t["label"], t["rule"])}


class Report:
    def __init__(self, cfg, platt, cut, items, alone, labels):
        """items(set name) -> [(key, item with the history the model saw)]; alone(targets) -> {(set, key): margin}
        of [(set, key, item)], or {} for a model that cannot score them."""
        self.platt, self.cut, self.set_items, self.alone, self.labels = platt, cut, items, alone, labels
        self.lines, self.shown, self.indexed = [], [], {}

    def items(self, set_name):
        if set_name not in self.indexed:
            self.indexed[set_name] = dict(self.set_items(set_name))
        return self.indexed[set_name]

    def p(self, margin):
        return score({"margin": margin}, self.platt)

    def show(self, set_name, rows_by_conv, r, context):
        """A placeholder for one message; render() fills it once every message's alone score is known."""
        self.shown.append((set_name, r["k"], rows_by_conv, r, context))
        self.lines.append(len(self.shown) - 1)

    def label(self, r):
        lab = self.labels.get(r["k"]) if isinstance(r["k"], int) else None
        return f"{lab[0]} ({lab[1]})" if lab and lab[1] else r["label"]

    def block(self, set_name, k, rows_by_conv, r, context, alone):
        item = self.items(set_name)[k]
        hist = item["history"][-context:]
        conv = rows_by_conv[r["conv"]]
        i = next(j for j, x in enumerate(conv) if x["k"] == k)
        # a context line's score when it is one of this set's scored rows: walk back through both, matching text
        # (val holds only labelled rows, and real-chat cases do not score their context-only lines)
        matched, j = [None] * len(hist), i - 1
        for h in range(len(hist) - 1, -1, -1):
            if j >= 0 and conv[j]["text"] == hist[h][1]:
                matched[h], j = conv[j], j - 1
        a = alone.get((set_name, k))
        head = (f"{r['conv']} · {r['speaker']} · {self.label(r)} · P {self.p(r['margin']):.3f} in context"
                + (f", {self.p(a):.3f} alone" if a is not None else ""))
        out = ["```", head]
        for (s, t), x in zip(hist, matched):
            lab, p = (self.label(x), f"{self.p(x['margin']):.3f}") if x else ("", "")
            out.append(f"   {lab:<{LABEL}} {p:5}  {s}: {cut_text(t)}")
        out += [f">> {self.label(r):<{LABEL}} {self.p(r['margin']):.3f}  {r['speaker']}: {cut_text(r['text'])}", "```"]
        return out

    def render(self):
        targets = [(s, k, self.items(s)[k]) for s, k in dict.fromkeys((s, k) for s, k, *_ in self.shown)]
        alone = self.alone(targets) if targets else {}
        out = []
        for x in self.lines:
            out += self.block(*self.shown[x], alone) if isinstance(x, int) else [x]
        return out

    def text(self, *lines):
        self.lines += lines


def by_conv(rows, ordered_by_key):
    convs = defaultdict(list)
    for r in sorted(rows, key=lambda r: r["k"]) if ordered_by_key else rows:
        convs[r["conv"]].append(r)
    return convs


def case_set(rep, set_name, rows):
    rows = [r for r in rows if r.get("margin") is not None]
    convs = by_conv(rows, False)
    players = defaultdict(list)
    for r in rows:
        if r["label"] != "amb":
            players[(r["conv"], r["speaker"])].append(r)
    best = {key: max(ms, key=lambda m: m["margin"]) for key, ms in players.items()}
    offenders = {key for key, ms in players.items() if any(m["label"] == "ban" for m in ms)}
    innocents = sorted((key for key in players if key not in offenders), key=lambda key: -best[key]["margin"])
    bar = best[innocents[0]]["margin"] if innocents else float("-inf")
    missed = sorted((key for key in offenders if best[key]["margin"] <= bar), key=lambda key: best[key]["margin"])
    rep.text("", f"## {set_name}", "",
             f"Zero-innocent bar: P {rep.p(bar):.3f}. {len(offenders) - len(missed)} of {len(offenders)} offenders "
             f"score above every innocent player. The innocents with the highest messages:")
    for key in innocents[:3]:
        rep.show(set_name, convs, best[key], CASE_CONTEXT)
    if missed:
        rep.text("", f"Offenders under the bar ({len(missed)}), each at their highest ban:")
        for key in missed:
            rep.show(set_name, convs, max((m for m in players[key] if m["label"] == "ban"),
                                          key=lambda m: m["margin"]), CASE_CONTEXT)
    if rep.cut is None:
        return
    bans = sorted((r for r in rows if r["label"] == "ban" and r["margin"] < rep.cut), key=lambda r: r["margin"])
    flagged = sorted((r for r in rows if r["label"] in ("ok", "mute") and r["margin"] >= rep.cut),
                     key=lambda r: -r["margin"])
    rep.text("", f"At the operating cut, P {rep.p(rep.cut):.3f}: {plural(len(bans), 'ban')} missed and "
                 f"{plural(len(flagged), 'other message')} flagged.")
    if bans:
        rep.text("", f"Missed bans, lowest first{f' (first {SHOW})' if len(bans) > SHOW else ''}:")
        for r in bans[:SHOW]:
            rep.show(set_name, convs, r, CASE_CONTEXT)
    if flagged:
        rep.text("", f"Flagged non-bans, highest first{f' (first {SHOW})' if len(flagged) > SHOW else ''}:")
        for r in flagged[:SHOW]:
            rep.show(set_name, convs, r, CASE_CONTEXT)


def val_set(rep, rows):
    rows = [r for r in rows if r["label"] in ("ban", "ok") and r.get("margin") is not None]
    convs = by_conv(rows, True)
    rep.text("", "## val", "", f"The {SHOW} highest-scoring ok rows:")
    for r in sorted((r for r in rows if r["label"] == "ok"), key=lambda r: -r["margin"])[:SHOW]:
        rep.show("val", convs, r, LOG_CONTEXT)
    rep.text("", f"The {SHOW} lowest-scoring bans:")
    for r in sorted((r for r in rows if r["label"] == "ban"), key=lambda r: r["margin"])[:SHOW]:
        rep.show("val", convs, r, LOG_CONTEXT)


def tf_set(rep, rows):
    if rep.cut is None:
        return
    rows = [dict(r, label=rep.labels.get(r["k"], ("ok",))[0]) for r in rows if r.get("margin") is not None]
    convs = by_conv(rows, True)
    players = defaultdict(list)
    for r in rows:
        players[r["speaker"].lower()].append(r)
    offenders = {p for p, ms in players.items() if any(m["label"] == "ban" for m in ms)}
    best = {p: max(ms, key=lambda m: m["margin"]) for p, ms in players.items()}
    missed = sorted((p for p in offenders if best[p]["margin"] < rep.cut), key=lambda p: best[p]["margin"])
    others = sorted((p for p in players if p not in offenders and best[p]["margin"] >= rep.cut),
                    key=lambda p: -best[p]["margin"])
    rep.text("", "## tf", "",
             f"At the operating cut, P {rep.p(rep.cut):.3f}: {len(offenders) - len(missed)} of {len(offenders)} "
             f"labelled offenders triggered, and {len(others)} other players.")
    if missed:
        rep.text("", "Offenders not triggered, each at their highest ban:")
        for p in missed:
            rep.show("tf", convs, max((m for m in players[p] if m["label"] == "ban"), key=lambda m: m["margin"]),
                     LOG_CONTEXT)
    if others:
        rep.text("", f"Other players triggered, each at their highest message"
                     f"{f' (first {SHOW})' if len(others) > SHOW else ''}:")
        for p in others[:SHOW]:
            rep.show("tf", convs, best[p], LOG_CONTEXT)


def report(cfg, name, scored, platt, cut, items, alone):
    """failures.md's lines. scored: {set name: current scored rows} of the sets in the summary."""
    rep = Report(cfg, platt, cut, items, alone,
                 message_labels(cfg) if {"val", "tf"} & set(scored) else {})
    rep.text(f"# {name}: where it fails", "",
             "Each message shows the last lines the model saw before it: label, P(ban), speaker and text. A line "
             "without them is not scored (context only, or unlabelled in val). P alone is the message scored with no "
             "earlier lines." +
             (f" The operating cut is val FPR 1%, P {rep.p(cut):.3f}." if cut is not None else ""))
    for set_name, rows in scored.items():
        if set_name == "val":
            val_set(rep, rows)
        elif set_name == "tf":
            tf_set(rep, rows)
        elif set_name not in ("train", "outside"):
            case_set(rep, set_name, rows)
    return rep.render()
