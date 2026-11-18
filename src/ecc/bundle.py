"""Build, verify and optionally upload the explicit C-only serving bundle.

Read the completed C export slot. Ship only graph c and tokenizer c. Preserve
the current judge settings and texts. Read calibrated thresholds from the slot;
keep owner data unchanged. Default to validation recall. The bundle records
exact input rules, their hash, outputs, kinds, thresholds and source provenance.
Build owned file copies in unique staging and verify the complete manifest.
Check the independent reference, real wire vectors and real CPU graph logits.
Use --no-upload to build and check without contacting the artifact service.
"""
import os

os.environ.setdefault("ONEDNN_MAX_CPU_ISA", "AVX2")    # the vectors are made as the host runs the graphs
os.environ.setdefault("RAYON_RS_NUM_CPUS", "6")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import tomllib  # noqa: E402
import uuid  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from . import inputs as I  # noqa: E402
from .backup import sha256  # noqa: E402
from .core import is_current, log, made_from, progress, run, write_text  # noqa: E402
from .export import (C_TOLERANCE, COMPILE, GRAPHS, c_run, folders, lines_path, load_lines, model_inputs,  # noqa: E402
                     graphs_made_from, pair, slot_thresholds)

HEADER = "## C<n>: ruling: <ruling>"        # ECC-Server core.ExampleHeader
FENCE = "~~~~~~"
RULINGS = ("nothing", "hold", "ban", "not ban")

# The worked example: invented words only, times in whole seconds, each line as the plugin sends it.
def _w(id, time, channel, speaker, text, shown=None, target=None, trusted=False):
    return dict(id=id, time=time, channel=channel, speaker=speaker, text=text, shown=shown, target=target,
                trusted=trusted)


EXAMPLE = [
    _w("L1", 0, "chat", "Brindle", "&6hey zorbik wanna build the §lkelpo tower", shown="[Mod] ~Brin"),
    _w("L2", 12, "chat", "zorbik", "ok meet me at vantor spawn", shown="zorbik"),
    _w("L3", 20, "msg", "Brindle", "bring the shiny gromp", target="Zorbik"),
    _w("L4", 25, "sign", "zorbik", "vantor shop open"),
    _w("L5", 30, "chat", "zorbik", "ok meet  me at\rvantor spawn", shown="§azorbik"),
    _w("L6", 31, "discord", "<b>Mallo</b>", "who is on tonight"),
    _w("L7", 40, "msg", "zorbik", "it is in my chest", target="brindle"),
    _w("L8", 44, "sign", "zorbik", "gromps for sale"),
    _w("L9", 1830, "guild", "brindle", "zorbik where did the gromp go", shown="brindle"),
    _w("L10", 1831, "tag", "zorbik", "&a&l"),
    _w("L11", 1840, "chat", "Trusty", "the kelpo tower is done", shown="Trusty", trusted=True),
    _w("L12", 1850, "chat", "zorbik", "pics at https://example.org/vantor/kelpo/tower/" + "gromp" * 10, shown="zorbik"),
    _w("L13", 1851, "chat", "Brindle", "meet at the kelpo tower", shown="Brindle"),
    _w("L14", 1852, "chat", "zorbik", "bWVldCBhdCB0aGUga2VscG8gdG93ZXI=", shown="zorbik"),
    _w("L15", 1853, "chat", "Brindle", "the ke\u200blpo tower", shown="Brindle"),
]


def training_rows(lines):
    """The training code's table rows for wire lines of one stream: each line's text prepared as the dossier prepares
    it, an empty one dropped and a copy folded as the dossier folds them. Returns (rows, dropped ids, folded ids)."""
    from . import dossier as D
    rows, dropped, folded, last = [], [], [], {}
    for e in lines:
        ch, acct = e["channel"], e["speaker"]
        pre = I.clean(e["text"], I.limit(ch))
        if not pre:
            dropped.append(e["id"])
            continue
        if ch == "discord":
            name_pre = I.discord_name(acct)
            account, shown = "", I.redact(name_pre)[0]
        elif ch == "chat":
            name_pre = D.who(e["shown"], acct)
            account, shown = acct, I.redact(name_pre)[0].split("=", 1)[0]
        else:
            name_pre = account = shown = acct
        tag = f"{ch}>{e['target']}" if ch in ("msg", "mail") or (ch == "nick" and e["target"]) else ch
        key = (name_pre.lower(), tag, pre)
        if key in last and e["time"] - last[key] <= D.FOLD_SECONDS:
            last[key] = e["time"]
            folded.append(e["id"])
            continue
        last[key] = e["time"]
        rows.append(dict(id=e["id"], time=e["time"], channel=ch, account=account, speaker=shown,
                         partner=e["target"] if ch == "msg" else None, text=I.redact(pre)[0],
                         raw_text=e["text"], trusted=e["trusted"],
                         judged=ch != "discord" and not e["trusted"]))
    return rows, dropped, folded


def private(cfg):
    return cfg.path(cfg["bundle"]["private"])


# ---------------------------------------------------------------- the judge's texts

def judge_fixed(cfg):
    """The fixed strings of a judge call: the question and the answer format cut from judge-v1.md, the added sentence
    and the separators from prompt.json."""
    j = private(cfg) / "judge"
    src = (j / "judge-v1.md").read_text(encoding="utf-8")
    q = src.split("## Question", 1)[1].split("## Answer format", 1)
    question = q[0].split("\n", 1)[1].strip()
    fmt = q[1].split("## For whoever runs", 1)[0].strip()
    p = json.loads((j / "prompt.json").read_text(encoding="utf-8"))
    return {"QUESTION": question, "ADDED": p["ADDED"], "SEP_CASES": p["SEP_CASES"], "BLOCK_SEP": p["BLOCK_SEP"],
            "SEP_NEW": p["SEP_NEW"], "SEP_FMT": p["SEP_FMT"], "FORMAT": fmt}


def judge_examples(cfg):
    """(cases.json's other fields, the owner's ruled cases in its order: id, ruling, date, text)."""
    rulings = json.loads(cfg.path(cfg["bundle"]["rulings"]).read_text(encoding="utf-8"))["rulings"]
    case_path = cfg.path(cfg["bundle"]["cases"])
    doc = json.loads(case_path.read_text(encoding="utf-8"))
    cases = doc.pop("cases")
    ids = [c["id"] for c in cases]
    if len(set(ids)) != len(ids) or set(ids) != set(rulings):
        raise SystemExit(f"{case_path} and {cfg['bundle']['rulings']} do not hold the same cases: "
                         f"{len(set(ids) - set(rulings))} without a ruling, {len(set(rulings) - set(ids))} without a "
                         "rendered case")
    bad = sorted({r for r in rulings.values() if r not in RULINGS})
    if bad:
        raise SystemExit(f"rulings {bad} are not one of {RULINGS}")
    return doc, [{"id": c["id"], "ruling": rulings[c["id"]], "date": c["date"], "text": c["text"]} for c in cases]


def judge_texts(cfg):
    """(question.md, examples.json) as text."""
    j = private(cfg) / "judge"
    fixed = judge_fixed(cfg)
    md = (j / "question.template.md").read_text(encoding="utf-8")
    md = md.replace("{{FIXED}}", json.dumps(fixed, ensure_ascii=False, indent=1), 1)
    for k in ("QUESTION", "ADDED", "FORMAT"):
        if FENCE in fixed[k] or "{{" in fixed[k]:
            raise SystemExit(f"{k} holds {FENCE} or {{{{, which question.md cannot show")
        md = md.replace("{{" + k + "}}", fixed[k], 1)
    if "{{" in md:
        raise SystemExit(f"{j / 'question.template.md'} has a placeholder that is not FIXED, QUESTION, ADDED or FORMAT")

    doc, examples = judge_examples(cfg)
    out = {"header": HEADER}
    for k, v in doc.items():
        if k == "header":
            continue
        out[k] = v
        if k == "order":
            out["count"] = len(examples)
        if k == "by_source":
            out["by_ruling"] = dict(Counter(e["ruling"] for e in examples))
    out["examples"] = examples
    return md, json.dumps(out, ensure_ascii=False, indent=1) + "\n"


# ---------------------------------------------------------------- bundle.json

def c_example(model_folder, max_tokens):
    """Build actual C example inputs and CPU logits without certifying a release."""
    import pandas as pd
    import torch
    from .models import load_c, score_c
    from .coverage import c_contexts
    torch.set_num_threads(6)
    rows, dropped, folded = training_rows(EXAMPLE)
    df = pd.DataFrame(rows).assign(dossier="example", run="example")
    df["text"] = [I.model_text(t, [], "") for t in df.text]
    ctok = I.CInputs(Path(model_folder) / "tokenizer.json")
    n = max_tokens
    ctx, routes = c_contexts(df, ctok.tok, n)
    scored = [i for i in range(len(df)) if df.judged[i] and routes[i] == "c"]
    rend = I.render(df, ctx, scored)
    cin = ctok.build(rend, n)
    zc = score_c(load_c(model_folder), cin, torch.device("cpu"))
    return {"note": "Invented wire lines. C inputs use the training builder; logits use real Torch fp32 on CPU.",
            "input": [{k: v for k, v in e.items() if v is not None} for e in EXAMPLE],
            "dropped": dropped, "folded": folded, "rejected": df.id[routes == "reject"].tolist(),
            "prepared": dict(zip(df.id, df.text)),
            "scored": [{"id": df.id[i], "context": [df.id[j] for j in ctx[i]],
                        "c": {"context_strings": rend[r][0], "target_string": rend[r][1],
                              "input_ids": cin[r].tolist(), "logits_torch_fp32": zc[r].astype(float).tolist()}}
                       for r, i in enumerate(scored)]}


def example(cfg):
    """The verified C run's exact worked example."""
    f = folders(cfg)
    return c_example(f["c"], c_run(f["c"])["max_tokens"])


def text_rules():
    """bundle.json `text`: how ECC-Server prepares a line's raw text and names, as the dossier prepared the training
    text (ecc.inputs)."""
    from .dossier import FOLD_SECONDS
    lim = ", ".join(f"{k} {v}" for k, v in I.LIMITS.items())
    return {
        "rule": "ECC-Server prepares each line's raw text on arrival, before the repeat key; the scorer (after the link "
                "rule), the context and the judge's render read the prepared text, and the record keeps the raw text "
                "beside it. Lengths count Unicode code points. Whitespace is the code points of `whitespace` (those "
                "for which Python's str.isspace() is true, which is also what \\s matches). The patterns are Python "
                "regular expressions: under (?i) letters compare without case as Python's re.IGNORECASE does (U+212A "
                "counts as k, U+017F as s), and \\d, \\w and \\s are Python's Unicode classes. Each replacement is one "
                "left-to-right pass that finds non-overlapping matches in the text as it stood before the pass, so a "
                "match that a removal forms is kept. "
                "1. Remove every match of patterns.section (a § and the character after it). "
                "2. Remove every match of patterns.codes. "
                "3. Replace each CR (U+000D) with a space, then each LF (U+000A) with ' / '. "
                "4. Remove whitespace at both ends, then replace each run of whitespace with one space (U+0020). "
                f"5. Each match of patterns.url longer than {I.URL_OVER} code points becomes its first {I.URL_KEEP} "
                f"code points followed by '{I.URL_CUT}'. "
                f"6. A text longer than its channel's limit L ({lim}; every other channel {I.LIMIT}) becomes its "
                f"first L code points followed by '{I.CUT}'. "
                f"7. Replace every match of patterns.ipv4 with '{I.IP}', then every match of patterns.ipv6 in the "
                f"result with '{I.IP}'. "
                "A text that is empty after step 6 is delivered and dropped: it is not scored and is never context.",
        "limits": {**I.LIMITS, "default": I.LIMIT},
        "discord_name_limit": I.NAME_LIMIT,
        "fold_seconds": FOLD_SECONDS,
        "url": {"over": I.URL_OVER, "keep": I.URL_KEEP, "mark": I.URL_CUT},
        "cut_mark": I.CUT,
        "ip_mark": I.IP,
        "whitespace": I.whitespace(),
        "patterns": {"section": I.SECTION.pattern, "codes": I.CODES.pattern, "url": I.URL.pattern,
                     "ipv4": I.IPV4.pattern, "ipv6": I.IPV6.pattern, "not_secure": I.NOT_SECURE.pattern,
                     "icons": I.ICONS.pattern, "discord_tags": I.TAGS.pattern},
        "shown": "A chat line's shown name (wire `shown`) takes steps 1 and 2, then loses one match of "
                 "patterns.not_secure at its start, then every match of patterns.icons, then takes step 4; when it is "
                 "then empty, the account name stands for it. That is the shown name before step 7; with step 7 it "
                 "is the prepared shown name.",
        "discord_name": f"A Discord relay line's name (wire `speaker`) loses every match of patterns.discord_tags, "
                        f"then takes steps 1 to 6 with L = {I.NAME_LIMIT}: the Discord name before step 7; with step "
                        "7 it is the prepared Discord name.",
        "names": "The name the judge's render shows for a line: on chat the prepared shown name; on discord the "
                 "prepared Discord name; on every other channel (guild included) the account name.",
        "repeat": f"A line is a copy when an earlier line of the stream has the same key and the key's previous copy "
                  f"(that line or a copy folded into it) is at most {FOLD_SECONDS} seconds older; a copy is folded "
                  "into the line (its count goes up) and is neither scored nor context. The key is (name, channel, "
                  "text) compared exactly. name: on chat the shown name before step 7, followed by '=' and the "
                  "account name unless it contains the account name (both compared after Python's str.lower()); on "
                  "discord the Discord name before step 7; on every other channel the account name; then "
                  "lowercased with Python's str.lower(). channel: the channel; for msg and mail followed by '>' and "
                  "the recipient's account name, for nick set on another player by '>' and that player's account "
                  "name, as written. text: the text after step 6, before step 7."}


def links_rules(canonical):
    return {
        "canonical": canonical,
        "pattern": I.LINK.pattern,
        "rule": "Run on the prepared text. Each match of `pattern` is a link: group 1 its host, group 2 its path "
                "(absent: the empty string). An entry of the module's allowed_links, `host` or `host/prefix`, allows "
                "a link when the link's lowercased host equals the entry's lowercased host or ends with '.' and it, "
                "and the link's lowercased path (trailing punctuation kept) starts with '/' and the entry's lowercased "
                "prefix (any path when the entry has none). Each allowed link, the whole match (scheme and port "
                "included), is replaced by `canonical` in the text the scorer reads; the judge's render and the "
                "record keep the link."}


# The rules ECC-Server implements in code, by their place in bundle.json: their hash is bundle.json `rules.sha256`.
RULES = ["context.rule", "c.roles",
         "c.line_string", "c.input_ids", "channels", "text.rule", "text.whitespace", "text.patterns.section",
         "text.patterns.codes", "text.patterns.url", "text.patterns.ipv4", "text.patterns.ipv6",
         "text.patterns.not_secure", "text.patterns.icons", "text.patterns.discord_tags", "text.shown",
         "text.discord_name", "text.names", "text.repeat", "links.pattern", "links.rule", "coverage.rule",
         "model_text.rule", "scoring.mode"]
RULES_HOW = ("sha256 of the UTF-8 of the values at the paths of `of`, in that order, joined by U+0000; a list is its "
             "items joined by one space, numbers in decimal")


def rules_hash(b):
    """The sha256 of the rules a bundle.json states (RULES_HOW)."""
    def value(path):
        v = b
        for k in path.split("."):
            v = v[k]
        return " ".join(str(x) for x in v) if isinstance(v, list) else v
    return hashlib.sha256("\x00".join(value(p) for p in RULES).encode("utf-8")).hexdigest()


def bundle_content(cfg, hashes):
    """bundle.json without its version. hashes: SHA-256 of every graph and tokenizer file, by bundle path."""
    f = folders(cfg)
    p = private(cfg)
    s = tomllib.loads((p / "bundle.toml").read_text(encoding="utf-8"))
    thr = slot_thresholds(cfg, f)
    graphs = json.loads((f["models"] / "graphs.json").read_text(encoding="utf-8"))
    c_meta = c_run(f["c"])

    def rel(path):
        for root, name in ((cfg.work, ""), (cfg.data, "data/")):
            if Path(path).is_relative_to(root):
                return name + Path(path).relative_to(root).as_posix()
        return str(path)
    g = {}
    for k in GRAPHS:
        g[k] = {"xml": f"models/{k}.xml", "xml_sha256": hashes[f"models/{k}.xml"], "bin": f"models/{k}.bin",
                "bin_sha256": hashes[f"models/{k}.bin"], "bin_bytes": (f["models"] / f"{k}.bin").stat().st_size,
                **graphs[k]}
    render = dict(s["judge"]["render"])
    kinds = render.pop("kinds")
    if list(kinds) != I.KINDS:
        raise SystemExit(f"data/v4/bundle.toml [judge.render.kinds] must name {I.KINDS} in that order")
    if set(thr["kinds"]) != set(I.KINDS):
        raise SystemExit(f"data/v4/thresholds.json kinds must give a threshold for each of {I.KINDS}")
    from .coverage import RULE as COVERAGE_RULE
    content = {
        "coverage": {"rule": COVERAGE_RULE},
        "model_text": {"rule": "v3_meaning_v1"},
        "scoring": {"mode": "c_only"},
        "models": {"c": f"{rel(f['c'])} ({c_meta['run']}), input cut to {c_meta['max_tokens']} tokens"},
        "graphs": g,
        "tokenizers": {"c": {"file": "models/c_tokenizer.json", "sha256": hashes["models/c_tokenizer.json"]}},
        "tokenizer_use": "HuggingFace tokenizers format. Encode each string alone with add_special_tokens = false, "
                         "no truncation, no padding; the ids are the encoding's ids.",
        "special_tokens": {"bos": I.BOS, "pad": I.PAD, "eos": I.EOS},
        "outputs": I.OUTS, "kinds": I.KINDS,
        "channels": I.CHANNELS,
        "text": text_rules(),
        "openvino": {"version": graphs["openvino"], "device": "CPU", "compile": COMPILE,
                     "batch": 1, "attention_mask": "all ones, the input's length"},
        "context": {
            "lines": I.CTX_LINES, "seconds": I.CTX_SECONDS,
            "public_channels": sorted(I.PUBLIC), "placed_channels": sorted(I.PLACED), "private_channel": I.PRIVATE,
            "author_key": "the account name lowercased; a Discord relay line (no account) is 'discord:' + the "
                          "prepared Discord name lowercased",
            "partner_key": "a private message's addressee, its account name lowercased",
            "time": "whole seconds",
            "rule": f"Per stream, in arrival order. A private message (msg) takes the earlier private messages between "
                    f"the same two keys (either direction); a placed line (sign, book, item, tag, nick) the same "
                    f"author's earlier placed lines (any placed channel); every other line the earlier public lines. "
                    f"Of that list take the last {I.CTX_LINES}, then keep those with time(line) - time(earlier) <= "
                    f"{I.CTX_SECONDS}. After a line is handled it is appended: a msg to its pair's list, a placed line "
                    f"to its author's list, a public-channel line (chat, guild, me, discord) to the public list. Any "
                    f"other channel (muted, mail, ...) takes public context but is appended nowhere."},
        "c": {
            "max_tokens": c_meta["max_tokens"],
            "roles": "names = {author(target): 't'}; walking context lines oldest first then the target: the speaker "
                     "is named first, then (msg) the partner; a new key gets 's' + len(names) (s1, s2, ...)",
            "line_string": "tag + role(author) + ': ' + text; tag = '' for chat, '[msg to ' + role(partner) + '] ' for "
                           "msg, '[' + channel + '] ' otherwise",
            "input_ids": "body = for each context string: ids + [2]; then [2] + ids(target string) + [2]; "
                         "input = [0] + last c.max_tokens - 1 ids of body (cut from the left, mid-line if need be)",
            "score": f"c(input_ids [1, T], attention_mask ones [1, T]) -> logits [1, 8]; the hold logit is index "
                     f"{I.HOLD}"},
        "thresholds": {
            "hold": {k: thr["hold_" + k] for k in ("1pct", "2pct", "recall")},
            "default": "recall", "kinds": {k: thr["kinds"][k] for k in I.KINDS},
            "rule": "Reject uncovered targets before folding, context, scoring or judgment. Ask the player to "
                    "shorten the target. Score every covered target with C. Hold when the C hold logit reaches "
                    "hold[threshold]. Mark kinds whose C logits reach kinds[kind], in kinds order. Kind marks "
                    "are diagnostic hints and do not clear a hold.",
            "set_on": thr["set_on"], "kinds_set_on": thr["kinds_set_on"],
            "guard_band": thr["guard_band"], "traffic": thr["traffic"], "provenance": thr["made_from"]},
        "links": links_rules(s["links"]["canonical"]),
        "judge": {"question": "judge/question.md", "examples": "judge/examples.json",
                  "checkpoints": s["judge"]["checkpoints"], "render": {**render, "kinds": kinds}},
        "example": example(cfg)}
    content["rules"] = {"sha256": rules_hash(content), "of": RULES, "how": RULES_HOW}
    return content


# ---------------------------------------------------------------- checks

SEP = {"SEP_CASES": "\n\n# The owner's cases\n\n", "BLOCK_SEP": "\n\n", "SEP_NEW": "\n\n# New cases\n\n",
       "SEP_FMT": "\n\n# Answer format\n\n"}      # ECC-Server core/prompt.go


def check_reader(out):
    """bundle.json, question.md and examples.json read the way ECC-Server's loadBundle reads them (bundle.go: the
    fields and their types, the ```json block of question.md, the prompt's separators, the example header)."""
    b = json.loads((out / "bundle.json").read_text(encoding="utf-8"))
    is_int = lambda x: isinstance(x, int) and not isinstance(x, bool)  # noqa: E731
    problems = []

    def need(ok, what):
        if not ok:
            problems.append(what)

    need(isinstance(b["version"], str) and b["version"], "version")
    files = [b["judge"]["question"], b["judge"]["examples"]]
    for g in b["graphs"].values():
        files += [g["xml"], g["bin"]]
    files += [t["file"] for t in b["tokenizers"].values()]
    for name in files:
        need(name and not Path(name).is_absolute() and ".." not in Path(name).parts and (out / name).is_file(), name)
    c, t, j = b["context"], b["thresholds"], b["judge"]
    need(is_int(c["lines"]) and c["lines"] >= 1 and is_int(c["seconds"]) and c["seconds"] >= 1, "context")
    need(t["hold"] and all(isinstance(v, float) and np.isfinite(v) for v in t["hold"].values()),
         "thresholds")
    need(t["default"] in t["hold"], "thresholds.default")
    need(isinstance(t["kinds"], dict) and set(t["kinds"]) <= set(b["kinds"])
         and all(isinstance(v, float) for v in t["kinds"].values()), "thresholds.kinds")
    cp, r = j["checkpoints"], j["render"]
    need(cp["schedule"] and all(is_int(n) and n >= 1 for n in cp["schedule"]), "judge.checkpoints.schedule")
    need(all(is_int(cp[k]) and cp[k] >= 0 for k in ("every", "wait_seconds", "review_held"))
         and is_int(cp["sitting_end_seconds"]) and cp["sitting_end_seconds"] >= 1, "judge.checkpoints")
    need(all(is_int(r[k]) and r[k] >= 0 for k in ("before", "after", "cut_before", "cut_after"))
         and r["max_chars"] >= 1, "judge.render")
    need(isinstance(r["kinds"], dict) and set(r["kinds"]) == set(b["kinds"]), "judge.render.kinds")
    need(b["links"]["canonical"].strip(), "links.canonical")
    need(all(isinstance(x, str) and x for x in b["channels"]), "channels")
    need(b.get("scoring", {}).get("mode") == "c_only", "scoring.mode")
    need(set(b["graphs"]) == {"c"} and set(b["tokenizers"]) == {"c"}, "C-only graph and tokenizer")
    tx = b["text"]
    need(all(isinstance(tx[k], str) and tx[k] for k in ("rule", "shown", "discord_name", "names", "repeat",
                                                       "cut_mark", "ip_mark")), "text rules")
    need(is_int(tx["limits"]["default"]) and all(is_int(v) and v >= 1 for v in tx["limits"].values()), "text.limits")
    need(all(is_int(c) for c in tx["whitespace"]) and 32 in tx["whitespace"], "text.whitespace")
    need(all(isinstance(v, str) and v for v in tx["patterns"].values()), "text.patterns")
    need(b["links"]["pattern"] and b["links"]["rule"], "links.pattern and links.rule")
    from .coverage import RULE
    need(b.get("coverage", {}).get("rule") == RULE, "coverage.rule")
    need(b.get("model_text", {}).get("rule") == "v3_meaning_v1", "model_text.rule")
    need(b["rules"]["of"] == RULES and rules_hash(b) == b["rules"]["sha256"], "rules.sha256")
    raw = (out / j["question"]).read_bytes().replace(b"\r\n", b"\n").decode("utf-8") + "\n"
    q = json.loads(raw.split("\n```json\n", 1)[1].split("\n```\n", 1)[0])
    need(q["QUESTION"] and q["FORMAT"], "question.md QUESTION and FORMAT")
    for k, v in SEP.items():
        need(q[k] == v, f"question.md {k}")
    ex = json.loads((out / j["examples"]).read_text(encoding="utf-8"))
    need(ex["header"] == HEADER and ex["examples"], "examples.json header")
    need(all(e["ruling"] and e["text"] for e in ex["examples"]), "examples.json rulings and texts")
    if problems:
        raise SystemExit(f"{out}: ECC-Server's loadBundle would reject {', '.join(problems)}")
    log.info("check: bundle.json, question.md and examples.json read as ECC-Server reads them (%d cases); rules "
             "sha256 %s", len(ex["examples"]), b["rules"]["sha256"])


def check_example(out):
    from . import reference
    ok, same, worst = reference.check_example(reference.Bundle(out))
    if not ok:
        raise SystemExit(f"ecc.reference does not rebuild the worked example: features, strings and ids "
                         f"{'equal' if same else 'differ'}, logits max |diff| {worst:.2e} against torch")
    log.info("check: ecc.reference rebuilds the worked example (features, strings and ids equal; logits within %.2e "
             "of torch fp32)", worst)


def uuid_of(name):
    """A stand-in UUID for an account (the logs hold none): the same name gives the same one."""
    import uuid
    return str(uuid.uuid5(uuid.NAMESPACE_URL, "ecc-vectors/" + name.lower())) if name else None


def wire(stream, lid, time, ch, account, partner, raw_text, raw_shown, trusted):
    """One line as the plugin sends it (bundle README, POST /line), with the stream and time the server gives it."""
    target = partner if ch in ("msg", "mail") or (ch == "nick" and partner) else None
    rec = {"stream": stream, "time": int(time), "id": lid, "channel": ch,
           "speaker_uuid": None if ch == "discord" else uuid_of(account),
           "speaker": raw_shown if ch == "discord" else account,
           "target_uuid": uuid_of(target) if target else None, "target": target, "text": raw_text,
           "trusted": bool(trusted)}
    if ch == "chat":
        rec["shown"] = raw_shown
    elif ch == "guild":
        rec["shown"] = account
    return rec


def vector_lines(cfg, table):
    """lines.jsonl: the first `lines` lines of each [vectors] run of the full table (with every copy folded into them,
    at its own time), in wire form; about trusted_share of the lines other than Discord relays marked trusted. Returns
    ([(dossier, id)] of the table rows, in stream order), the records."""
    v = tomllib.loads((private(cfg) / "bundle.toml").read_text(encoding="utf-8"))["vectors"]
    rng = np.random.default_rng(v["seed"])
    rows, recs = [], []
    for spec in v["streams"]:
        t = table[table.run == spec["run"]].iloc[:spec["lines"]]
        if len(t) < spec["lines"]:
            raise SystemExit(f"[vectors] run {spec['run']} has {len(t)} rows in [export] batch, not {spec['lines']}")
        # rows and copies in the dossier's order: by file, second, log lines before record lines, line number
        files = {d: k for k, d in enumerate(dict.fromkeys(t.dossier))}

        def at(dossier, lid, time):
            return files[dossier], time, lid[0] != "L", int(lid[1:])

        events = []
        for r in t.itertuples(index=False):
            trusted = r.channel != "discord" and rng.random() < v["trusted_share"]
            events.append((at(r.dossier, r.id, r.time), (r.dossier, r.id),
                           wire(r.run, f"{r.dossier}:{r.id}", r.time, r.channel, r.account, r.partner, r.raw_text,
                                r.raw_shown, trusted)))
            for c in r.copies:
                events.append((at(r.dossier, c["id"], c["time"]), None,
                               wire(r.run, f"{r.dossier}:{c['id']}", c["time"], r.channel, r.account, r.partner,
                                    c["raw_text"], c["raw_shown"], trusted)))
        for _, key, rec in sorted(events, key=lambda e: e[0]):
            rows.append(key)
            recs.append(rec)
    return rows, recs


def prep_records(cfg, table, recs):
    """prep.jsonl: raw to prepared, by the training code (ecc.inputs, the dossier's who()), for every text and name of
    the vectors, every text and name of the test weeks, and invented edge cases. Each record names what it prepares
    (text, shown, discord_name), its input, and the outputs: pre7 (before step 7) and prepared; a text also linked
    (prepared with data/v4/allowlist.txt's links replaced by links.canonical), a shown name its repeat-key name, a
    Discord name its author key."""
    from . import dossier as D
    from .truth import allowlist
    allow = allowlist(cfg)
    canon = tomllib.loads((private(cfg) / "bundle.toml").read_text(encoding="utf-8"))["links"]["canonical"]

    def text(ch, raw, note=None):
        pre = I.clean(raw, I.limit(ch))
        prep = I.redact(pre)[0]
        linked = I.model_text(prep, allow, canon)
        return {"what": "text", "channel": ch, "raw": raw, **({"note": note} if note else {}), "pre7": pre,
                "prepared": prep, "linked": linked}

    def shown(raw, account, note=None):
        pre = I.shown(raw or "") or account
        return {"what": "shown", "raw": raw, "account": account, **({"note": note} if note else {}), "pre7": pre,
                "prepared": I.redact(pre)[0], "repeat_name": D.who(raw, account).lower()}

    def discord(raw, note=None):
        pre = I.discord_name(raw)
        prep = I.redact(pre)[0]
        return {"what": "discord_name", "raw": raw, **({"note": note} if note else {}), "pre7": pre,
                "prepared": prep, "author": "discord:" + prep.lower()}

    out, seen = [], set()

    def add(r):
        k = json.dumps(r, sort_keys=True, ensure_ascii=False)
        if k not in seen:
            seen.add(k)
            out.append(r)

    def line(ch, account, raw_text, raw_shown):
        add(text(ch, raw_text))
        if ch == "chat":
            add(shown(raw_shown, account))
        elif ch == "discord":
            add(discord(raw_shown))

    for r in recs:
        line(r["channel"], r["speaker"], r["text"], r.get("shown") if r["channel"] == "chat" else r["speaker"])
    for r in table[table.split == "test"].itertuples(index=False):
        line(r.channel, r.account, r.raw_text, r.raw_shown)
        for c in r.copies:
            line(r.channel, r.account, c["raw_text"], c["raw_shown"])
    url = "https://example.org/" + "a" * 60
    for ch, raw, note in [
            ("chat", "a\u00a0b\u3000c\u2028d\u0085e\tf\x1cg", "Unicode whitespace: no-break space, ideographic space, "
                                                               "line separator, NEL, tab, file separator"),
            ("chat", "\u00a0\u3000 lead and trail \u2003\u00a0", "Unicode whitespace at both ends"),
            ("chat", "zero\u200bwidth\ufeffspace", "U+200B and U+FEFF are not whitespace"),
            ("chat", "&\u212a kelvin &\u212aK", "U+212A counts as k in the codes"),
            ("chat", "&\u017f long s", "U+017F is s, which is no code"),
            ("chat", "&&aa §§a §", "one pass: a code a removal forms stays; a § at the end"),
            ("chat", "&#ABCDEF hex &#abcde short &x&R&r&K&O &g &z", "hex colour, short hex, codes in either case"),
            ("chat", "§#ffffff after section", "§ takes one character"),
            ("chat", "line one\nline two\r\nline three\rfour\n", "LF, CRLF, CR, trailing LF"),
            ("book", "page one\n\n\nmore\n" * 120, "an LF-heavy book over its limit"),
            ("book", "x" * 1000, "a book at its limit"), ("book", "x" * 1001, "a book over its limit"),
            ("nick", "&6&lKing" + "n" * 80, "a nickname over its limit after its codes"),
            ("nick", "n" * 80, "a nickname at its limit"),
            ("chat", "x" * 400, "at the limit"), ("chat", "x" * 401, "one over the limit"),
            ("chat", "y" * 380 + " " + url, "a long link straddling the limit"),
            ("chat", "y" * 345 + " https://example.org/" + "b" * 39, "a 59-character link ending past the limit"),
            ("chat", "see HTTPS://Example.org/" + "c" * 36 + " and http://e.x/" + "d" * 44, "links of 60 and 61, "
                                                                                           "any case"),
            ("chat", "join 51.68.1.2 or 1.2.3.4.5 or 999.1.1.1:25565", "IPv4, a five-part number, an address and "
                                                                       "port"),
            ("chat", "ip \u0661\u0662.\u0663.\u0664.\u0665 arabic-indic", "Arabic-Indic digits in an address"),
            ("chat", "v6 2001:db8::1 and fe80::1:2:3 and ::1 and 2001:0db8:0000:0000:0000:ff00:0042:8329",
             "IPv6 forms"),
            ("chat", "at 12:30:45 and a:b:c:d and FE80::ABCD:1", "a time of day, hex words, upper case"),
            ("chat", "url with ip http://10.0.0.1/" + "p" * 60, "an address inside a link that step 5 cuts"),
            ("me", "&a", "empty after its codes"), ("tag", "  \u3000 ", "whitespace only"),
            ("sign", "line one\nline two\nline three\nline four", "a sign's lines"),
            ("msg", "rock&roll &Lol", "& codes inside words"),
    ]:
        add(text(ch, raw, note))
    host = next((h for h, prefix in allow if not prefix), None)
    if host:
        add(text("chat", f"{host}/x, prep.{host}:443/watch?v=1, {host}.invalid/x",
                 "allowed host, subdomain with port, and a host that only contains the allowed host"))
    rule = next(((h, prefix) for h, prefix in allow if prefix and not I.allowed(h, "/", allow)), None)
    if rule:
        host, prefix = rule
        add(text("chat", f"{host}{prefix} {host}{prefix.upper()}?v=1, {host}/",
                 "path prefix, case, trailing punctuation, and a path outside the prefix"))
    for raw, account, note in [
            ("[Not Secure] [Mod] ~Brin", "Brindle", "the client's mark, a rank tag, a nickname not containing the "
                                                     "account"),
            ("[Not Secure]  [Not Secure] x", "x", "only the first mark goes"),
            ("§6[§lAdmin§6] Brindle", "Brindle", "codes, the account inside"),
            ("[item/diamond_sword] |||Brindle", "brindle", "icon codes; the account in another case"),
            ("&a&l", "Zorbik", "empty: the account stands for it"),
            ("[Guild] PrepZib271", "PrepZib271", "a guild tag"),
            ("Mod 1.2.3.4", "Bob", "an address in a name")]:
        add(shown(raw, account, note))
    for raw, note in [("<b>Mallo</b>", "tags"), ("<@123>[Dev] Mallo " + "m" * 70, "over its limit"),
                      ("&cRed 10.0.0.1", "codes and an address")]:
        add(discord(raw, note))
    return out


def vectors(cfg, out, vec_dir):
    """Writes the vectors and checks ecc.reference's outputs on them against the training code and torch."""
    import pandas as pd

    from . import reference
    f = folders(cfg)
    torch_path = f["out"] / "torch.parquet"
    graphs_made = made_from(f["models"] / "graphs.json")
    tp = made_from(torch_path)
    if tp is None or graphs_made is None or tp.get("mode") != "c_only" or tp["c"] != graphs_made["c"]:
        raise SystemExit(f"{torch_path} is not from the models of the graphs; run `uv run export`")
    table = pd.read_parquet(lines_path(cfg))
    keys, recs = vector_lines(cfg, table)
    vec_dir.mkdir(parents=True, exist_ok=True)

    def write_jsonl(path, xs):
        path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in xs), encoding="utf-8", newline="\n")

    write_jsonl(vec_dir / "lines.jsonl", recs)
    write_jsonl(vec_dir / "prep.jsonl", prep_records(cfg, table, recs))
    # scored as ECC-Server scores them once the plugin has sent its default allowed links (data/v4/allowlist.txt)
    b = reference.Bundle(out, allow=reference.read_allow(cfg.path(cfg["truth"]["allowlist"])))
    n = len(recs)
    with progress("vectors (ecc.reference, CPU)", n, "lines") as advance:
        _, secs, timing = reference.score_file(b, vec_dir / "lines.jsonl", vec_dir / "expected.tmp", ids=True,
                                               progress=advance)
    os.replace(vec_dir / "expected.tmp", vec_dir / "expected.jsonl")
    log.info("vectors: %d lines in %.0f s; %s", n, secs, timing)
    got = [json.loads(x) for x in open(vec_dir / "expected.jsonl", encoding="utf-8")]
    # `linked` is under the module's default allowed links, the allowlist the readers' labels used
    bp = reference.Bundle(out, allow=reference.read_allow(cfg.path(cfg["truth"]["allowlist"])), graphs=False)
    n_prep, prep_same = reference.prep_file(bp, vec_dir / "prep.jsonl", vec_dir / "prep.reference.tmp")
    (vec_dir / "prep.reference.tmp").unlink()

    # the training code's inputs and the torch scores of the same lines
    df, _ = load_lines(cfg)
    at = {(d, i): k for k, (d, i) in enumerate(zip(df.dossier, df.id))}
    wid = [f"{d}:{i}" for d, i in zip(df.dossier, df.id)]
    author = I.keys(df)
    copy_of = {f"{d}:{c['id']}": f"{d}:{i}" for d, i, cs in zip(table.dossier, table.id, table.copies) for c in cs}
    problems = Counter()
    kept = [(at[key], rec, r) for key, rec, r in zip(keys, recs, got) if key is not None]
    for key, rec, r in zip(keys, recs, got):
        if key is None:
            problems["copy not folded into its line"] += (r.get("verdict") != "reject" and
                                                          r.get("folded_into") != copy_of[rec["id"]])
    for i, rec, r in kept:
        problems["line dropped, refused or folded"] += bool({"dropped", "refused", "folded_into"} & set(r))
        problems["prepared text"] += r.get("text") != df.text.iloc[i]
        problems["author key"] += r.get("author") != author[i]
        problems["a covered line not trusted and not a Discord relay is not scored"] += ("c" in r) != (
            not rec["trusted"] and rec["channel"] != "discord" and r.get("verdict") != "reject")
    # The wire stream includes trusted lines. Preserve that exemption when building its context.
    df["trusted"] = False
    for i, rec, _ in kept:
        df.loc[i, "trusted"] = rec["trusted"]
    scored = [(i, r) for i, _, r in kept if "c" in r]
    ctx, cids, routes = model_inputs(cfg, df, [i for i, _ in scored])
    import torch
    from .models import load_c, score_c
    torch.set_num_threads(6)
    zc = score_c(load_c(f["c"]), cids, torch.device("cpu"))
    thr = b.cfg["thresholds"]
    worst_c = 0.0
    passed = held = marked = on_threshold = 0
    for k, (i, r) in enumerate(scored):
        problems["context"] += r["context_ids"] != [wid[j] for j in ctx[i]]
        problems["C input ids"] += r.get("c_input_ids") != cids[k].tolist()
        problems["coverage route"] += r.get("coverage") != routes[i]
        tc = zc[k]
        worst_c = max(worst_c, float(np.abs(np.subtract(r["c"], tc)).max()))
        passed += 1
        differ = False
        exact = False
        for name, th in thr["hold"].items():
            differ |= (tc[I.HOLD] >= th) != r[f"held_{name}"]
            exact |= tc[I.HOLD] == th
        t_kinds = [kd for kd, th in thr["kinds"].items() if tc[I.OUTS.index(kd)] >= th]
        differ |= t_kinds != r["kinds"]
        exact |= any(tc[I.OUTS.index(kd)] == th for kd, th in thr["kinds"].items())
        if differ:
            on_threshold += bool(exact)
            problems["decision off a threshold"] += not exact
        held += r["held_recall"]
        marked += bool(r["kinds"])
    problems = {p: v for p, v in problems.items() if v}
    if worst_c > C_TOLERANCE:
        problems[f"C logits {worst_c:.2e} from torch"] = 1
    if prep_same != n_prep:
        problems["prep records ecc.reference prepares otherwise"] = n_prep - prep_same
    chans = Counter(rec["channel"] for _, rec, _ in kept)
    summary = (f"{n} lines in wire form ({len(kept)} table rows: {', '.join(f'{c} {v}' for c, v in chans.most_common())}; "
               f"{n - len(kept)} copies; {sum(rec['trusted'] for _, rec, _ in kept)} trusted; "
               f"{sum(rec['text'] != r.get('text') for _, rec, r in kept)} whose raw text the preparation changes), "
               f"{len(scored)} scored, {passed} passed, {held} held at recall, {marked} marked with a kind. Prepared "
               f"texts, author keys, copies folded, token ids and contexts: "
               f"{'equal to the training code' if not problems else 'see problems'}. Against torch fp32: max |diff| "
               f"C {worst_c:.2e}; decisions differing on a line exactly on a threshold "
               f"{on_threshold}. prep.jsonl: {n_prep} records, {prep_same} prepared by ecc.reference as by the "
               f"training code.")
    write_text(vec_dir / "check.md", summary + ("" if not problems else f"\nProblems (lines): {problems}") + "\n")
    if problems:
        raise SystemExit(f"vectors: {problems}; see {vec_dir / 'check.md'}")
    log.info("check: vectors: %s", summary)


# ---------------------------------------------------------------- building

def readme_text(cfg):
    """Describe the C-only artifact without changing the owner's old private template."""
    t = slot_thresholds(cfg, folders(cfg))
    n = c_run(folders(cfg)["c"])["max_tokens"]
    return (f"# C-only safety classifier\n\nScore each covered target once with C at {n} tokens. "
            "Reject incomplete targets before folding, context or judgment. Read the exact preparation, "
            "normalization, context and token rules from bundle.json. The eight outputs keep their recorded order. "
            "Use the hold output to decide hold or deliver. Kind marks are hints only.\n\n"
            f"Use the recall threshold by default. Calibration: {t['set_on']} "
            f"The CPU logit error guard is {t['guard_band']}. Read diagnostic thresholds and measured model-held, "
            "rejected and total traffic counts from bundle.json. The bundle includes one C graph and one C "
            "tokenizer. Preserve the manifest and verify every file before loading it.\n")


def ship_thresholds(cfg, f):
    """Validate calibrated slot thresholds. Keep the owner's original private files unchanged."""
    slot_thresholds(cfg, f)


def build(cfg):
    """Builds and checks the bundle folder; returns (version, folder)."""
    f = folders(cfg)
    pair(f)
    if not (f["models"] / "graphs.json").exists() or not all((f["models"] / f"{k}.bin").exists() for k in GRAPHS):
        raise SystemExit(f"no graphs in {f['models']}; run `uv run export`")
    if not is_current(f["models"] / "graphs.json", graphs_made_from(f)):
        raise SystemExit(f"the graphs in {f['models']} are not from the current models; run `uv run export`")
    ship_thresholds(cfg, f)
    from .regressions import require_known_offenders
    require_known_offenders(cfg, f)
    sources = {f"models/{k}.{e}": f["models"] / f"{k}.{e}" for k in GRAPHS for e in ("xml", "bin")}
    sources["models/c_tokenizer.json"] = f["c"] / "tokenizer.json"
    log.info("hashing the graphs and tokenizers")
    hashes = {k: sha256(p) for k, p in sources.items()}
    question, examples = judge_texts(cfg)
    readme = readme_text(cfg)
    log.info("worked example (torch, CPU)")
    content = bundle_content(cfg, hashes)
    h = hashlib.sha256(json.dumps(content, sort_keys=True, ensure_ascii=False).encode())
    for text in (question, examples, readme):
        h.update(hashlib.sha256(text.encode()).digest())
    version = f"v4-{h.hexdigest()[:12]}"
    out = cfg.work / "bundles" / version
    if (out / "manifest.json").exists():
        verified_manifest(out)
        log.info("bundle %s already exists: %s", version, out)
        return version, out
    tmp = out.with_name(f"{version}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    (tmp / "models").mkdir(parents=True)
    for rel, src in sources.items():
        shutil.copy2(src, tmp / rel)
        if sha256(tmp / rel) != hashes[rel]:
            raise SystemExit(f"{src} changed while building the bundle; preserve {tmp} for inspection")
    bundle = {"version": version, **content}
    (tmp / "bundle.json").write_text(json.dumps(bundle, indent=1, ensure_ascii=False), encoding="utf-8", newline="\n")
    (tmp / "judge").mkdir()
    (tmp / "judge" / "question.md").write_text(question, encoding="utf-8", newline="\n")
    (tmp / "judge" / "examples.json").write_text(examples, encoding="utf-8", newline="\n")
    (tmp / "README.md").write_text(readme, encoding="utf-8", newline="\n")
    files = {}
    for p in sorted(tmp.rglob("*")):
        if p.is_file():
            rel = p.relative_to(tmp).as_posix()
            files[rel] = {"sha256": hashes.get(rel) or sha256(p), "bytes": p.stat().st_size}
    (tmp / "manifest.json").write_text(json.dumps({"version": version, "files": files}, indent=1) + "\n",
                                       encoding="utf-8")
    verified_manifest(tmp, version)
    os.replace(tmp, out)
    log.info("bundle %s: %s (%.2f GB)", version, out, sum(v["bytes"] for v in files.values()) / 1e9)
    return version, out


def verified_manifest(out, version=None):
    """Refuse changed or incomplete bundle bytes before reuse or upload."""
    m = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    if m.get("version") != (version or out.name) or not m.get("files"):
        raise SystemExit(f"{out}: the manifest does not identify this bundle")
    expected = set(m["files"]) | {"manifest.json"}
    actual = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()}
    if actual != expected:
        raise SystemExit(f"{out}: bundle files differ from its manifest: {sorted(actual ^ expected)}")
    try:
        b = json.loads((out / "bundle.json").read_text(encoding="utf-8"))
        if b.get("proof_only") or b.get("scoring", {}).get("mode") != "c_only" or \
                set(b.get("graphs", {})) != {"c"} or set(b.get("tokenizers", {})) != {"c"}:
            raise ValueError("release requires a production C-only graph and tokenizer")
        required = {"bundle.json", "README.md", b["judge"]["question"], b["judge"]["examples"]}
        marks = {}
        for graph in b["graphs"].values():
            for kind in ("xml", "bin"):
                required.add(graph[kind])
                marks[graph[kind]] = graph[kind + "_sha256"]
        for tokenizer in b["tokenizers"].values():
            required.add(tokenizer["file"])
            marks[tokenizer["file"]] = tokenizer["sha256"]
        if b["version"] != m["version"] or not required <= set(m["files"]):
            raise ValueError("bundle identity or referenced files differ from the manifest")
        if any(m["files"][name]["sha256"] != digest for name, digest in marks.items()):
            raise ValueError("model or tokenizer hashes differ from the manifest")
    except (KeyError, ValueError, OSError, TypeError) as error:
        raise SystemExit(f"{out}: incomplete or inconsistent bundle metadata: {error}") from error
    for name, mark in m["files"].items():
        p = out / name
        if not p.resolve().is_relative_to(out.resolve()) or p.stat().st_size != mark["bytes"] or \
                sha256(p) != mark["sha256"]:
            raise SystemExit(f"{p}: bundle bytes differ from its manifest")
    return m


def bundle_files(out):
    """Every verified file of a bundle folder by its path in it."""
    m = verified_manifest(out)
    return {**{k: v["bytes"] for k, v in m["files"].items()}, "manifest.json": (out / "manifest.json").stat().st_size}


# ---------------------------------------------------------------- W&B

def wandb_path(cfg, alias):
    w = cfg["wandb"]
    return f"{w['entity']}/{w['project']}/{w['artifact']}:{alias}"


def wandb_api(cfg):
    cfg.load_env()
    if not os.environ.get("WANDB_API_KEY"):
        raise SystemExit("set WANDB_API_KEY in .env (wandb.ai/authorize)")
    os.environ.setdefault("WANDB_SILENT", "true")
    os.environ.setdefault("WANDB_DIR", str(cfg.work))     # W&B's local run files go to <work>/wandb/
    import wandb
    return wandb


def uploaded(cfg, version):
    """Whether W&B holds the bundle version (its artifact version carries the version as an alias)."""
    wandb = wandb_api(cfg)
    try:
        wandb.Api().artifact(wandb_path(cfg, version))
        return True
    except wandb.errors.CommError:
        return False


def upload_plan(cfg, version, out):
    files = bundle_files(out)
    lines = [f"W&B artifact {wandb_path(cfg, version)} (type classifier-bundle, alias {version}), "
             f"{len(files)} files, {sum(files.values()) / 1e9:.2f} GB:"]
    lines += [f"  {name}  {size:,} bytes" for name, size in sorted(files.items())]
    return files, "\n".join(lines)


def upload(cfg, version, out):
    """Logs the bundle folder as a version of the [wandb] artifact, each file under its path in the folder, aliased
    with the bundle version."""
    if uploaded(cfg, version):
        log.info("W&B already holds bundle %s (%s)", version, wandb_path(cfg, version))
        return
    wandb = wandb_api(cfg)
    w = cfg["wandb"]
    files, plan = upload_plan(cfg, version, out)
    log.info("uploading %s", plan)
    run_ = wandb.init(entity=w["entity"], project=w["project"], job_type="bundle", name=f"bundle-{version}",
                      config={"version": version})
    art = wandb.Artifact(w["artifact"], type="classifier-bundle", metadata={"version": version})
    for name in files:
        # immutable: W&B uploads the file in place instead of copying 2.7 GB into its staging folder first
        art.add_file(str(out / name), name=name, policy="immutable")
    run_.log_artifact(art, aliases=[version]).wait()
    run_.finish()
    log.info("uploaded %s; `uv run promote %s` makes it the bundle the model host serves", wandb_path(cfg, version),
             version)


def body(cfg, args):
    version, out = build(cfg)
    check_reader(out)
    check_example(out)
    vectors(cfg, out, out.with_name(version + "-vectors"))
    if args.no_upload:
        log.info("not uploaded (--no-upload); the upload would send %s", upload_plan(cfg, version, out)[1])
        return
    upload(cfg, version, out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-upload", action="store_true", help="build and check the bundle without uploading it to W&B")
    run("bundle", body, ap)


if __name__ == "__main__":
    main()
