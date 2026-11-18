"""Reference scorer for a v4 bundle: reads a stream of chat lines as the plugin sends them and handles each the way
ECC-Server does (the text and names prepared, copies folded, OpenVINO on the CPU, batch 1), from bundle.json's rules
and the bundle's graphs and tokenizers alone. It imports only openvino, tokenizers, numpy and the standard library,
and none of the pipeline's code, so it checks the bundle independently of the training code.

    uv run python -m ecc.reference BUNDLE IN.jsonl OUT.jsonl [--threads 6] [--all-c] [--ids] [--allow FILE]
    uv run python -m ecc.reference BUNDLE --check-example
    uv run python -m ecc.reference BUNDLE --prep IN.jsonl OUT.jsonl

IN.jsonl, one line a record, in arrival order: the fields of ECC-Server's POST /line, and the stream and time the
server would give it:
    {"stream": "...", "time": <whole seconds>, "id": "...", "channel": "chat", "speaker_uuid": "..." | null,
     "speaker": "<account name; for discord the Discord name>", "shown": "<chat and guild: the name as the console
     shows it>", "target_uuid": ..., "target": "<msg, mail: the recipient; nick: the player named, when not the
     author>" | null, "text": "<raw>", "trusted": false}
A stream is one conversation space (one server uptime); each stream has its own context and copies. A trusted line
and a Discord relay line are context for later lines and are not scored.

OUT.jsonl, one record per input record, by what happened to the line:
    {"stream", "id", "refused": "unknown channel", "verdict": "hold"}   a channel outside bundle.json `channels`
    {"stream", "id", "dropped": true, "verdict": "deliver"}           empty after the text's step 6
    {"stream", "id", "folded_into": "<id>"}                           a copy (text.repeat)
    {"stream", "id", "context_only": true}
    {"stream", "id", "passed": true, "c": [8 logits], "held_1pct": bool,
     "held_2pct": bool, "verdict": "hold" | "deliver" (at the bundle's default threshold), "kinds": [the kinds marked,
     in the bundle's order]}
Every covered target runs C. --all-c remains an explicit parity option with the same C-only behavior.
--ids adds the prepared text, model text, repeat key, author key, context ids and C input ids.
--allow names an allowed-link file (one host or host/prefix per line, with # comments).
--check-example rebuilds bundle.json's worked example and compares every feature, string and token id
exactly and the logits within 1e-3. --prep reads prep.jsonl records (`what`: text, shown or discord_name; the raw
input) and writes each with the outputs this scorer gives, for comparison with the expected ones.

The host runs the graphs capped to AVX2; set ONEDNN_MAX_CPU_ISA=AVX2 before the run to score as it does.
"""
import argparse
import hashlib
import base64
import binascii
import json
import re
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import openvino as ov
from tokenizers import Tokenizer


class Text:
    """bundle.json `text` and `links`: the preparation of a line's raw text and names, and the link rule."""

    def __init__(self, cfg, allow=()):
        t = cfg["text"]
        self.p = {k: re.compile(v) for k, v in t["patterns"].items()}
        self.limits, self.url, self.cut, self.ip = t["limits"], t["url"], t["cut_mark"], t["ip_mark"]
        self.fold_seconds = t["fold_seconds"]
        ws = "".join(chr(c) for c in t["whitespace"])
        self.ws = ws
        self.ws_run = re.compile("[" + re.escape(ws) + "]+")
        self.name_limit = t["discord_name_limit"]
        lk = cfg["links"]
        self.link, self.canonical = re.compile(lk["pattern"]), lk["canonical"]
        self.allow = [(h, "/" + p if p else "") for h, _, p in (a.lower().partition("/") for a in allow)]

    def steps(self, text, limit):
        """Steps 1-6."""
        text = self.p["section"].sub("", text)
        text = self.p["codes"].sub("", text)
        text = text.replace("\r", " ").replace("\n", " / ")
        text = self.ws_run.sub(" ", text.strip(self.ws))
        over, keep, mark = self.url["over"], self.url["keep"], self.url["mark"]
        text = self.p["url"].sub(lambda m: m.group(0) if len(m.group(0)) <= over else m.group(0)[:keep] + mark, text)
        return text[:limit] + self.cut if len(text) > limit else text

    def step7(self, text):
        return self.p["ipv6"].sub(self.ip, self.p["ipv4"].sub(self.ip, text))

    def shortened(self, raw, channel):
        text = self.p["codes"].sub("", self.p["section"].sub("", raw))
        text = self.ws_run.sub(" ", text.replace("\r", " ").replace("\n", " / ").strip(self.ws))
        return (len(text) > self.limits.get(channel, self.limits["default"]) or
                any(len(m.group(0)) > self.url["over"] for m in self.p["url"].finditer(text)))

    def text(self, raw, channel):
        """(the text after step 6, the prepared text)."""
        pre = self.steps(raw, self.limits.get(channel, self.limits["default"]))
        return pre, self.step7(pre)

    def shown(self, display, account):
        """A chat line's (shown name before step 7, prepared shown name)."""
        s = self.p["codes"].sub("", self.p["section"].sub("", display or ""))
        s = self.p["icons"].sub("", self.p["not_secure"].sub("", s, count=1))
        s = self.ws_run.sub(" ", s.strip(self.ws)) or account
        return s, self.step7(s)

    def discord_name(self, raw):
        pre = self.steps(self.p["discord_tags"].sub("", raw or ""), self.name_limit)
        return pre, self.step7(pre)

    def links(self, text):
        """The text the models read: every allowed link replaced by the canonical one."""
        if not self.allow:
            return text

        def sub(m):
            h, p = m.group(1).lower(), (m.group(2) or "").lower()
            ok = any((h == a or h.endswith("." + a)) and p.startswith(pre) for a, pre in self.allow)
            return self.canonical if ok else m.group(0)
        return self.link.sub(sub, text)

    def read(self, text):
        """Normalize meaning for the models. Keep the original text for repeat keys and judge rendering."""
        if unicodedata.unidata_version != "15.1.0":
            raise ValueError("v3_meaning_v1 requires Unicode 15.1 normalization tables")
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"\r\n|[\n\v\f\r\x1c-\x1e\x85\u2028\u2029]", " ", text)
        text = re.sub(r"[\u202b\u202e\u2067]([^\u202c\u2069]*)[\u202c\u2069]?",
                      lambda m: m.group(1)[::-1], text)
        text = "".join(ch for ch in text if unicodedata.category(ch) != "Cf")

        def decode(match):
            token = match.group(0)
            body = token.rstrip("=")
            if len(body) < 8 or len(body) % 4 == 1 or (body != token and len(token) % 4):
                return token
            try:
                raw = base64.b64decode(body + "=" * (-len(body) % 4), validate=True)
            except (ValueError, binascii.Error):
                return token
            letters = sum(b == 32 or 65 <= b <= 90 or 97 <= b <= 122 for b in raw)
            if b" " not in raw or any(b < 32 or b > 126 for b in raw) or letters * 10 < len(raw) * 7:
                return token
            return raw.decode("ascii")

        text = re.sub(r"[A-Za-z0-9+/]+={0,2}", decode, text)
        return self.links(self.step7(text))


class Bundle:
    def __init__(self, folder, threads=6, allow=(), graphs=True):
        self.dir = Path(folder)
        self.cfg = json.loads((self.dir / "bundle.json").read_text(encoding="utf-8"))
        if self.cfg.get("coverage", {}).get("rule") != "complete_target_v1":
            raise ValueError("bundle must require complete current targets")
        if self.cfg.get("model_text", {}).get("rule") != "v3_meaning_v1":
            raise ValueError("bundle must declare the model text normalization rule")
        if self.cfg.get("scoring", {}).get("mode") != "c_only":
            raise ValueError("bundle must declare scoring.mode=c_only")
        if set(self.cfg.get("graphs", {})) != {"c"} or set(self.cfg.get("tokenizers", {})) != {"c"}:
            raise ValueError("C-only bundles require exactly graph c and tokenizer c")
        rules = self.cfg.get("rules", {})
        paths = ["context.rule", "c.roles", "c.line_string", "c.input_ids", "channels", "text.rule",
                 "text.whitespace", "text.patterns.section", "text.patterns.codes", "text.patterns.url",
                 "text.patterns.ipv4", "text.patterns.ipv6", "text.patterns.not_secure", "text.patterns.icons",
                 "text.patterns.discord_tags", "text.shown", "text.discord_name", "text.names", "text.repeat",
                 "links.pattern", "links.rule", "coverage.rule", "model_text.rule", "scoring.mode"]
        values = []
        for path in paths:
            value = self.cfg
            for key in path.split("."):
                value = value[key]
            values.append(" ".join(str(x) for x in value) if isinstance(value, list) else value)
        if rules.get("of") != paths or rules.get("sha256") != hashlib.sha256("\x00".join(values).encode()).hexdigest():
            raise ValueError("bundle has invalid C-only input rules")
        kinds = ["ask_contact", "give_contact", "ask_personal", "give_personal", "off_server", "sexual", "romantic"]
        if self.cfg.get("outputs") != kinds + ["hold"] or self.cfg.get("kinds") != kinds:
            raise ValueError("bundle has invalid output order")
        entries = list(self.cfg["tokenizers"].values())
        if graphs:
            entries += [{"file": g[k], "sha256": g[k + "_sha256"]}
                        for g in self.cfg["graphs"].values() for k in ("xml", "bin")]
        for entry in entries:
            path = self.dir / entry["file"]
            if Path(entry["file"]).is_absolute() or ".." in Path(entry["file"]).parts or \
                    not path.resolve().is_relative_to(self.dir.resolve()):
                raise ValueError("bundle file escapes its folder")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != entry["sha256"]:
                raise ValueError(f"bundle file hash differs: {entry['file']}")
        self.text = Text(self.cfg, allow)
        self.channels = set(self.cfg["channels"])
        self.req = {}
        if graphs:
            core = ov.Core()
            oc = dict(self.cfg["openvino"]["compile"], INFERENCE_NUM_THREADS=threads)
            self.req = {k: core.compile_model(self.dir / g["xml"], "CPU", oc).create_infer_request()
                        for k, g in self.cfg["graphs"].items()}
        self.tok = {}
        for k in ("c",):
            tk = Tokenizer.from_file(str(self.dir / self.cfg["tokenizers"][k]["file"]))
            tk.no_padding()
            tk.no_truncation()
            self.tok[k] = tk
        c, th = self.cfg["context"], self.cfg["thresholds"]
        self.n_ctx, self.secs = c["lines"], c["seconds"]
        self.public, self.placed = set(c["public_channels"]), set(c["placed_channels"])
        self.private = c["private_channel"]
        self.c_len = self.cfg["c"]["max_tokens"]
        self.hold, self.default = th["hold"], th["default"]
        outs = self.cfg["outputs"]
        self.kinds = [(k, outs.index(k), th["kinds"][k]) for k in self.cfg["kinds"] if k in th.get("kinds", {})]
        sp = self.cfg["special_tokens"]
        self.BOS, self.EOS = sp["bos"], sp["eos"]
        self.HOLD = self.cfg["outputs"].index("hold")

    def run(self, name, **feeds):
        r = self.req[name]
        r.infer(feeds)
        return r.get_output_tensor(0).data.copy()


class Line:
    __slots__ = ("rid", "time", "channel", "author", "partner", "text")


class Stream:
    """One conversation space: the earlier lines that can still be context, and the last copy of each repeat key."""

    def __init__(self):
        self.lines = []
        self.public = []
        self.pairs = defaultdict(list)
        self.placed = defaultdict(list)
        self.copies = {}


def prepared(t, rec):
    """What the server makes of a wire record before scoring: None for an empty text, else (prepared text, the text
    the models read, author key, partner key, repeat key)."""
    ch, account = rec["channel"], rec.get("speaker") or ""
    pre, text = t.text(rec.get("text") or "", ch)
    if not pre:
        return None
    target = rec.get("target")
    if ch == "discord":
        name_pre, name = t.discord_name(account)
        author, key_name = "discord:" + name.lower(), name_pre
    elif ch == "chat":
        s, _ = t.shown(rec.get("shown"), account)
        author, key_name = account.lower(), s if account.lower() in s.lower() else f"{s}={account}"
    else:
        author, key_name = account.lower(), account
    if ch in ("msg", "mail"):
        key_ch = f"{ch}>{target if target is not None else 'unknown'}"
    elif ch == "nick" and target:
        key_ch = f"nick>{target}"
    else:
        key_ch = ch
    partner = (target or "").lower() if ch == "msg" else ""
    return text, t.read(text), author, partner, (key_name.lower(), key_ch, pre)


class Scorer:
    def __init__(self, bundle, all_c=False, ids=False, inputs_only=False):
        self.b = bundle
        self.all_c = all_c
        self.ids = ids
        self.inputs_only = inputs_only
        self.streams = defaultdict(Stream)
        self.cstr = {}
        self.ms = defaultdict(list)

    # --- inputs -------------------------------------------------------------------------------------------------
    def history(self, st, ln):
        if ln.channel == self.b.private:
            return st.pairs[frozenset((ln.author, ln.partner))]
        if ln.channel in self.b.placed:
            return st.placed[ln.author]
        return st.public

    def c_strings(self, seq):
        names = {seq[-1].author: "t"}

        def name(a):
            if a not in names:
                names[a] = f"s{len(names)}"
            return names[a]

        out = []
        for ln in seq:
            w = name(ln.author)
            if ln.channel == self.b.private:
                tag = f"[msg to {name(ln.partner)}] "
            else:
                tag = "" if ln.channel == "chat" else f"[{ln.channel}] "
            out.append(f"{tag}{w}: {ln.text}")
        return out

    def c_ids(self, strings):
        if len(self.cstr) > 200000:
            self.cstr = {}
        enc = []
        for s in strings:
            if s not in self.cstr:
                self.cstr[s] = self.b.tok["c"].encode(s, add_special_tokens=False).ids
            enc.append(self.cstr[s])
        body = []
        for e in enc[:-1]:
            body += e + [self.b.EOS]
        body += [self.b.EOS] + enc[-1] + [self.b.EOS]
        return [self.b.BOS] + body[-(self.b.c_len - 1):]

    # --- one line -----------------------------------------------------------------------------------------------
    def score(self, rec, debug=None):
        b = self.b
        st = self.streams[rec["stream"]]
        out = {"stream": rec["stream"], "id": rec["id"]}
        if rec["channel"] not in b.channels:
            return {**out, "refused": "unknown channel", "verdict": "hold"}
        p = prepared(b.text, rec)
        if p is None:
            return {**out, "dropped": True, "verdict": "deliver"}
        text, read, author, partner, key = p
        t = int(rec["time"])
        if self.ids:
            out.update(prepared=text, text=read, author=author, repeat_key=list(key))
        ln = Line()
        ln.rid, ln.time, ln.channel, ln.author, ln.partner, ln.text = rec["id"], t, rec["channel"], author, partner, read
        hist = self.history(st, ln)
        ctx = [j for j in hist[-b.n_ctx:] if ln.time - st.lines[j].time <= b.secs]
        context_only = rec.get("trusted") or rec.get("context") or rec["channel"] == "discord"
        seq = [st.lines[j] for j in ctx] + [ln]
        if not context_only:
            target = self.c_strings(seq)[-1]
            target_ids = b.tok["c"].encode(target, add_special_tokens=False).ids
            if b.text.shortened(rec.get("text") or "", rec["channel"]) or len(target_ids) + 3 > b.c_len:
                return {**out, "coverage": "reject", "verdict": "reject"}
        last = st.copies.get(key)
        if last is not None and t - last[1] <= b.text.fold_seconds:
            st.copies[key] = (last[0], t)
            return {**out, "folded_into": last[0]}
        st.copies[key] = (rec["id"], t)
        if context_only:
            out["context_only"] = True
        else:
            out["coverage"] = "c"
            strings = self.c_strings(seq)
            cids = self.c_ids(strings)
            ci = np.asarray([cids], np.int64)
            t1 = time.perf_counter()
            zc = None if self.inputs_only else b.run("c", input_ids=ci, attention_mask=np.ones_like(ci))[0]
            if zc is not None:
                if not np.isfinite(zc).all() or len(zc) != 8:
                    raise ValueError("C returned invalid logits")
                self.ms["c"].append(1000 * (time.perf_counter() - t1))
                held = {k: bool(zc[b.HOLD] >= v) for k, v in b.hold.items()}
                out.update(c=[float(x) for x in zc], passed=True,
                           **{"held_" + k: v for k, v in held.items()},
                           verdict="hold" if held[b.default] else "deliver",
                           kinds=[k for k, i, v in b.kinds if zc[i] >= v])
            else:
                out.update(inputs_only=True, verdict="unscored", c=None, passed=None, kinds=[])
            if self.ids:
                out["context_ids"] = [st.lines[j].rid for j in ctx]
                out["c_input_ids"] = cids
            if debug is not None:
                debug[rec["id"]] = {"context_idx": ctx, "strings": strings, "c_ids": cids, "c": out["c"]}
        # remember the line for later lines' context
        st.lines.append(ln)
        i = len(st.lines) - 1
        if ln.channel == b.private or ln.channel in b.placed:
            hist.append(i)
        elif ln.channel in b.public:
            st.public.append(i)
        for lst in (hist, st.public):   # only the last `lines` entries of a list are ever read
            if len(lst) > 4 * b.n_ctx:
                del lst[:-b.n_ctx]
        return out


def score_file(bundle, inp, out, all_c=False, ids=False, progress=None):
    """Scores IN.jsonl into OUT.jsonl; returns (lines, seconds, timing summary)."""
    sc = Scorer(bundle, all_c, ids)
    t0 = time.perf_counter()
    n = 0
    with open(inp, encoding="utf-8") as fi, open(out, "w", encoding="utf-8", newline="\n") as fo:
        for raw in fi:
            fo.write(json.dumps(sc.score(json.loads(raw)), ensure_ascii=False) + "\n")
            n += 1
            if progress:
                progress()
    timing = "; ".join(f"{k}: {len(v)} calls, median {np.median(v):.1f} ms, p99 {np.percentile(v, 99):.1f}"
                       for k, v in sc.ms.items())
    return n, time.perf_counter() - t0, timing


def prep_record(t, rec):
    """A prep.jsonl record with this scorer's outputs for its raw input."""
    w = rec["what"]
    if w == "text":
        pre, text = t.text(rec["raw"], rec["channel"])
        return {**rec, "pre7": pre, "prepared": text, "linked": t.read(text)}
    if w == "shown":
        pre, name = t.shown(rec["raw"], rec["account"])
        return {**rec, "pre7": pre, "prepared": name,
                "repeat_name": (pre if rec["account"].lower() in pre.lower() else f"{pre}={rec['account']}").lower()}
    if w == "discord_name":
        pre, name = t.discord_name(rec["raw"])
        return {**rec, "pre7": pre, "prepared": name, "author": "discord:" + name.lower()}
    raise ValueError(f"unknown prep record {w!r}")


def prep_file(bundle, inp, out):
    """--prep: (records, records whose outputs equal the expected ones in IN)."""
    n = same = 0
    with open(inp, encoding="utf-8") as fi, open(out, "w", encoding="utf-8", newline="\n") as fo:
        for raw in fi:
            rec = json.loads(raw)
            got = prep_record(bundle.text, {k: v for k, v in rec.items() if k in ("what", "channel", "raw", "account",
                                                                                    "note")})
            fo.write(json.dumps(got, ensure_ascii=False) + "\n")
            n += 1
            same += all(got[k] == rec[k] for k in got if k in rec)
    return n, same


def check_example(b):
    """Rebuilds bundle.json's worked example: (ok, the largest logit difference against torch fp32)."""
    ex = b.cfg["example"]
    sc = Scorer(b, ids=True)
    dbg = {}
    ids = {}
    worst = 0.0
    ok = True
    for rec in ex["input"]:
        rec = dict(rec, stream="example")
        r = sc.score(rec, debug=dbg)
        ok &= ("folded_into" in r) == (rec["id"] in ex["folded"]) and r.get("dropped", False) == (rec["id"] in
                                                                                                ex["dropped"])
        rejected = r.get("verdict") == "reject"
        ok &= rejected == (rec["id"] in ex["rejected"])
        if not rejected and not {"folded_into", "dropped", "refused"} & set(r):
            ids[len(sc.streams["example"].lines) - 1] = rec["id"]
            ok &= r["text"] == ex["prepared"][rec["id"]]
    for s in ex["scored"]:
        d = dbg[s["id"]]
        ctx = [ids[j] for j in d["context_idx"]]
        ok &= ctx == s["context"]
        ok &= d["strings"][:-1] == s["c"]["context_strings"] and d["strings"][-1] == s["c"]["target_string"]
        ok &= d["c_ids"] == s["c"]["input_ids"]
        worst = max(worst, float(np.abs(np.array(d["c"]) - s["c"]["logits_torch_fp32"]).max()))
    return ok and worst < 1e-3, ok, worst


def read_allow(path):
    if not path:
        return ()
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return tuple(x for x in (ln.split("#", 1)[0].strip() for ln in lines) if x)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundle")
    ap.add_argument("inp", nargs="?")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--all-c", action="store_true")
    ap.add_argument("--ids", action="store_true", help="also write each line's C token ids and context")
    ap.add_argument("--allow", help="a file of allowed links, replaced by links.canonical in what the models read")
    ap.add_argument("--check-example", action="store_true")
    ap.add_argument("--prep", action="store_true", help="IN and OUT are prep.jsonl records")
    a = ap.parse_args()
    if a.prep:
        n, same = prep_file(Bundle(a.bundle, allow=read_allow(a.allow), graphs=False), a.inp, a.out)
        print(f"{n} prep records, {same} equal to the expected outputs", file=sys.stderr, flush=True)
        return 0 if n == same else 1
    b = Bundle(a.bundle, a.threads, read_allow(a.allow))
    if a.check_example:
        passed, ok, worst = check_example(b)
        print(f"example: features, strings and token ids {'equal' if ok else 'DIFFER'}; logits max |diff| against "
              f"torch fp32 {worst:.2e}")
        return 0 if passed else 1
    n, secs, timing = score_file(b, a.inp, a.out, a.all_c, a.ids)
    print(f"{n} lines in {secs:.0f} s; {timing}", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
