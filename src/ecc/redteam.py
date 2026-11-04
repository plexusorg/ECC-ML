"""uv run redteam [--set NAME]: the slow check as ECC-Server runs it (Kimi K3 on Bedrock, verified by code, with
the fine-tuned fast-check classifier) against the red-team and regression sets.

Kimi reports evidence; code decides each player's outcome. Every Kimi call is cached in data/redteam/<tag>.calls.jsonl,
so a rerun sends only missing or failed calls; Bedrock is not contacted when every call is cached. Decisions go to
<tag>.decisions.jsonl and the counts to <tag>.report.md. A set with `samples = 2` runs twice with separate caches
(<set>, <set>-run2): two independent Kimi samples.

Fast check: P(ban) of every case message from `uv run eval --model <[redteam] fast_model>` (its case-set scores,
Platt-calibrated when the model has Platt parameters), paired with the case files by conversation, text and
occurrence; every scorable message of the sets being run must have a score. A player triggers at their first message
with P(ban) >= [redteam] trigger. A conversation where nobody triggers gets no Kimi call, and each of its players is
not involved (cause "no trigger").
Kimi calls in a conversation where someone triggers (player text is datamarked: whitespace inside a message becomes
^, and each line is a JSON object whose speaker comes from the server's record): per view (the whole conversation,
and the conversation cut at each trigger), per player, verbatim quotes of their violating messages with a tag, and
their stated age with the message stating it.
Code decides:
  quotes    kept only when they match one of that player's own lines in that view, else dropped.
  age       a minor only when the age message is one of the player's own lines before the trigger, holds the number,
            reads as about themselves, and no other validated pre-trigger age is 18 or more.
  triggered offender: any kept quote in the whole or cut view, unless at_risk. at_risk: a minor whose kept quotes are
            all responding/self and who is not the first violator in any view. Cleared (not involved) only with no
            kept quote and every message of theirs before the trigger below [redteam] clearing; otherwise held (not
            sure yet). A call that still fails after 3 retries holds.
  not triggered  offender only with corroboration: a kept quote tagged starting, directed, pushing or grooming in the
            whole view, and P(ban) >= [redteam] corroboration on at least one of their kept quoted messages; else not
            involved.
Expected roles come from the set's roles files; a player not listed is not involved.
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import re
import threading
import time
from collections import Counter

from .chat import load_cases, load_roles
from .core import JsonlAppender, log, progress, read_jsonl, run, save_state, write_text
from .evaluate import case_items, case_keys, scorable, score, set_scores

MARK = "^"
RETRIES = 3
OFFENDING = {"starting", "directed", "pushing", "grooming"}
CHANNELS = {"chat": "public chat", "msg": "private messages", "sign": "signs placed in the world",
            "book": "a written book"}

def evidence_system(cfg):
    """The slow check's system prompt: data/prompts/slow-check.txt with data/policy.md (below its title) in place of
    {policy}. Both are private; the classifier bundle carries the result to the model host, which makes the call."""
    policy = (cfg.data / "policy.md").read_text(encoding="utf-8").split("\n", 1)[1].strip()
    return (cfg.data / "prompts" / "slow-check.txt").read_text(encoding="utf-8").replace("{policy}", policy)


def parse_answer(text):
    """The last top-level JSON object in the text."""
    text = re.sub(r"```(?:json)?", "", text or "")
    found, depth, start, in_str, esc = None, 0, None, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0:
                try:
                    found = json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    pass
    return found


def age_statement(t):
    words = t.split()
    return len(words) <= 3 or bool({"i", "im", "i'm", "me", "my"} & set(words))


def nrm(s):
    return " ".join(re.findall(r"[^\W_]+", str(s).replace(MARK, " ").casefold()))


def render(lines):
    return "\n".join(json.dumps({"line": i + 1, "speaker": s, "text": re.sub(r"\s+", MARK, t.strip())},
                                ensure_ascii=False) for i, (s, t, _) in enumerate(lines))


def match_line(quote, line_no, own):
    """Index of the player's own line that the quote copies, or None. own: [(index, text)].
    Runs of one repeated word count as one, since a model cannot copy "lol lol lol ..." spam to the exact count."""
    def squash(s):
        return re.sub(r"\b(\w+)(?: \1\b)+", r"\1", nrm(s))
    q = squash(quote)
    if not q:
        return None
    hits = [i for i, t in own if q == squash(t) or (len(q) >= 4 and q in squash(t))]
    if not hits:
        return None
    if isinstance(line_no, int) and line_no - 1 in hits:
        return line_no - 1
    return hits[0]


class Kimi:
    """Bedrock Converse calls with a per-tag cache; the client is created only when a call is not cached."""

    def __init__(self, cfg, cache_path, cached_only=False):
        self.cached_only = cached_only
        self.cfg, self.model, self.cache_path = cfg, cfg["redteam"]["model"], cache_path
        self.cache = {r["key"]: r for r in read_jsonl(cache_path, repair=True) if r["ok"]}
        self.client, self.sent = None, 0
        self.lock = threading.Lock()
        self.out = JsonlAppender(cache_path)

    def _client(self):
        with self.lock:
            if self.client is None:
                import boto3
                from botocore.config import Config as BotoConfig
                self.cfg.load_env()
                region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
                self.client = boto3.client("bedrock-runtime", region_name=region,
                                           config=BotoConfig(read_timeout=300,
                                                             retries={"max_attempts": 1, "mode": "standard"}))
            return self.client

    def ask(self, kind, system, user, check):
        key = hashlib.sha256(json.dumps([self.model, system, user]).encode()).hexdigest()
        if key in self.cache:
            return self.cache[key]
        if self.cached_only:
            raise SystemExit(f"a {kind} call is not cached and --cached was given; nothing was sent to Bedrock")
        client = self._client()
        attempts, tokens_in, tokens_out, errors, answer, raw = 0, 0, 0, [], None, ""
        for attempt in range(RETRIES + 1):
            attempts += 1
            if attempt:
                time.sleep(2 ** attempt)
            try:
                resp = client.converse(modelId=self.model, system=[{"text": system}],
                                       messages=[{"role": "user", "content": [{"text": user}]}],
                                       inferenceConfig={"maxTokens": 8000})
            except Exception as e:  # any API error is retried, then the ban stays
                errors.append(f"api: {type(e).__name__}: {str(e)[:200]}")
                continue
            usage = resp.get("usage", {})
            tokens_in += (usage.get("inputTokens", 0) + usage.get("cacheReadInputTokens", 0)
                          + usage.get("cacheWriteInputTokens", 0))
            tokens_out += usage.get("outputTokens", 0)
            stop = resp.get("stopReason")
            raw = "".join(b.get("text", "") for b in resp["output"]["message"]["content"])
            if stop in ("content_filtered", "guardrail_intervened") or not raw.strip():
                errors.append(f"refused: {stop}")
                continue
            if stop == "max_tokens":
                errors.append("token cap")
                continue
            try:
                answer = check(parse_answer(raw))
                break
            except (ValueError, TypeError, KeyError, AttributeError) as e:
                errors.append(f"unparseable: {e}")
                answer = None
        rec = {"key": key, "kind": kind, "ok": answer is not None, "answer": answer, "raw": raw,
               "attempts": attempts, "errors": errors, "tokens_in": tokens_in, "tokens_out": tokens_out}
        with self.lock:
            self.out.write(rec)
            self.sent += 1
            if rec["ok"]:
                self.cache[key] = rec
        return rec


def keyed(obj, players):
    if not isinstance(obj, dict):
        raise ValueError("no JSON object")
    if isinstance(obj.get("players"), dict) and not any(p in obj for p in players):
        obj = obj["players"]
    lower = {k.casefold(): v for k, v in obj.items() if isinstance(k, str)}
    out = {}
    for p in players:
        if p.casefold() not in lower:
            raise ValueError(f"missing player {p!r}")
        out[p] = lower[p.casefold()]
    return out


def evidence_check(players):
    def check(obj):
        out = {}
        for p, v in keyed(obj, players).items():
            if not isinstance(v, dict) or not isinstance(v.get("violations", []), list):
                raise ValueError(f"bad entry for {p!r}")
            viol = []
            for q in v.get("violations") or []:
                if isinstance(q, str):
                    q = {"quote": q}
                if not isinstance(q, dict) or not isinstance(q.get("quote"), str):
                    raise ValueError("bad violation")
                line = q.get("line")
                viol.append({"quote": q["quote"], "line": int(line) if str(line).isdigit() else None,
                             "tag": str(q.get("tag", "")).strip().lower()})
            age = v.get("own_age")
            age = int(age) if isinstance(age, (int, float)) or (isinstance(age, str) and age.strip().isdigit()) else None
            msg = v.get("age_message")
            out[p] = {"own_age": age, "age_message": msg if isinstance(msg, str) else None, "violations": viol}
        return out
    return check


def conversations(case_files):
    convs = {}
    for it in load_cases(case_files):
        c = convs.setdefault(it["conv"], {"id": it["conv"], "channel": it["channel"], "lines": []})
        c["lines"].append((it["speaker"], it["text"], it["label"]))
    for c in convs.values():
        c["keys"] = case_keys([(c["id"], t) for _, t, _ in c["lines"]])
    return list(convs.values())


def fast_scores(cfg, model, sets):
    """evaluate.case_key -> the fast-check model's P(ban) (None for an error row, which has no margin), from its eval
    case sets that are current and fully scored. Refuses when a scorable message of `sets` (the [redteam.sets] being
    run) has no score."""
    spec = cfg["models"][model]
    if "gguf" not in spec:
        raise SystemExit(f"[models.{model}] has no gguf; run `uv run export` and `uv run eval --model {model}` first")
    platt = spec.get("platt")
    out = {}
    for s in cfg["eval"]["sets"]:
        current = set_scores(cfg, model, s)
        if current is None or current[1]:
            log.warning("%s's scores on eval set %s are %s; they are not used", model, s,
                        f"incomplete ({current[1]:,} unscored)" if current else "missing or from other inputs")
            continue
        for r in current[0]:
            out[r["k"]] = None if r.get("margin") is None else score(r, platt)
    for name, s in sets.items():
        missing = sum(k not in out for k, _ in scorable(case_items(cfg, s["cases"], None)))
        if missing:
            files = {cfg.path(f) for f in s["cases"]}
            holders = [e for e, fs in cfg["eval"]["sets"].items() if files & {cfg.path(f) for f in fs}]
            raise SystemExit(f"redteam set {name}: {missing:,} messages have no {model} score; " + (
                f"run `uv run eval --model {model} --set {' '.join(holders)}` first" if holders
                else f"add its case files to an [eval.sets] entry and run `uv run eval --model {model}` first"))
    return out


def players(conv, roles, probs, trigger):
    """Per speaker: the expected role and the index of their triggering message (None when they never trigger)."""
    lines = conv["lines"]
    out = {}
    for s in dict.fromkeys(s for s, _, _ in lines):
        hit = next((i for i, (sp, _, _) in enumerate(lines)
                    if sp == s and probs[i] is not None and probs[i] >= trigger), None)
        out[s] = {"expected": roles.get(s, "not_involved"), "trigger": hit}
    return out


def ask_evidence(kimi, system, conv, cut):
    lines = conv["lines"][:cut + 1]
    players = list(dict.fromkeys(s for s, _, _ in lines))
    user = (f"Messages in {CHANNELS[conv['channel']]}, in order:\n{render(lines)}\n\n"
            f"Players to judge (exact speaker names, one JSON string each): {json.dumps(players, ensure_ascii=False)}")
    return kimi.ask("evidence", system, user, evidence_check(players))


def validate(rec, conv, cut, trigger_of):
    """Kept evidence of one evidence call: {player: {"quotes": [(index, tag)], "dropped": n, "ages": [age]}}."""
    lines = conv["lines"][:cut + 1]
    out = {}
    for p, v in rec["answer"].items():
        own = [(i, t) for i, (s, t, _) in enumerate(lines) if s == p]
        kept, dropped = [], 0
        for q in v["violations"]:
            i = match_line(q["quote"], q["line"], own)
            if i is None:
                dropped += 1
            else:
                kept.append((i, q["tag"]))
        ages = []
        age, msg = v["own_age"], v["age_message"]
        trig = trigger_of.get(p, cut)
        if age is not None and msg:
            said = nrm(msg)
            for i, t in own:
                nt = nrm(t)
                if i >= trig or not said or not (said in nt or nt in said):
                    continue
                # the number in the player's own words, or a short confirmation ("yea") of the nearest earlier line
                # by someone else that holds it ("ur 11 right")
                prev = next((nrm(lines[j][1]) for j in range(i - 1, -1, -1) if lines[j][0] != p), "")
                if (str(age) in nt.split() and age_statement(nt)) or (len(nt.split()) <= 3 and str(age) in prev.split()):
                    ages.append(age)
                    break
        out[p] = {"quotes": kept, "dropped": dropped, "given": len(v["violations"]), "ages": ages}
    return out


def first_violator(evidence):
    best = min(((i, p) for p, qs in evidence.items() for i, _ in qs), default=None)
    return best[1] if best else None


def decide(p, info, conv, calls, probs, rc):
    """Outcome for one player; calls is None in a conversation where nobody triggered. Returns (outcome, cause)."""
    if calls is None:
        return "not_involved", "no trigger"
    trig = info["trigger"]
    full = calls["full"]
    if trig is None:
        if full is None:
            return "not_involved", "failure"
        quotes = full.get(p, {}).get("quotes", [])
        if (any(tag in OFFENDING for _, tag in quotes)
                and any(probs[i] is not None and probs[i] >= rc["corroboration"] for i, _ in quotes)):
            return "offender", "corroborated"
        return "not_involved", "not triggered"
    views = [full, calls["cuts"][trig]]
    if any(v is None for v in views):
        return "held", "failure"
    kept = [q for v in views for q in v.get(p, {}).get("quotes", [])]
    if kept:
        tags = {t for _, t in kept}
        firsts = {first_violator({k: e["quotes"] for k, e in v.items()}) for v in views}
        ages = [a for v in views for a in v.get(p, {}).get("ages", [])]
        minor = bool(ages) and all(a < 18 for a in ages)
        if minor and p not in firsts and tags <= {"responding", "self"}:
            return "at_risk", "kimi"
        return "offender", "kimi"
    before = [i for i, (s, _, _) in enumerate(conv["lines"][:trig]) if s == p]
    if any(probs[i] is None or probs[i] >= rc["clearing"] for i in before):
        return "held", "fast"
    return "not_involved", "cleared"


def run_set(cfg, tag, convs, roles, scores, system, cached_only, suffix=""):
    """One sample of a set. tag names its Kimi cache; tag + suffix (the thresholds that differ from ecc.toml) names its
    report and decisions."""
    rc = cfg["redteam"]
    out = cfg.data / "redteam"
    kimi = Kimi(cfg, out / f"{tag}.calls.jsonl", cached_only)
    tag += suffix
    probs = {c["id"]: [scores.get(k) for k in c["keys"]] for c in convs}
    flags = {c["id"]: players(c, roles.get(c["id"], {}), probs[c["id"]], rc["trigger"]) for c in convs}
    recs = {}
    jobs = {}
    # the slow check runs only for triggered players: a conversation where nobody triggers gets no Kimi call
    triggered = {c["id"] for c in convs if any(f["trigger"] is not None for f in flags[c["id"]].values())}
    for c in convs:
        if c["id"] not in triggered:
            continue
        for cut in sorted({len(c["lines"]) - 1} | {f["trigger"] for f in flags[c["id"]].values()} - {None}):
            jobs[(c["id"], cut)] = (kimi, system, c, cut)
    pool = cf.ThreadPoolExecutor(rc["workers"])
    try:
        with progress(f"{tag} evidence", len(jobs), "calls") as advance:
            futures = {pool.submit(ask_evidence, *a): k for k, a in jobs.items()}
            for fut in cf.as_completed(futures):
                recs[futures[fut]] = fut.result()
                advance()
    except KeyboardInterrupt:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown()
    kimi.out.close()

    rows = []
    for c in convs:
        trigger_of = {p: f["trigger"] for p, f in flags[c["id"]].items() if f["trigger"] is not None}
        cuts = {cut: validate(r, c, cut, trigger_of) if r["ok"] else None
                for (cid, cut), r in recs.items() if cid == c["id"]}
        calls = {"full": cuts[len(c["lines"]) - 1], "cuts": cuts} if c["id"] in triggered else None
        for p, f in flags[c["id"]].items():
            outcome, cause = decide(p, f, c, calls, probs[c["id"]], rc)
            rows.append({"conv": c["id"], "player": p, "expected": f["expected"],
                         "trigger": None if f["trigger"] is None else f["trigger"] + 1,
                         "evidence": {str(k + 1): (v or {}).get(p) for k, v in cuts.items()},
                         "outcome": outcome, "cause": cause})
    decisions = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)

    used = list(recs.values())
    n = Counter()
    for r in rows:
        o, e = r["outcome"], r["expected"]
        n["oc"] += e == "offender" and o == "not_involved"
        n["om"] += e == "offender" and r["trigger"] is None and o == "not_involved"
        n["oa"] += e == "offender" and o == "at_risk"
        n["ac"] += e == "at_risk" and o == "not_involved"
        n["ao"] += e == "at_risk" and o in ("offender", "held")
        n["ib"] += e == "not_involved" and o in ("offender", "at_risk")
        n["ih"] += e == "not_involved" and o == "held"
        n["bn"] += e == "not_involved" and o in ("offender", "at_risk") and r["conv"].startswith("bn-")
        n["fail"] += r["cause"] == "failure"
        n["corr"] += r["cause"] == "corroborated"
    failures = n["oc"] + n["oa"] + n["ac"] + n["ib"]
    report = [f"# {tag}: {len(convs)} conversations, {len(rows)} players; fast check {rc['fast_model']}, P(ban) "
              f"trigger {rc['trigger']}, clearing {rc['clearing']}, corroboration {rc['corroboration']}",
              f"Kimi calls {len(used)} (failed after retries {sum(not r['ok'] for r in used)}), "
              f"attempts {sum(r['attempts'] for r in used)}, retries {sum(r['attempts'] - 1 for r in used)}, "
              f"tokens in {sum(r['tokens_in'] for r in used)}, out {sum(r['tokens_out'] for r in used)}",
              "errors: " + "; ".join(f"{e} x{k}" for e, k in Counter(e.split(':')[0] for r in used
                                                                      for e in r["errors"]).items()),
              f"players triggered: {sum(r['trigger'] is not None for r in rows)}; banned by corroboration: {n['corr']}; "
              f"conversations with no trigger (no Kimi call): {len(convs) - len(triggered)}",
              "", "| offender cleared (never triggered) | offender->at_risk | at_risk cleared | at_risk->offender/held "
                  "| innocent banned (benign controls) | innocent held | held by failure | failures |",
              "|" + "---|" * 8,
              f"| {n['oc']} ({n['om']}) | {n['oa']} | {n['ac']} | {n['ao']} | {n['ib']} ({n['bn']}) | {n['ih']} | "
              f"{n['fail']} | {failures} |", "", "## Mismatches"]
    for r in rows:
        if r["outcome"] != r["expected"] and not (r["expected"] == "offender" and r["outcome"] == "held"):
            report.append(f"- [{r['conv']}] {r['player']}: expected {r['expected']}, got {r['outcome']} ({r['cause']})")
    text = "\n".join(report) + "\n"
    same = all((out / f"{tag}.{s}").exists() and (out / f"{tag}.{s}").read_text(encoding="utf-8") == t
               for s, t in (("report.md", text), ("decisions.jsonl", decisions)))
    write_text(out / f"{tag}.decisions.jsonl", decisions)
    write_text(out / f"{tag}.report.md", text)
    log.info("%s: %d conversations, %d players; %d new Kimi calls; failures %d%s", tag, len(convs), len(rows),
             kimi.sent, failures, "; report and decisions unchanged" if same else "")
    return {"decisions": len(rows), "failures": failures, "new_calls": kimi.sent}


def body(cfg, args):
    rc = cfg["redteam"]
    overrides = {k: getattr(args, k) for k in ("trigger", "clearing", "corroboration") if getattr(args, k) is not None}
    rc.update(overrides)
    # a run with other thresholds keeps its own reports; Kimi answers are cached per call, so they are shared
    suffix = "".join(f"-{k[:4]}{v}" for k, v in overrides.items())
    system = evidence_system(cfg)
    chosen = {name: s for name, s in rc["sets"].items() if not args.set or name in args.set}
    scores = fast_scores(cfg, rc["fast_model"], chosen)
    results = {}
    for name, s in chosen.items():
        convs = conversations([cfg.path(f) for f in s["cases"]])
        roles = load_roles([cfg.path(f) for f in s["cases"]])
        for sample in range(1, s.get("samples", 1) + 1):
            tag = name if sample == 1 else f"{name}-run{sample}"
            results[tag + suffix] = run_set(cfg, tag, convs, roles, scores, system, args.cached, suffix)
    inputs = [cfg.data / "policy.md", cfg.data / "prompts" / "slow-check.txt",
              *[cfg.path(f) for s in rc["sets"].values() for f in s["cases"]]]
    save_state(cfg, "redteam", inputs, runs=results)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", nargs="+", help="only these [redteam.sets]")
    ap.add_argument("--cached", action="store_true", help="use cached Kimi answers only; never contact Bedrock")
    for k in ("trigger", "clearing", "corroboration"):
        ap.add_argument(f"--{k}", type=float, help=f"P(ban) {k} threshold for this run in place of [redteam] {k}; "
                                                   "the reports get it in their name")
    run("redteam", body, ap)
