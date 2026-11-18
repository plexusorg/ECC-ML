"""How a chat line becomes the input of the two v4 models. Training, evaluation, export and the bundle all build their
inputs here, so the models see in deployment exactly what they were trained on. ECC-Server builds the same inputs from
the rules the bundle states (bundle.json `context`, `sh`, `c`); `ecc.reference` is a second, plain implementation of
those rules.

A table of lines (pandas) has the columns run (one stream: one server uptime, over one or more log files), id, time
(whole seconds, counting on across midnights within a run), channel, account (the author's account name; empty for a
Discord relay line), speaker (the shown name), partner (a private message's addressee) and text, in arrival order
within each run.

- The context of a line: the earlier lines of its list (a private message: the pair's messages; a placed line: its
  author's placed lines; any other line: the public lines), the last CTX_LINES of them within CTX_SECONDS.
- SH reads every line alone (`line_tokens`) once, as a vector, and a small transformer reads the vectors of the context
  and the line with each one's speaker slot, addressee slot, channel, time gap and position (`sh_features`).
- C reads the context and the line as one string per line with role names (`render`), tokenized and joined
  (`CInputs.build`), cut from the left to C's length, which is data of the C run (its input.json `max_tokens`).

A line's text and names are prepared before any of this (`prepare`, `shown`, `discord_name`): the dossier builds the
training text with these functions, and ECC-Server applies the same rules, which the bundle states (bundle.json
`text`), to the raw text the plugin sends. Before model input construction, `model_text` normalizes meaning,
redacts newly revealed addresses and replaces allowed links. Source records and repeat keys keep the prepared text.
"""
import collections
import math
import re

import numpy as np

KINDS = ["ask_contact", "give_contact", "ask_personal", "give_personal", "off_server", "sexual", "romantic"]
OUTS = KINDS + ["hold"]                 # the eight outputs of both models; only hold decides
HOLD = OUTS.index("hold")
PUBLIC = {"chat", "guild", "me", "discord"}         # appended to the public list
PLACED = {"sign", "book", "item", "tag", "nick"}    # appended to the author's placed list
PRIVATE = "msg"                                     # appended to the pair's list
CHANNELS = ["chat", "guild", "msg", "mail", "me", "nick", "sign", "book", "item", "tag", "muted", "discord"]
BOS, PAD, EOS = 0, 1, 2
CTX_LINES, CTX_SECONDS = 30, 1800
SH_LINE_TOKENS = 48                 # <s> + the first 46 ids of the line string + </s>
SEQ = CTX_LINES + 1                 # SH's sequence: the context and the line
SLOT_MAX = SEQ                      # slot numbers are capped at 31
ADDR_NONE = SEQ + 1                 # the addressee slot of a line that is not a private message
GAPS = 12                           # gap buckets: 0, then bit_length(seconds) capped at 11


# ---------------------------------------------------------------- text preparation

SECTION = re.compile(r"(?s)§.?")                                     # step 1
CODES = re.compile(r"(?i)[§&](?:#[0-9a-f]{6}|[0-9a-fk-orx])")       # step 2
URL = re.compile(r"(?i)https?://\S+")                                # step 5
IPV4 = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")     # step 7
# three or more colons between hex groups, or a "::" form with three or more groups; a time of day has two colons
IPV6 = re.compile(r"(?i)(?<![\w:])(?:[0-9a-f]{1,4}:){3,7}[0-9a-f]{1,4}(?![\w:])|"
                  r"(?<![\w:])(?=(?:[0-9a-f]*:){2,})(?:[0-9a-f]{1,4}:)+:(?:[0-9a-f]{1,4}:?)+(?![\w:])")
NOT_SECURE = re.compile(r"^\[Not Secure\]\s*")                       # the client's mark before a chat name
ICONS = re.compile(r"\[(?:item|block)/[^\]]*\]|\|\|\|")              # icon codes in a chat name
TAGS = re.compile(r"<[^>]*>")                                        # tags in a Discord name
LIMITS = {"book": 1000, "nick": 80}     # step 6: a text over its channel's limit is cut
LIMIT = 400                             # the limit of every other channel
NAME_LIMIT = 60                         # a Discord name's limit
URL_OVER, URL_KEEP = 60, 50             # step 5: a link over URL_OVER code points keeps URL_KEEP
CUT, URL_CUT, IP = " [...]", "[...]", "[ip]"
# a link: group 1 the host, group 2 the path
LINK = re.compile(r"(?i)(?:https?://)?((?:[a-z0-9-]+\.)+[a-z]{2,24})(?::\d+)?(/[^\s]*)?")


def limit(channel):
    return LIMITS.get(channel, LIMIT)


def clean(text, limit=LIMIT):
    """Steps 1-6 of the text preparation: colour codes removed, line breaks shown as ' / ', whitespace collapsed, long
    links and long texts cut."""
    text = CODES.sub("", SECTION.sub("", text)).replace("\r", " ").replace("\n", " / ").strip()
    text = re.sub(r"\s+", " ", text)
    text = URL.sub(lambda u: u.group(0) if len(u.group(0)) <= URL_OVER else u.group(0)[:URL_KEEP] + URL_CUT, text)
    return text if len(text) <= limit else text[:limit] + CUT


def redact(text):
    """Step 7: IPv4 then IPv6 addresses replaced by [ip]. Returns (text, addresses replaced)."""
    text, n4 = IPV4.subn(IP, text)
    text, n6 = IPV6.subn(IP, text)
    return text, n4 + n6


def prepare(text, channel):
    """Prepare a source line: steps 1-7. Model-read normalization is separate."""
    return redact(clean(text, limit(channel)))[0]


def shown(display):
    """A chat name as players saw it, before step 7: colour codes, the client's [Not Secure] mark and icon codes
    removed, whitespace collapsed."""
    return re.sub(r"\s+", " ", ICONS.sub("", NOT_SECURE.sub("", CODES.sub("", SECTION.sub("", display))))).strip()


def discord_name(raw):
    """A Discord relay line's name before step 7: tags removed, then steps 1-6 at NAME_LIMIT."""
    return clean(TAGS.sub("", raw), NAME_LIMIT)


def whitespace():
    """The code points Python's str.isspace() (and the regular expressions' \\s) count as whitespace."""
    return [c for c in range(0x110000) if chr(c).isspace()]


def allowed(host, path, allow):
    """Whether one link (LINK's host and path groups) is on the allowlist: [(host, path prefix)], lowercased."""
    h, p = host.lower(), (path or "").lower()
    return any((h == a or h.endswith("." + a)) and p.startswith(pre) for a, pre in allow)


def link(text, allow, canonical):
    """The text the models read (bundle.json `links.rule`): each link the allowlist allows replaced by `canonical`."""
    return LINK.sub(lambda m: canonical if allowed(m.group(1), m.group(2), allow) else m.group(0), text) if allow \
        else text


def model_text(text, allow, canonical):
    """Read normalized meaning, redact any address it reveals, then replace allowed links. Do not use for repeat keys."""
    from .model_text import normalize
    return link(redact(normalize(text))[0], allow, canonical)


def keys(df):
    """Who wrote each line: the account name lowercased; a Discord relay line (no account) 'discord:' + its shown name
    lowercased."""
    acc = df.account.fillna("").str.lower().to_numpy(object)
    spk = df.speaker.fillna("").str.lower().to_numpy(object)
    return np.where(acc == "", "discord:" + spk, acc)


def contexts(df, n=CTX_LINES, seconds=CTX_SECONDS, accept=None):
    """Per row: indexes of the rows of its context, oldest first. Of the line's list take the last n, then keep those
    within `seconds`. After a line, a private message is appended to its pair's list, a placed line to its author's
    list, a public-channel line to the public list; any other channel (muted, mail, ...) takes public context and is
    appended nowhere. When supplied, accept(row, context) runs before appending the row. A false result keeps it out
    of all later context. A table whose time runs backwards within a run is not in arrival order and is refused."""
    out = [None] * len(df)
    chan, sec = df.channel.to_numpy(), df.time.to_numpy()
    who = keys(df)
    part = df.partner.fillna("").str.lower().to_numpy(object)
    for run, idx in df.groupby("run", sort=False).indices.items():
        back = int((np.diff(sec[idx]) < 0).sum())
        if back:
            raise ValueError(f"run {run}: time runs backwards {back} times; contexts need each run's lines in "
                             "arrival order")
        public, private, placed = [], collections.defaultdict(list), collections.defaultdict(list)
        for i in idx:
            c = chan[i]
            if c == PRIVATE:
                prev = private[frozenset((who[i], part[i]))]
            elif c in PLACED:
                prev = placed[who[i]]
            else:
                prev = public
            out[i] = [j for j in prev[-n:] if sec[i] - sec[j] <= seconds]
            if accept is not None and not accept(i, out[i]):
                continue
            if c == PRIVATE or c in PLACED:
                prev.append(i)
            elif c in PUBLIC:
                public.append(i)
    return out


# ---------------------------------------------------------------- SH

def line_string(chan, text):
    return text if chan == "chat" else f"[{chan}] {text}"


def line_tokens(tok, strings):
    """Each line string alone: <s>, its first SH_LINE_TOKENS - 2 ids, </s>."""
    out = []
    for c in range(0, len(strings), 100000):
        for e in tok.encode_batch(list(strings[c:c + 100000]), add_special_tokens=False):
            out.append(np.asarray([BOS] + e.ids[:SH_LINE_TOKENS - 2] + [EOS], np.int32))
    return out


def gap_bucket(g):
    return 0 if g <= 0 else min(GAPS - 1, 1 + int(math.log2(g)))


def sh_features(df, ctx, rows):
    """Per row of `rows`: the row indexes of its sequence (context, then the line last) and their slot, addressee,
    channel, gap and position features, padded to SEQ (line -1 = padding); n = the sequence's length. df needs account
    (`keys`) and partner (lowercased) columns. Slots: the line's author is 0; the others are numbered in order of first
    appearance walking the sequence, each line's speaker before a private message's addressee."""
    chan, part, acc, sec = df.channel.to_numpy(), df.partner.to_numpy(), df.account.to_numpy(), df.time.to_numpy()
    cid = {c: k for k, c in enumerate(CHANNELS)}
    R = len(rows)
    F = {"line": np.full((R, SEQ), -1, np.int64), "slot": np.zeros((R, SEQ), np.int64),
         "addr": np.full((R, SEQ), ADDR_NONE, np.int64), "chan": np.zeros((R, SEQ), np.int64),
         "gap": np.zeros((R, SEQ), np.int64), "pos": np.zeros((R, SEQ), np.int64), "n": np.zeros(R, np.int64)}
    for r, i in enumerate(rows):
        names = {acc[i]: 0}

        def name(a):
            if a not in names:
                names[a] = len(names)
            return min(names[a], SLOT_MAX)

        seq = list(ctx[i]) + [i]
        n = len(seq)
        F["n"][r] = n
        for k, j in enumerate(seq):
            F["line"][r, k] = j
            F["slot"][r, k] = name(acc[j]) if j != i else 0
            if chan[j] == PRIVATE:
                F["addr"][r, k] = name(part[j])
            F["chan"][r, k] = cid.get(chan[j], 0)
            F["gap"][r, k] = gap_bucket(sec[i] - sec[j])
            F["pos"][r, k] = n - 1 - k
    return F


def sh_table(df):
    """The columns sh_features reads: account = keys(df), partner lowercased, text filled."""
    sh = df.copy()
    sh["account"] = keys(df)
    sh["partner"] = df.partner.fillna("").str.lower()
    sh["text"] = df.text.fillna("")
    return sh


# ---------------------------------------------------------------- C

def render(df, ctx, rows):
    """For each row of `rows`: (context strings, the line's string). The line's author is t, the others s1, s2, ... in
    order of first appearance (each line's speaker before a private message's addressee); a line not in public chat
    carries its channel, a private message as [msg to <role>]."""
    chan, part, txt = df.channel.to_numpy(), df.partner.fillna("").str.lower().to_numpy(object), df.text.to_numpy()
    who = keys(df)
    out = []
    for i in rows:
        names = {who[i]: "t"}

        def name(a):
            if a not in names:
                names[a] = f"s{len(names)}"
            return names[a]

        def fmt(j):
            w = name(who[j])
            tag = f"[msg to {name(part[j])}] " if chan[j] == PRIVATE else ("" if chan[j] == "chat" else f"[{chan[j]}] ")
            return f"{tag}{w}: {txt[j]}"

        out.append(([fmt(j) for j in ctx[i]], fmt(i)))
    return out


class CInputs:
    """<s> ctx1 </s> ... ctxN </s></s> line </s>, cut from the left to max_tokens tokens (the <s> kept)."""

    def __init__(self, tokenizer_path):
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(str(tokenizer_path))
        self.tok.no_padding()
        self.tok.no_truncation()
        self.cache = {}

    def ids(self, strings):
        new = sorted({s for s in strings if s not in self.cache})
        for s, e in zip(new, self.tok.encode_batch(new, add_special_tokens=False)):
            self.cache[s] = e.ids
        return [self.cache[s] for s in strings]

    def build(self, rendered, max_tokens):
        self.ids([t for _, t in rendered] + [c for cs, _ in rendered for c in cs])
        out = []
        for cs, t in rendered:
            body = []
            for c in cs:
                body += self.cache[c] + [EOS]
            body += [EOS] + self.cache[t] + [EOS]
            out.append(np.array([BOS] + body[-(max_tokens - 1):], np.int32))
        self.cache = {}
        return out


def tokenizer(path):
    """A HuggingFace tokenizers file, with no padding and no truncation."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(path))
    tok.no_padding()
    tok.no_truncation()
    return tok
