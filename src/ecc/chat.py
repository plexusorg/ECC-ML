"""Chat formats shared by the steps: risk terms, labeling windows, case files, the classifier prompt."""
import base64
import hashlib
import inspect
import json
import re
import tomllib
import unicodedata
from pathlib import Path

from .core import read_jsonl


def terms(cfg):
    """The private term lists in data/labeling/terms.toml, compiled case-insensitively: "risk" (a hit opens a labeling
    window), "sexual" (a first-pass ban on a line with none of these is an over-ban suspect) and "miss" (lures and
    child-sexual phrasing; such a line left ok is a miss suspect)."""
    raw = tomllib.loads((cfg.data / "labeling" / "terms.toml").read_text(encoding="utf-8"))
    return {name: re.compile(raw[name], re.I) for name in ("risk", "sexual", "miss")}

CHANNEL_NAMES = {"chat": "public chat", "msg": "a private message", "guild": "guild chat", "nick": "a nickname",
                 "sign": "a sign", "book": "a book", "item": "an item name"}


def render_window(w):
    head = f"=== WINDOW {w['id']} | {w['source']} | {'private messages' if w['channel'] == 'msg' else 'public chat'}"
    return "\n".join([head] + [f"[{row}] {speaker}: {text}" for row, speaker, text in w["lines"]])


def load_windows(path):
    return {w["id"]: w for w in read_jsonl(path)}


def files_in(paths, suffix):
    """Each path, or for a folder its files with the suffix, in name order. Case sets are folders: their case files
    (.txt) and expected roles (.json) side by side."""
    out = []
    for p in map(Path, paths):
        out += sorted(p.glob(f"*{suffix}")) if p.is_dir() else [p]
    return out


def load_cases(paths):
    """Case files (or folders of them): "## <id> [<channel>]" starts a conversation; lines are label|speaker|text; #
    lines are comments. A file without ## headers is one conversation named after the file."""
    items = []
    for path in files_in(paths, ".txt"):
        conv_id, channel, history = Path(path).stem, "chat", []
        for raw in Path(path).read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or (line.startswith("#") and not line.startswith("##")):
                continue
            if line.startswith("##"):
                m = re.match(r"##\s*(\S+)\s*\[(\w+)\]", line)
                conv_id, channel, history = m.group(1), m.group(2), []
                continue
            label, speaker, text = line.split("|", 2)
            items.append({"conv": conv_id, "channel": channel, "speaker": speaker, "text": text,
                          "label": label, "history": list(history)})
            history.append((speaker, text))
    return items


def load_roles(paths):
    """The expected roles of case sets: the .json files of each folder (or the files given)."""
    roles = {}
    for p in files_in(paths, ".json"):
        roles.update({k: v for k, v in json.loads(Path(p).read_text(encoding="utf-8")).items() if k != "_comment"})
    return roles


# The fine-tuned classifier's prompt: the header, then one line per message, "speaker: text\n", the target last. The
# answer token (" ban" or " ok") follows the target's newline. A conversation's prompt only grows by whole lines, so
# each prompt is a token prefix of the next and llama-server decodes only the new line. Training, eval and the model
# host (which builds each conversation's prompt and calls llama-server) must build exactly this text.
CLASSIFIER_HEADER = ("Minecraft chat moderation. Ban the target message if it is sexual talk aimed at people, explicit "
                     "content, grooming, or anything sexual involving minors. Otherwise ok.\n")
# A conversation's prompt holds 15 to 29 earlier lines plus the target (fewer at its start): its last WINDOW_MIN to
# WINDOW_MAX - 1 earlier lines. When a new line would make it hold WINDOW_MAX earlier lines, or the prompt would pass
# WINDOW_TOKENS, it is rebuilt from the last WINDOW_MIN earlier lines, dropping the oldest of those while it would
# still pass WINDOW_TOKENS. WINDOW_TOKENS is also the cap on every prompt in training and scoring: with the
# deployment's `-np 4 -c 4096` a llama-server slot holds 1,024 tokens.
WINDOW_MIN = 15
WINDOW_MAX = 30
WINDOW_TOKENS = 768
# Every character that can end a line: a player can type all of these except \r and \n (Paper rejects only §, codes
# below space and DEL), and the model reads any of them as a new line, which would let a message forge the next line.
LINE_BREAKS = re.compile("\r\n|[\n\v\f\r\x1c\x1d\x1e\x85\\u2028\\u2029]")
# Right-to-left override, embedding and isolate open a run that is shown reversed, until its pop (or the end)
RTL_RUN = re.compile("[\\u202b\\u202e\\u2067]([^\\u202c\\u2069]*)[\\u202c\\u2069]?")


def clean(s):
    """Speaker names and texts as the prompt shows them, in the order and letters a player sees: NFKC (fullwidth and
    styled letters fold to plain ones), every line break becomes one space, a right-to-left override run is put in
    display order, and invisible format characters (zero-width, bidi controls, soft hyphen) are removed."""
    s = LINE_BREAKS.sub(" ", unicodedata.normalize("NFKC", s))
    s = RTL_RUN.sub(lambda m: m.group(1)[::-1], s)
    return "".join(c for c in s if unicodedata.category(c) != "Cf")


# A run of base64 characters with optional padding. ASCII only, so the model host's Go matches it exactly.
BASE64 = re.compile(r"[A-Za-z0-9+/]+={0,2}")
BASE64_MIN = 8


def decode_base64(s):
    """Base64 a player typed, decoded in place, so the classifier reads "i want to fuck you decode it" rather than its
    encoding. A run of at least BASE64_MIN characters is replaced only when it decodes to printable ASCII words (a
    space, and at least 70% letters or spaces). Ordinary words that happen to be valid base64 ("kerosene", "akshally")
    decode to one garbled word and stay as typed, so a decode never hides what a player wrote."""
    def decoded(m):
        token = m.group(0)
        body = token.rstrip("=")
        if len(body) < BASE64_MIN or len(body) % 4 == 1 or (body != token and len(token) % 4):
            return token
        raw = base64.b64decode(body + "=" * (-len(body) % 4))
        if 0x20 not in raw or any(b < 0x20 or b > 0x7E for b in raw):
            return token
        if sum(0x41 <= b <= 0x5A or 0x61 <= b <= 0x7A or b == 0x20 for b in raw) * 10 < len(raw) * 7:
            return token
        return raw.decode("ascii")
    return BASE64.sub(decoded, s)


def prompt_line(speaker, text):
    return f"{clean(speaker)}: {decode_base64(clean(text))}\n"


def classifier_prompt(lines):
    """lines: (speaker, text) pairs, oldest first; the last one is the target."""
    return CLASSIFIER_HEADER + "".join(prompt_line(s, t) for s, t in lines)


def fitted_prompt(history, speaker, text, count_tokens):
    """classifier_prompt of history plus the target. The oldest lines are dropped until the prompt is at most
    WINDOW_TOKENS tokens; a target too long on its own is cut from the end."""
    h = list(history)
    p = classifier_prompt(h + [(speaker, text)])
    while count_tokens(p) > WINDOW_TOKENS:
        if h:
            h = h[1:]
        else:
            text = text[:int(len(text) * 0.9)]
        p = classifier_prompt(h + [(speaker, text)])
    return p


def window_sizes(lines, count_tokens):
    """The number of earlier lines in each line's prompt when a conversation's lines (speaker, text), in order, are
    scored as the model host scores them (see WINDOW_MIN)."""
    header = count_tokens(CLASSIFIER_HEADER)
    cost = [count_tokens(prompt_line(s, t)) for s, t in lines]
    sizes, n, used = [], 0, header
    for i, c in enumerate(cost):
        if n == WINDOW_MAX or used + c > WINDOW_TOKENS:
            n = min(n, WINDOW_MIN)
            used = header + sum(cost[i - n:i])
            while n and used + c > WINDOW_TOKENS:
                used -= cost[i - n]
                n -= 1
        sizes.append(n)
        n += 1
        used += c
    return sizes


# What a classifier prompt depends on: the header, the window bounds and the code that builds it. Subsets and scores
# record it and are redone when any of it changes.
PROMPT_FORMAT = {"header": CLASSIFIER_HEADER, "window": [WINDOW_MIN, WINDOW_MAX, WINDOW_TOKENS],
                 "code": hashlib.sha256("".join(
                     [LINE_BREAKS.pattern, RTL_RUN.pattern, BASE64.pattern, str(BASE64_MIN)] +
                     [inspect.getsource(f) for f in (clean, decode_base64, prompt_line, classifier_prompt,
                                                     fitted_prompt, window_sizes)]).encode()).hexdigest()}


def split_context(context):
    """train/val parquet keep history as 'speaker: text' lines; a line without ': ' continues the previous text."""
    out = []
    for line in context.split("\n") if context else []:
        s, sep, t = line.partition(": ")
        if sep or not out:
            out.append((s, t))
        else:
            out[-1] = (out[-1][0], out[-1][1] + "\n" + line)
    return out
