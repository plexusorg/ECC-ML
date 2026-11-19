"""uv run transcript BATCH: build the parts of a reader batch from the raw chat, and check them.

A transcript is the chat of one server log file (one server uptime, or one day), or of one Creayun server on one day,
as plain text a reader reads top to bottom: one line per event, `<id> <time> <channel> <speaker>: <text>`. Long
transcripts are cut into parts of at most 12,000 tokens. A batch is a folder under [transcript] batches: manifest.json
(its kind, the sources and what was built from them), parts/ (the part files), reads/<reader>/ (the answers,
`uv run read`) and the line-label table (`uv run line-labels`). A part file names its transcript a "Dossier", the word
the readers' questions use.

    uv run transcript BATCH                     rebuild BATCH's parts from its sources and compare them with the saved
                                                parts; write them only when it has none or no read has answered them
    uv run transcript BATCH --files "tf/2026-11-1*.log.gz" ...
                                                a new server-log batch from these log files (globs under [transcript]
                                                logs)
    uv run transcript BATCH --creayun           a new Creayun batch: every server and day of the Creayun message table
    uv run transcript BATCH --check             only check the saved parts
    uv run transcript BATCH --rewrite NAME      write the rebuilt parts over parts reads have answered; the saved parts
                                                and the answers to parts that gain a line or show a line's speaker or
                                                addressee otherwise go to [transcript] history/NAME

A rebuild of a batch that has parts cuts each transcript where its saved parts were cut, so a line the builder now adds
or leaves out changes only the part that holds it.

Every build ends with the check: no IPv4 or IPv6 address, no authentication command as a line's text, no argument a
player typed to an authentication command in that player's own lines, and for Creayun no e-mail address, key, token,
password stated in a sentence or card number. The check prints line ids and lengths, never text.
"""
import argparse
import collections
import copy
import gzip
import json
import re
from datetime import date, timedelta

from . import inputs as I
from .core import log, run, write_text
from .inputs import IPV4, IPV6, clean, shown
from .parse import (ADMIN, BOOT, CHAT, CMD, DATED, EDGE, GUILD, LEAVE, LINE, NICK, SUDO, VANILLA, Players, command,
                    flagged_blocks, fold, plain_name, strings, strip_colour)

TOKEN_BUDGET = 12_000          # per part, header and overlap included
CHARS_PER_TOKEN = 3
OVERLAP = 30                   # lines of the previous part a later part starts with
FOLD_SECONDS = 60              # a speaker's exact repeat on a channel within this many seconds is folded
BOOT_LINES = 200               # a log file whose first lines hold the startup line starts a run

# Authentication: the command name after any plugin prefix (/authme:login is login). Matched whole, so /region and
# /regen stay. Typos seen in the logs are listed; a name glued to its argument (/register1888888) is caught by prefix.
AUTH = re.compile(r"^(?:l|lo|log|logon|logn|loign|lgoin|login\W*|relog|logout|reg|regi|regis|regist\w*|register\w*|"
                  r"regsiter\w*|regiseter\w*|registwer\w*|regester\w*|unreg|unregister|changepassword|changepass|"
                  r"changepw|changepwd|cp|passwd|password|pass|pw|pwd|setpassword|resetpassword|shib\w*|2fa|totp|"
                  r"captcha|verify|email|deauth|auth\w*|premium|unpremium|nlogin|recovery)$", re.I)
# A chat line that is an authentication command typed without its slash or with a stray prefix.
AUTH_CHAT = re.compile(r"^\s*(?:[/\\.]+\s*(?:l|login|log|reg|register|changepassword|cp|shib)\b|"
                       r"(?:login|register|changepassword)\s+\S+(?:\s+\S+)?\s*$)", re.I)
# A text that starts with an authentication command after a slash, backslash or dot ("/reg x", ".password").
AUTH_WORDS = (r"(?:l|log|login|logon|reg|regi|register|unregister|changepassword|changepass|cp|passwd|password|"
              r"setpassword|shib|shibboleth|2fa|totp)\b")
AUTH_START = re.compile(r"(?i)^\s*[/\\.]+\s*" + AUTH_WORDS)
AUTH_MENTION = re.compile(r"(?i)[\s:][/\\]\s*" + AUTH_WORDS)

DISCORD = re.compile(r"^\[DiscordSRV\] Chat: (?:\[Discord[^\]]*\]|<[^>]*>\[<[^>]*>Discord</[^>]*>\]</[^>]*>) (.*?) "
                     r"(?:>|<gray>></gray>) (.*)$", re.S)
# NUSH's staff feed shows a quarantined player's chat line, which only its author saw, as "[NUSH] " + the line as
# rendered for its author; a line after lines the feed hid ends in " (+N more)"
NUSH = "[NUSH] "
NUSH_MORE = re.compile(r" \(\+\d+ more\)$")
# Module-ECC's alert for a line it holds, logged when it is held: the account, the channel's words, whether ECC-Server
# was down (the line reaches it when it answers again), the text (line breaks as one space, cut at 256 code points +
# "..."). A held chat or guild line reaches the log only when it is delivered, later, on one of the module's threads,
# without "[Not Secure]".
HELD = re.compile(r"^\[Plex\] \[ECC\] Held (\S+)'s (.+?)( because ECC-Server did not answer)?: (.*)$", re.S)
UUID_OF = re.compile(r"^UUID of player (\S+) is ([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$")
HELD_CHANNELS = {"public chat": "chat", "guild chat": "guild"}
LATE_THREADS = ("Plex-ECC-Actions", "Plex-ECC-Asker")
MODULE_LOG = re.compile(r"^\[(?:Plex(?: [A-Z][a-z]+)?|Module-[^\]]*)\] ")     # Plex's and the module's own log lines
HELD_TEXT_LIMIT = 256
ADDRESS = re.compile(r"^(\S+?)\[/(.*)\] logged in with entity id")
PART = re.compile(r"^(.+)\.part(\d+)of(\d+)$")
PART_MARK = re.compile(r"^--- part \d+ begins here ---$")
LINE_ID = re.compile(r"^([LR]\d+) ")
EVENT = re.compile(r"^([LR]\d+) \d\d:\d\d:\d\d (\S+) (.+?): (.*)$")

# EssentialsX's message commands under every alias its plugin.yml gives them
ESS_MSG = {"msg", "w", "m", "t", "pm", "emsg", "epm", "tell", "etell", "whisper", "ewhisper"}
ESS_REPLY = {"r", "er", "reply", "ereply"}
ESS_MAIL = {"mail", "email", "eemail", "memo", "ememo"}
ESS_ME = {"me", "action", "eaction", "describe", "edescribe", "eme"}
# NUSH cancels these for a quarantined player: Plex's block_on_mute (me, say, msg, reply, mail), each command under
# every alias. It logs each as a feed line "[NUSH] <account>: <command line>", except a player's lines after the first
# in one FEED_SECONDS interval of the feed (the next shown line counts them).
BLOCKED_ON_MUTE = ESS_MSG | ESS_REPLY | ESS_MAIL | ESS_ME | {"say"}
FEED_COMMAND = re.compile(r"^\[NUSH\] (\S+): (/.*)$", re.S)
FEED_SECONDS = 10
SEP = re.compile(" » ")
UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
CHAT_RECORD_SECONDS = 2  # a chat line in the log and in the record are one line within this many seconds
SUDO_ALERT_SECONDS = 1   # Module-ECC logs the held alert of a line /sudo forced within this many seconds of the /sudo
SIGN_CMDS = {"sign", "editsign"}
ITEM_CMDS = {"itemname", "iname", "irename", "itemrename", "rename", "lore"}
TAG_CMDS = {"tag", "prefix"}
JOINMSG_CMDS = {"setloginmessage", "loginmessage", "setloginmsg"}
REPORT_CMDS = {"report"}
RECORD_V4 = {"sign", "book", "item", "muted"}      # the channels the transcript takes from a v4 record
SAME_SECONDS = 10       # a line in the log and in the record are one line within this many seconds
SHIB_ACTIONS = ("register", "login", "password", "changepassword", "setpassword")


# ---------------------------------------------------------------- shared by both sources

def redact(text, counts):
    text, n = I.redact(text)
    counts["addresses in text replaced"] += n
    return text


def hms(sec):
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def tokens(text):
    return -(-len(text) // CHARS_PER_TOKEN)


def auth_arguments(name, args):
    """The words a player gave away by typing them to the authentication command `name` (lower case, any plugin
    prefix removed): its arguments (for /shib only after a register, login or password action), and the argument
    glued to the name (/register1888888). Words shorter than three characters are not kept."""
    if name.startswith("shib"):
        args = args[1:] if args and args[0].lower() in SHIB_ACTIONS else []
    glued = re.sub(r"^(?:register|regist|login)", "", name)
    return [tok for tok in args + ([glued] if glued and glued != name else []) if len(tok) >= 3]


def gives_secret(mine, text):
    """Whether `text` holds one of the words in `mine`: as a word of its own, or glued to other text by punctuation
    ("name:password")."""
    return bool(mine) and (any(tok.strip(".,!?\"'") in mine for tok in text.split())
                           or any(re.search(r"(?<!\w)" + re.escape(x) + r"(?!\w)", text) for x in mine))


def split(lines, head):
    """Parts of at most TOKEN_BUDGET tokens, cut at the longest quiet gap in the last 40% of each part's room, each
    later part starting with the OVERLAP lines before its cut. `lines` are (second, text); head(part, parts) is a
    part's header. Returns [(text, first index, end index)]."""
    budget = TOKEN_BUDGET * CHARS_PER_TOKEN
    room0 = len(head(99, 99))
    sizes = [len(t) + 1 for _, t in lines]
    cuts, start = [], 0
    while True:
        lead = max(0, start - OVERLAP) if start else 0
        room = budget - room0 - sum(sizes[lead:start])
        end, used = start, 0
        while end < len(lines) and used + sizes[end] <= room:
            used += sizes[end]
            end += 1
        if end >= len(lines):
            cuts.append((lead, start, len(lines)))
            break
        lo = start + max(1, int((end - start) * 0.6))
        best = max(range(lo, end + 1), key=lambda i: (lines[i][0] - lines[i - 1][0], i))
        cuts.append((lead, start, best))
        start = best
    return render_parts(lines, cuts, head)


def split_at(lines, ids, head, bodies):
    """Parts cut where a batch's saved parts were cut, so a rebuild that gains or loses lines keeps each part's name
    and lines: part k's body starts at the first line of the saved part k's body that is still built (the first part at
    the first line), and each later part starts with the OVERLAP lines before its cut. bodies: the saved parts' body
    ids, in part order."""
    pos = {x: i for i, x in enumerate(ids)}
    starts = [0]
    for k, body in enumerate(bodies[1:], 2):
        kept = [pos[x] for x in body if x in pos]
        if not kept or min(kept) <= starts[-1]:
            raise SystemExit(f"saved part {k} has no line the rebuild keeps after part {k - 1}'s")
        starts.append(min(kept))
    ends = starts[1:] + [len(lines)]
    return render_parts(lines, [(max(0, s - OVERLAP) if s else 0, s, e) for s, e in zip(starts, ends)], head)


def render_parts(lines, cuts, head):
    """[(text, first index, end index)] of the parts `cuts` [(lead, first, end)] describe."""
    out = []
    for k, (lead, s, e) in enumerate(cuts, 1):
        body = "\n".join(t for _, t in lines[lead:s])
        if lead < s:
            body += f"\n--- part {k} begins here ---\n"
        body += "\n".join(t for _, t in lines[s:e])
        out.append((head(k, len(cuts)) + body + "\n", s, e))
    return out


# ---------------------------------------------------------------- server logs

def log_lines(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            yield line.rstrip("\r\n")


def file_day(path):
    """The day a log file covers: its dated name; for latest.log the day of a full timestamp inside it, else the day
    after its folder's newest dated file."""
    m = DATED.match(path.name)
    if m:
        return m.group(1)
    for line in log_lines(path):
        im = re.search(r"\b(20\d\d-\d\d-\d\d)T\d\d:\d\d:\d\d", line)
        if im:
            return im.group(1)
    dated = sorted(DATED.match(p.name).group(1) for p in path.parent.iterdir() if DATED.match(p.name))
    return (date.fromisoformat(dated[-1]) + timedelta(days=1)).isoformat() if dated else "unknown"


def runs(folder):
    """{log file name: (run, the run's first day, the file before it in the run or None)} for one server logs folder, in
    (date, number) order. A run is one server uptime: a file whose first BOOT_LINES lines hold the startup line starts
    one (named by its transcript stem), and any other file continues the run of the file before it in (date, number)
    order, as Paper starts a new file at midnight without a restart. latest.log comes after every dated file."""
    files = []
    for p in folder.iterdir():
        m = DATED.match(p.name)
        if m:
            files.append((m.group(1), int(m.group(2)), p))
        elif p.name == "latest.log":
            files.append((file_day(p), float("inf"), p))
    out, prev = {}, None
    for day, _, p in sorted(files, key=lambda f: (f[0], f[1])):
        starts = any(BOOT in line for _, line in zip(range(BOOT_LINES), log_lines(p)))
        stem = f"{day}-latest" if p.name == "latest.log" else p.name.replace(".log.gz", "").replace(".log", "")
        out[p.name] = (stem, day, None) if starts or prev is None else (*out[prev][:2], prev)
        prev = p.name
    return out


def record_file(record, day):
    for name in (f"{day}.jsonl.zst", f"{day}.jsonl"):
        if (record / name).exists():
            return record / name
    return None


def held_text(text):
    """A line's text as Module-ECC's held alert shows it: each run of line breaks one space, over HELD_TEXT_LIMIT code
    points cut there + '...'."""
    line = re.sub(r"[\n\x0b\x0c\r\x85\u2028\u2029]+", " ", text)
    return line if len(line) <= HELD_TEXT_LIMIT else line[:HELD_TEXT_LIMIT] + "..."


def record_entries(record, day):
    """The JSON entries of the record of `day`, in order, with their line numbers."""
    path = record_file(record, day)
    if path is None:
        return []
    if path.suffix == ".zst":
        import pyarrow as pa
        data = pa.CompressedInputStream(pa.OSFile(str(path)), "zstd").read()
    else:
        data = path.read_bytes()
    out = []
    for i, raw in enumerate(data.splitlines(), 1):
        try:
            out.append((i, json.loads(raw)))
        except json.JSONDecodeError:
            continue
    return out


def record_sent(record, day):
    """The public chat and guild lines ECC-Server received from the day before `day` to the day after, as its v4
    record holds them (`line` entries and the `repeat` entries of their copies): dicts of sec (seconds from `day`'s
    midnight), channel, account (the UUID), name (the name the judge is shown), raw_name (the name as sent) and raw."""
    out = []
    for k in (-1, 0, 1):
        d = date.fromisoformat(day) + timedelta(days=k)
        lines = {}
        for _, e in record_entries(record, d.isoformat()):
            if e.get("type") == "line" and e.get("channel") in HELD_CHANNELS.values():
                lines[(e.get("stream"), e.get("id"))] = e
                src = e
            elif e.get("type") == "repeat" and (e.get("stream"), e.get("of")) in lines:
                src = lines[(e.get("stream"), e.get("of"))]
            else:
                continue
            at = e["time"]
            sec = ((date.fromisoformat(at[:10]) - date.fromisoformat(day)).days * 86400 + int(at[11:13]) * 3600
                   + int(at[14:16]) * 60 + float(at[17:].rstrip("Z")))
            out.append(dict(sec=sec, channel=src["channel"], account=src.get("account"), name=src.get("name") or "",
                            raw_name=src.get("raw_name"), raw=e["raw"]))
    return out


def sent_line(sent, used, ch, alert, account, t, down, uuids):
    """The record line (an index into `sent`) a held line never delivered was, or None: the same channel, the text
    the alert shows, received within SAME_SECONDS of the alert (when ECC-Server was down, from then on), by the
    account's UUID when the log names it, else by no other named account's; the line whose name holds the account
    first, then the nearest in time."""
    own = uuids.get(account.lower())
    others = set(uuids.values()) - {own}
    hits = [i for i, e in enumerate(sent) if i not in used and e["channel"] == ch and held_text(e["raw"]) == alert
            and t - SAME_SECONDS <= e["sec"] and (down or e["sec"] <= t + 1 + SAME_SECONDS)
            and (e["account"] == own if own else e["account"] not in others)]
    return min(hits, key=lambda i: (account.lower() not in sent[i]["name"].lower(), abs(sent[i]["sec"] - t)),
               default=None)


def record_events(record, day, start, end):
    """(second of day, id, channel, speaker, text) of the record's signs, books, item names and muted lines on `day`
    within [start, end] seconds: of ECC-Server's v3 record (`message` entries; a muted line is a shadowed chat, /me or
    nickname line) and of its v4 record (`line` entries and the `repeat` entries of their copies, the speaker the
    account name and the text raw)."""
    out, lines = [], {}
    for i, e in record_entries(record, day):
        # Signs, books and item names never reach the server log; a line typed while muted reaches it only through
        # NUSH's feed, which hides some.
        if e.get("type") == "message":
            if e.get("channel") in ("sign", "book", "anvil"):
                ch = "item" if e["channel"] == "anvil" else e["channel"]
            elif e.get("shadowed") and e.get("channel") == "nick":
                ch = "nick"     # a quarantined player's /nick is checked as any /nick is; the log holds the command
            elif e.get("shadowed") and e.get("channel") in ("chat", "me"):
                ch = "muted"
            else:
                continue
            at, speaker, text = e["ts"], e["speaker"], e["text"]
        elif e.get("type") == "line":
            lines[(e.get("stream"), e.get("id"))] = (e.get("channel"), e.get("name"))
            if e.get("channel") not in RECORD_V4:
                continue
            ch, at, speaker, text = e["channel"], e["time"], e["name"], e["raw"]
        elif e.get("type") == "repeat":
            ch, speaker = lines.get((e.get("stream"), e.get("of")), (None, None))
            if ch not in RECORD_V4:
                continue
            at, text = e["time"], e["raw"]
        else:
            continue
        h, m, s = at[11:19].split(":")
        sec = int(h) * 3600 + int(m) * 60 + int(s)
        if start <= sec <= end:
            out.append((sec, f"R{i}", ch, speaker, text))
    return out


class LogPlayers(Players):
    """parse.Players as the transcript reads a log: a typed player name resolves as EssentialsX resolves it, and a chat
    name's end tells who spoke."""

    def recipient(self, typed, offline):
        """The account EssentialsX's matchUser gives a typed name, or None when it gives none (the command then sends
        nothing): an online account of that name (any case); for /msg (offline False) the online account whose name
        starts with it, the shortest; an account of that name that is offline (/msg refuses it, mail takes it); the
        online account whose name contains it, the one of that name first; the online player whose display name (the
        /nick, or the name they last spoke under, both folded) contains it."""
        t = typed.lower()
        if not t:
            return None
        if UUID.fullmatch(t):       # an online player's UUID
            return next((self.online[a] for a, u in getattr(self, "uuids", {}).items() if u == t and a in self.online),
                        None)
        if t in self.online:
            return self.online[t]
        if not offline:
            starts = [a for k, a in self.online.items() if k.startswith(t)]
            if starts:
                return min(starts, key=len)
        if t in self.known:
            return self.known[t] if offline else None
        hits = [a for k, a in self.online.items() if t in k]
        if hits:
            return next((a for a in hits if a.startswith(typed)), hits[0])
        f = fold(typed)
        return next((a for k, a in self.online.items()
                     if f and (f in self.nicks.get(k, "") or f in self.shown.get(k, ""))), None)

    def shows(self, display):
        """The account a chat name shows at its end, where the console puts the display name: an online account's
        name as is, or the /nick of exactly one online player; else None."""
        tokens = [t for t in plain_name(display).split() if re.search(r"\w", t)]
        if not tokens:
            return None
        name = EDGE.sub("", tokens[-1])
        if not tokens[-1].startswith("~") and name.lower() in self.online:
            return self.online[name.lower()]
        f = fold(name)
        nicked = [a for k, a in self.online.items() if f and self.nicks.get(k) == f]
        return nicked[0] if len(nicked) == 1 else None


def chat_parts(players, msg):
    """(shown name, text) of a console chat line, or None. Paper logs a chat line as the rendered name, " » " and the
    text, and a name can hold " » " itself (a tag ending in "»"): the name ends at the first " » " unless it shows no
    player (LogPlayers.shows) and a name ending at a later " » " does."""
    c = CHAT.match(msg)
    if not c:
        v = VANILLA.match(msg)
        return (v.group(1), v.group(2)) if v else None
    if players.shows(c.group(1)) is None:
        for m in SEP.finditer(msg, c.end(1) + 1):
            if players.shows(msg[:m.start()]) is not None:
                return msg[:m.start()], msg[m.end():]
    return c.group(1), c.group(2)


def message_command(players, speaker, cmd, rest, counts):
    """(channel, target, text) of an EssentialsX /msg, /r or /mail command line as EssentialsX sends it, or None when
    it sends nothing (no such player, /r with no one to answer, a mail subcommand that sends no mail). /msg and /r set
    both players' reply targets; mail sets none. /mail sendall and sendtempall send to every player: one line, target
    "*"."""
    if cmd in ESS_MAIL:
        args = rest.split(" ")
        sub = args[0].lower()
        if sub == "send" and len(args) >= 3:
            to, text = players.recipient(args[1], True) or args[1], " ".join(args[2:])
        elif sub == "sendtemp" and len(args) >= 4:
            to, text = players.recipient(args[1], True) or args[1], " ".join(args[3:])
        elif sub == "sendall" and len(args) >= 2:
            to, text = "*", " ".join(args[1:])
        elif sub == "sendtempall" and len(args) >= 3:
            to, text = "*", " ".join(args[2:])
        else:
            return None
        return "mail", to, text
    if cmd in ESS_REPLY:
        to, text = players.partner.get(speaker.lower()), rest
        why = "/r with no one to answer"
    else:
        typed, _, text = rest.partition(" ")
        to = speaker if typed in ("@p", "@s") else "Console" if typed.lower() == "console" else \
            players.recipient(typed, False)
        why = "/msg to an offline account" if typed.lower() in players.known else \
            "/msg to every player (* or **)" if typed in ("*", "**") else "/msg to no player"
    if not text.strip():
        return None
    if not to:
        counts[why] += 1
        return None
    players.partner[speaker.lower()] = to
    players.partner[to.lower()] = speaker
    return "msg", to, text


def feed_commands(path):
    """{(lower account, command line): [second of day]} of NUSH's feed lines for the commands it cancelled, each
    command line without the " (+N more)" the feed adds after lines it hid."""
    out = collections.defaultdict(list)
    for line in log_lines(path):
        m = LINE.match(line)
        f = FEED_COMMAND.match(m.group(5)) if m and m.group(4).startswith("Server thread") else None
        if f:
            out[(f.group(1).lower(), NUSH_MORE.sub("", f.group(2)))].append(int(m.group(1)) * 3600 + int(m.group(2)) * 60
                                                         + int(m.group(3)))
    return out


def record_chat(record, day):
    """The public chat lines of `day`'s record, as [second of day, account (an account name, or for a v4 record the
    UUID), raw text, used]: v3 `message` entries not shadowed, v4 `line` entries of channel chat and their copies."""
    out, lines = [], {}
    for _, e in record_entries(record, day):
        if e.get("type") == "message" and e.get("channel") == "chat" and not e.get("shadowed"):
            at, who, raw = e["ts"], e["speaker"], e["text"]
        elif e.get("type") == "line" and e.get("channel") == "chat":
            lines[(e.get("stream"), e.get("id"))] = e
            at, who, raw = e["time"], ("uuid", e.get("account")), e["raw"]
        elif e.get("type") == "repeat" and (e.get("stream"), e.get("of")) in lines:
            at, who, raw = e["time"], ("uuid", lines[(e.get("stream"), e.get("of"))].get("account")), e["raw"]
        else:
            continue
        h, m, s = at[11:19].split(":")
        out.append([int(h) * 3600 + int(m) * 60 + int(s), who, raw, False])
    return out


def typed_secrets(path):
    """{lower name: words that name typed to an authentication command anywhere in the log file}. A player who types
    a password into chat by mistake gives it away; any line of theirs holding one of these words is dropped."""
    secrets = collections.defaultdict(set)
    for line in log_lines(path):
        m = LINE.match(line)
        c = CMD.match(m.group(5)) if m else None
        if not c:
            continue
        name = c.group(2).lower().split(":")[-1]
        if AUTH.match(name):
            secrets[c.group(1).lower()].update(auth_arguments(name, (c.group(3) or "").split()))
    return secrets


def who(display, account):
    """The speaker field: the name as shown, and the account it resolved to when the shown name does not contain it.
    A chat line read from its held alert alone has no shown name (None): its account."""
    s = shown(display or "") or account
    return s if account.lower() in s.lower() else f"{s}={account}"


def server_log_lines(path, record, counts, carry=None, state=None):
    """The transcript lines of one log file, in order. Returns the day, [(second of day, rendered line)], the ids, per
    line (the account its speaker resolved to, how), and per line what the plugin sends for it: raw (the text before
    preparation), shown (a chat line's name as the log shows it, a Discord line's name as relayed; else None) and the
    copies folded into it [(id, second of day, raw, shown)], with the line's channel, name (as the judge is shown it),
    text, count and second of day. A text empty after preparation makes no line.

    A chat or guild line Module-ECC held is read where its held alert was logged: with what the console shows when it
    is delivered late on the module's thread; when this file has no delivery, as the v4 record holds it (the shown name
    and raw text ECC-Server received), or from the alert (account and text) when the record does not hold it.
    A quarantined player's line in NUSH's feed is a muted line by account, unless the record holds it; a sign or item
    name set by command is read from the log even when the record holds it too. A chat line the day's record holds
    takes the record's account. A command NUSH cancelled, and a message EssentialsX sends to no one, make no line.
    A chat line a player's `/sudo <player> c:<text>` forced is the issuer's, under the forced player's shown name, as
    the plugin sends it: EssentialsX has the player say it while the command runs, so Paper logs it on the server
    thread in that second (NUSH's feed line for a quarantined player, a muted line); held, its alert names the issuer.
    The player's own chat, on the chat thread, is always theirs. A command /sudo forced is the issuer's too.

    carry: the repeat keys and held alerts still open at the end of the file before in the run, {key: (transcript, line
    id, second relative to this file's midnight)} (a held alert's key starts with None); then also returns this
    file's open keys and alerts (the next file's carry, with this file's transcript unset for its own lines) and the
    copies folded into the file before's lines [(transcript, line id, copy)].

    state: the player state at the end of the file before in the run, (parse.Players, {lower account: UUID}), its
    seconds relative to this file's midnight (carried_state), so a player online since before midnight keeps their
    account, nickname and UUID; a file that starts a run starts with nobody known, as the server does. This file's
    state at its end is returned last; the given state is changed in place."""
    day = file_day(path)
    secrets = typed_secrets(path)
    feed = feed_commands(path)
    feed_last = {}      # lower account -> the second of its last NUSH feed line
    in_record = record_chat(record, day)
    players, uuids = state or (LogPlayers(), {})    # uuids: lower account name -> UUID, from "UUID of player" lines
    players.uuids = uuids
    aliases = {}
    out = []        # (second, order, id, channel, speaker, text, (account, how), (raw, shown))
    first = last = None
    # Held chat and guild lines not yet delivered, by (channel, alert text): [[second, order, id, account, read, down]],
    # read when the line was read in the file before in the run, down when ECC-Server did not answer
    # and the lower accounts the alert's account forced to say its text with /sudo (the line is the issuer's)
    held = collections.defaultdict(list)
    for key, (stem, rid, t) in (carry or {}).items():
        if key[0] is None:
            held[key[2:4]].append([t, -1, rid, key[4], stem, False, set()])

    def player(ch, speaker, raw, acct, shown_raw=None, limit=None, at=None):
        text = clean(raw, limit or I.limit(ch.split(">")[0]))
        if text:
            s, k, i = at or (sec, n, lid)
            out.append((s, k, i, ch, speaker, text, acct, (raw, shown_raw)))

    def delivered(ch, account, raw):
        """Where a held line delivered late on the module's thread was written, and the /sudo issuer it belongs to:
        ((second, order, id) of the oldest alert for its channel and text, its author's first, then one of an issuer
        who forced its author; the issuer or None); (None, None) when no alert is open (the line is read where it is
        logged); (False, None) when the line was read in the file before."""
        open_ = held.get((ch, held_text(raw)))
        if not open_:
            counts[f"late {ch} lines without their held alert"] += 1
            return None, None
        low = account.lower()
        a = next((x for x in open_ if x[3].lower() == low), None) or next((x for x in open_ if low in x[6]), open_[0])
        open_.remove(a)
        if a[4] is not None:
            counts[f"late {ch} lines read in the file before"] += 1
            return False, None
        counts[f"late {ch} lines read where they were held"] += 1
        return tuple(a[:3]), (a[3] if a[3].lower() != low and low in a[6] else None)

    def recorded(account, raw, t):
        """The account the day's record gives a chat line: the unused record line of that raw text within
        CHAT_RECORD_SECONDS, the resolved account's first, then the nearest; None when the record holds none."""
        hits = [x for x in in_record if not x[3] and x[2] == raw and abs(x[0] - t) <= CHAT_RECORD_SECONDS]
        if not hits:
            return None
        names = {u: a for a, u in uuids.items()}

        def name(x):
            return names.get(x[1][1]) if isinstance(x[1], tuple) else x[1]
        x = min(hits, key=lambda x: ((name(x) or "").lower() != account.lower(), abs(x[0] - t)))
        x[3] = True
        acct = name(x)
        return (players.known.get(acct.lower(), acct)) if acct else None

    def chat(display, raw, late):
        """A public chat line: as the console showed it on the chat thread, or delivered late on the module's thread.
        A guild tag ("[Guild] name >> text") is part of the shown name. Where the day's record holds the line, its
        account is the record's; a held line another player forced with /sudo is that issuer's."""
        account, how = players.speaker(display)
        at, issuer = delivered("chat", account, raw) if late else (None, None)
        if at is False:
            return
        r = recorded(issuer or account, raw, at[0] if at else sec)
        if r:
            counts["chat lines whose account the record gives" + ("" if r.lower() == account.lower() else
                                                                  ", another than the log's")] += 1
            account, how = r, "record"
        if issuer:
            counts["held lines forced by /sudo, read as the issuer's"] += 1
            account, how = issuer, "sudo"
        else:
            tokens = display.split()
            players.shown[account.lower()] = fold(tokens[-1]) if tokens else ""
        if AUTH_CHAT.match(strip_colour(raw)):
            counts["authentication dropped"] += 1
            return
        player("chat", who(display, account), raw, (account, how), display, at=at)

    for n, line in enumerate(log_lines(path), 1):
        m = LINE.match(line)
        if not m:
            continue
        sec = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
        first = sec if first is None else first
        last = sec
        thread, msg = m.group(4), m.group(5)
        lid = f"L{n}"
        for k in [k for k, (_, t, kind) in players.sudo.items() if kind == "chat" and sec - t > SUDO_ALERT_SECONDS] \
                if players.sudo else ():
            del players.sudo[k]
            counts["/sudo c: targets with no line on the server thread"] += 1
        h = HELD.match(msg)
        if h:
            if h.group(2) in HELD_CHANNELS:
                # a forced line is held while its /sudo runs; the alert names the issuer
                forced = {k for k, (issuer, t, kind) in players.sudo.items() if kind == "chat" and
                          issuer.lower() == h.group(1).lower() and 0 <= sec - t <= SUDO_ALERT_SECONDS}
                held[(HELD_CHANNELS[h.group(2)], h.group(4))].append([sec, n, lid, h.group(1), None, bool(h.group(3)),
                                                                      forced])
            continue
        u = UUID_OF.match(msg) if thread.startswith("User Authenticator") else None
        if u:
            uuids[u.group(1).lower()] = u.group(2)
            continue
        late = thread.startswith(LATE_THREADS)
        g = GUILD.match(msg)
        if g:       # Module-Guilds' guild chat, on whichever thread logged it
            at = delivered("guild", g.group(3), g.group(4))[0] if late else None
            if at is not False:
                players.seen(g.group(3))
                player("guild", g.group(3), g.group(4), (g.group(3), "guild"), at=at)
            continue
        if late:
            # a held chat line delivered late: the console's rendering, as on the chat thread without "[Not Secure]"
            c = None if MODULE_LOG.match(msg) else chat_parts(players, msg)
            if c:
                chat(*c, True)
            continue
        if thread.startswith("Async Chat Thread"):
            if ADMIN.match(msg):
                counts["staff chat left out"] += 1
                continue
            c = chat_parts(players, msg)
            if not c:
                continue
            if msg.startswith(NUSH):
                # a quarantined player's line, which only its author saw: sent to ECC-Server as muted, by account
                account, how = players.speaker(c[0])
                feed_last[account.lower()] = sec
                if AUTH_CHAT.match(strip_colour(c[1])):
                    counts["authentication dropped"] += 1
                    continue
                player("muted", account, NUSH_MORE.sub("", c[1]), (account, how))
                continue
            if " » " in c[0]:
                counts["chat names holding ' » '"] += 1
            chat(*c, False)
            continue
        if thread.startswith("JDA"):
            d = DISCORD.match(msg)
            if d:
                player("discord", I.discord_name(d.group(1)), d.group(2), ("", "discord"), d.group(1))
            continue
        if not thread.startswith("Server thread"):
            continue
        f = FEED_COMMAND.match(msg)
        if f:
            feed_last[f.group(1).lower()] = sec
            continue
        c = CMD.match(msg)
        if c:
            name, cmd, rest = c.group(1), c.group(2).lower().split(":")[-1], c.group(3) or ""
            players.seen(name)
            if AUTH.match(cmd) or AUTH.match(c.group(2).lower()):
                counts["authentication dropped"] += 1
                continue
            forced = players.credit(name, sec, "command", 2)
            speaker = forced or name
            acct = (speaker, "sudo" if forced else "command")
            shown_feed = any(0 <= t - sec <= 2 for t in feed.get((name.lower(), msg.split(": ", 1)[1]), ()))
            if cmd in BLOCKED_ON_MUTE and (
                    shown_feed or sec - feed_last.get(name.lower(), -FEED_SECONDS - 1) <= FEED_SECONDS):
                # NUSH cancelled it (a quarantined player's): its feed line follows, or the feed showed one of the
                # player's lines in this interval and hid this one
                counts["commands NUSH cancelled, " + ("in its feed" if shown_feed else "hidden by its feed")] += 1
                continue
            if cmd in SUDO:
                typed, _, what = rest.partition(" ")
                if what[:2].lower() == "c:" and not what[2:].startswith("/"):
                    # EssentialsX makes each player it names (every online player for * and **; never the issuer)
                    # say the text while the command runs, on this thread, in this second
                    targets = players.online.values() if typed in ("*", "**") else [players.recipient(typed, False)]
                    for t in targets:
                        if t and t.lower() != name.lower():
                            players.sudo[t.lower()] = (name, sec, "chat")
                else:
                    command(players, {}, speaker, cmd, rest, sec, "", "")
                continue
            if cmd in ESS_ME:
                player("me", speaker, rest, acct)
                continue
            if cmd in SIGN_CMDS or cmd in ITEM_CMDS or cmd in TAG_CMDS or cmd in JOINMSG_CMDS:
                text = re.sub(r"^(?:set|add|-o)\s+(?:\d+\s+)?", "", rest.strip(), flags=re.I)
                if text and text.lower() not in ("clear", "remove", "off"):
                    player("sign" if cmd in SIGN_CMDS else "item" if cmd in ITEM_CMDS else
                           "tag" if cmd in TAG_CMDS else "joinmsg", speaker, text, acct)
                continue
            if cmd in REPORT_CMDS:
                typed, _, text = rest.partition(" ")
                if typed:
                    target = players.account(typed) or typed
                    out.append((sec, n, lid, f"report>{target}", speaker, clean(text) or "(no reason)", acct,
                                (text, None)))
                continue
            if cmd in ESS_MSG or cmd in ESS_REPLY or cmd in ESS_MAIL:
                got = message_command(players, speaker, cmd, rest, counts)
                if got is not None:
                    player(f"{got[0]}>{got[1]}", speaker, got[2], acct)
                continue
            if cmd in NICK:
                rec = command(players, {}, speaker, cmd, rest, sec, "", "")
                if rec is not None and not rec.get("drop"):
                    owner = rest.split(" ", 1)[0] if rest.split(" ", 1)[0].lower() in players.online and \
                        len(rest.split(" ", 1)) > 1 else None
                    player("nick" if not owner or owner.lower() == speaker.lower()
                           else f"nick>{players.online[owner.lower()]}", speaker, rec["raw"], acct)
            continue
        c = chat_parts(players, msg) if any(kind == "chat" and t == sec for _, t, kind in players.sudo.values()) \
            and not msg.startswith("[AdminChat]") else None
        if c:
            # the line a /sudo c: in this second forced: the issuer's, under the forced player's name; NUSH's feed
            # line when the forced player is quarantined (a muted line)
            target, _ = players.speaker(c[0])
            issuer = players.credit(target, sec, "chat", 0)
            if issuer:
                counts["/sudo forced chat lines read as the issuer's"] += 1
                if AUTH_CHAT.match(strip_colour(c[1])):
                    counts["authentication dropped"] += 1
                elif msg.startswith(NUSH):
                    feed_last[target.lower()] = sec
                    player("muted", issuer, NUSH_MORE.sub("", c[1]), (issuer, "sudo"))
                else:
                    player("chat", who(c[0], issuer), c[1], (issuer, "sudo"), c[0])
                continue
        lm = ADDRESS.match(msg)
        if lm:
            name, addr = lm.group(1), lm.group(2)
            host = re.sub(r":\d+$", "", addr).strip("[]")
            if host not in aliases:
                aliases[host] = f"addr-{len(aliases) + 1:02d}"
            players.seen(name)
            out.append((sec, n, lid, "join", name, aliases[host], (name, "join"), (None, None)))
            continue
        lv = LEAVE.match(msg)
        if lv:
            gone = players.online.pop(lv.group(1).lower(), None)
            if not msg.startswith(lv.group(1) + " lost connection"):
                out.append((sec, n, lid, "leave", gone or lv.group(1), "", (gone or lv.group(1), "leave"),
                            (None, None)))
            continue
        if msg.startswith(BOOT):
            players.online.clear()
            players.sudo.clear()
            out.append((sec, n, lid, "start", "server", "the server starts", ("", "start"), (None, None)))
    # A held line never delivered in this file: its alert holds its account and text, not the name the console shows.
    # The v4 record holds the line as ECC-Server received it, its shown name and raw text; without the record it is
    # read from the alert, its account as the name.
    held_tail, sent, used = {}, None, set()
    for t, k, rid, account, stem, down, _, ch, alert in sorted(
            (a + [ch, alert] for (ch, alert), open_ in held.items() for a in open_), key=lambda a: (a[0], a[1])):
        if stem is None:
            if sent is None:
                sent = record_sent(record, day)
            i = sent_line(sent, used, ch, alert, account, t, down, uuids)
            if i is None:
                counts[f"held {ch} lines read from their alert"] += 1
                player(ch, account, alert, (account, "held"), at=(t, k, rid))
            else:
                used.add(i)
                e = sent[i]
                counts[f"held {ch} lines read from the record"] += 1
                if ch == "chat":
                    player(ch, who(e["raw_name"], account), e["raw"], (account, "held"), e["raw_name"], at=(t, k, rid))
                else:
                    player(ch, e["name"], e["raw"], (account, "held"), at=(t, k, rid))
        held_tail[(None, "held", ch, alert, account, stem, rid)] = (stem, rid, t)
    if first is not None:
        # A line the log and the record both hold is read once: a quarantined line in NUSH's feed is the record's
        # line (the feed hides some), a sign or item name set by command (a v4 record holds those too) and a
        # quarantined player's /nick (a v3 record holds it as shadowed) the log's.
        logged = [i for i, r in enumerate(out) if r[3] in ("muted", "sign", "item", "nick")]
        matched, dropped = set(), set()
        for sec, rid, ch, speaker, raw in record_events(record, day, first, last):
            i = next((i for i in logged if i not in matched and out[i][3] == ch and out[i][7][0] == raw and
                      abs(out[i][0] - sec) <= SAME_SECONDS and out[i][6][0].lower() == speaker.lower()), None)
            if i is not None:
                matched.add(i)
                if ch != "muted":
                    counts["lines set by command the record holds"] += 1
                    continue
                dropped.add(i)
                counts["quarantined lines of the feed the record holds"] += 1
            text = clean(raw, I.limit(ch))
            if text:
                out.append((sec, 10 ** 9, rid, ch, speaker, text, (speaker, "record"), (raw, None)))
        out = [r for i, r in enumerate(out) if i not in dropped]
    kept = []
    for r in out:
        # an authentication command as the text of any line, whatever carried it (a private message, a Discord
        # relay, a sign, a book)
        if r[3] not in ("join", "leave", "start") and AUTH_CHAT.match(strip_colour(r[5])):
            counts["authentication dropped"] += 1
            continue
        # the password list is keyed by account; a tagged name such as "[Admin] Bob" carries no "=account", so the
        # resolved account is tried as well as the name's tail
        if r[3] not in ("join", "leave", "start", "discord") and (
                gives_secret(secrets.get(r[4].rsplit("=", 1)[-1].lower()), r[5])
                or gives_secret(secrets.get((r[6][0] or "").lower()), r[5])):
            counts["authentication dropped"] += 1
            counts["password typed as text dropped"] += 1
            continue
        kept.append(r)
    out = sorted(kept, key=lambda r: (r[0], r[1]))
    # Fold a speaker's exact repeats on one channel within FOLD_SECONDS of the previous copy; a copy of a line of the
    # file before in the run (carry) is folded into that line.
    rows, last_copy = [], {}
    for key, (stem, rid, t) in (carry or {}).items():
        if key[0] is not None:
            last_copy[key] = (dict(stem=stem, id=rid, repeats=0, copies=[], carried=True), t)
    for sec, _, lid, ch, speaker, text, (acct, how), (raw, shown_raw) in out:
        if ch not in ("join", "leave", "start"):
            key = (speaker.lower(), ch, text)
            prev = last_copy.get(key)
            if prev and sec - prev[1] <= FOLD_SECONDS:
                prev[0]["repeats"] += 1
                prev[0]["until"] = sec
                prev[0]["copies"].append((lid, sec, raw, shown_raw))
                last_copy[key] = (prev[0], sec)
                counts["repeats folded"] += 1
                continue
        # an address can sit in a tag or a nickname as well as in the text, so every shown field is cleaned
        name = (I.shown(shown_raw or "") or acct) if ch == "chat" else speaker
        row = dict(sec=sec, id=lid, ch=redact(ch, counts), speaker=redact(speaker, counts),
                   name=redact(name, collections.Counter()), text=redact(text, counts), repeats=1, until=sec,
                   account=redact(acct, collections.Counter()), how=how, raw=raw, shown=shown_raw, copies=[])
        rows.append(row)
        if ch not in ("join", "leave", "start"):
            last_copy[(speaker.lower(), ch, text)] = (row, sec)
    tail = {k: (r.get("stem"), r["id"], t) for k, (r, t) in last_copy.items()} | held_tail
    carried = [(r["stem"], r["id"], c) for r, _ in last_copy.values() if r.get("carried") for c in r["copies"]]
    return (day, [(r["sec"], render(r)) for r in rows], [r["id"] for r in rows],
            [(r["account"], r["how"]) for r in rows],
            [dict(channel=r["ch"], name=r["name"], text=r["text"], sent=r["repeats"], sec=r["sec"], raw=r["raw"],
                  shown=r["shown"], copies=r["copies"]) for r in rows], tail, carried, (players, uuids))


def carried_state(state, shift):
    """A copy of a file's end state (server_log_lines) for the next file of its run, whose midnight is `shift` seconds
    later: the pending /sudo seconds move by it."""
    players, uuids = copy.deepcopy(state)
    players.sudo = {k: (issuer, t - shift, kind) for k, (issuer, t, kind) in players.sudo.items()}
    return players, uuids


def render(r):
    if r["ch"] == "join":
        return f"{r['id']} {hms(r['sec'])} join {r['speaker']} from {r['text']}"
    if r["ch"] == "leave":
        return f"{r['id']} {hms(r['sec'])} leave {r['speaker']}"
    if r["ch"] == "start":
        return f"{r['id']} {hms(r['sec'])} --- the server starts; everyone is offline ---"
    rep = f" [sent {r['repeats']} times until {hms(r['until'])}]" if r["repeats"] > 1 else ""
    return f"{r['id']} {hms(r['sec'])} {r['ch']} {r['speaker']}: {r['text']}{rep}"


def server_log_header(stem, day, record_merged, muted=False):
    rec = " Signs, books, item names and muted lines come from the server's record (ids R<n>); 'muted' is a " \
          "line typed while its speaker was muted, which nobody saw." if record_merged else \
        " This day has no record of signs, books or item names; only signs set by command appear."
    if muted and not record_merged:
        rec += " 'muted' is a line typed while its speaker was quarantined, which only they saw."

    def head(part, parts):
        over = f" The first {OVERLAP} lines repeat the end of part {part - 1}, for context." if part > 1 else ""
        # The part text the readers read; its words stay as the reads answered them, so rebuilt parts stay byte-equal.
        return (f"# Dossier {stem}, part {part} of {parts}. Total Freedom (a Minecraft server), {day}, times UTC.\n"
                f"# Line: id time channel speaker: text. Channels: chat (everyone sees), msg>NAME (private message to "
                f"NAME), mail>NAME, me (/me action), nick (nickname set), guild, discord (relayed from Discord), sign, "
                f"book, item (item name), tag, joinmsg, report>NAME. 'join NAME from addr-NN' gives the login's "
                f"address as an alias; the same alias is the same address within this dossier. Names are not "
                f"identities: anyone can log in under any name.{rec}{over}\n\n")
    return head


def server_log_transcript(path, record, bodies=None, carry=None, state=None):
    """One log file's transcript: stem, day, counts, parts [(text, first index, end index)], ids, accounts, each line's
    fields and what the plugin sends for it (server_log_lines `raw`), the repeat keys open at its end (`tail`, keyed to this
    transcript), the copies folded into lines of the file before (`carried`), the player state at its end (`state`),
    whether the day's record was merged. bodies: the body ids of the saved parts to cut at (split_at), or None for a
    new cut; carry, state: as server_log_lines."""
    counts = collections.Counter()
    day, lines, ids, accounts, raw, tail, carried, end = server_log_lines(path, record, counts, carry, state)
    stem = path.name.replace(".log.gz", "").replace(".log", "")
    if stem == "latest":
        stem = f"{day}-latest"
    tail = {k: (st or stem, rid, t) for k, (st, rid, t) in tail.items()}
    merged = record_file(record, day) is not None
    head = server_log_header(stem, day, merged, any(r["channel"] == "muted" for r in raw))
    parts = (split_at(lines, ids, head, bodies) if bodies else split(lines, head)) if lines else []
    return dict(stem=stem, day=day, counts=counts, parts=parts, ids=ids, accounts=accounts, raw=raw, tail=tail,
                carried=carried, state=end, record=merged)


# ---------------------------------------------------------------- Creayun

# Secret shapes, replaced in this order. URL-bound shapes run before the e-mail and sentence shapes.
WEBHOOK = re.compile(r"(?i)(discord(?:app)?\.com/api/webhooks/)\d+/[\w-]+")
URL_KEY = re.compile(r"(?i)([?&#;](?:key|api_?key|apikey|token|access_?token|auth|sig|signature|secret|password|pass|"
                     r"pwd|passcode|code|session|sid|ticket)=)[^&\s#]+")
ANY_URL = re.compile(r"(?i)(?:https?|wss?|ftp|file)://\S+|\b(?:www\.|docs\.google\.com/|drive\.google\.com/)\S+")
URL_ID = re.compile(r"(?<![A-Za-z0-9])(?=[A-Za-z0-9_-]*\d)(?=[A-Za-z0-9_-]*[A-Za-z])[A-Za-z0-9_-]{20,}")
EMAIL_PASS = re.compile(r"(?i)[\w.+-]+@[\w-]+(?:\.[\w-]+)+\s*[:|]\s*\S{3,}")
EMAIL = re.compile(r"(?i)[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[a-z]{2,}")
SAID = re.compile(r"(?i)\b(pass(?:word)?s?|pw|pwd|passcode|pin)(\s*(?:is|=|:|-|was)\s*)(?!\[secret\])(\S+)")
DISCORD_TOKEN = re.compile(r"\b[MNO][\w-]{23,27}\.[\w-]{6}\.[\w-]{27,40}\b")
JWT = re.compile(r"\beyJ[\w-]{10,}\.[\w-]{10,}\.[\w-]{5,}")
API_KEY = re.compile(r"\b(?:sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{30,}|xox[abprs]-[\w-]{10,}|AKIA[0-9A-Z]{16}|"
                     r"AIza[\w-]{30,})")
DIGITS = re.compile(r"(?<![\d-])\d(?:[ -]?\d){14,18}(?![\d])")
# a run of 32 or more characters with upper and lower case letters and digits: a key, not a keyboard mash
KEYLIKE = re.compile(r"(?<![\w/+=-])(?=[\w+/=-]*\d)(?=[\w+/=-]*[a-z])(?=[\w+/=-]*[A-Z])[\w+/=-]{32,}")
NETWORK = ("ayunfreedom", "ayunboom", "ayunsurvival", "creative", "survival", "boxes", "zombers", "legacy")


def card_number(s):
    """A 15- or 16-digit number that passes the Luhn check and starts like a card (3-6), not a run of few digits."""
    d = re.sub(r"\D", "", s)
    if len(d) not in (15, 16) or d[0] not in "3456" or len(set(d)) <= 3:
        return False
    total = 0
    for i, ch in enumerate(reversed(d)):
        x = int(ch) * (2 if i % 2 else 1)
        total += x - 9 if x > 9 else x
    return total % 10 == 0


def hide_secrets(text, counts):
    """The text with every secret shape replaced; counts by shape."""
    def sub(rx, repl, t, name):
        t, n = rx.subn(repl, t)
        counts[name] += n
        return t

    text = sub(WEBHOOK, r"\1[secret]", text, "webhook token")
    text = sub(URL_KEY, r"\1[secret]", text, "key in a link")

    def url_ids(m):
        u, n = URL_ID.subn("[id]", m.group(0))
        counts["long id in a link"] += n
        return u
    text = ANY_URL.sub(url_ids, text)
    text = sub(EMAIL_PASS, "[email] [secret]", text, "e-mail with password")
    text = sub(EMAIL, "[email]", text, "e-mail address")
    text = sub(DISCORD_TOKEN, "[secret]", text, "Discord token")
    text = sub(JWT, "[secret]", text, "JSON web token")
    text = sub(API_KEY, "[secret]", text, "API key")
    text = sub(SAID, r"\1\2[secret]", text, "password stated in a sentence")

    def card(m):
        if card_number(m.group(0)):
            counts["card-shaped number"] += 1
            return "[long number]"
        return m.group(0)
    text = DIGITS.sub(card, text)
    return sub(KEYLIKE, "[random string]", text, "key-like string")


def creayun_secrets(exports):
    """{lower username: words typed to an authentication command} from every flagged export in `exports` (the
    message table keeps no authentication command, so the exports are the only place they show), and the count of
    commands seen."""
    secrets, seen = collections.defaultdict(set), 0
    for path in sorted(p for p in exports.iterdir() if p.is_file() and "flagged" in p.name):
        for _, _, user, _, message in flagged_blocks(path):
            raw = strip_colour("\n".join(strings(message))).strip()
            m = re.match(r"^[/\\.]+\s*(\S+)\s*(.*)$", raw, re.S)
            if m and (AUTH.match(m.group(1).lower().split(":")[-1]) or AUTH.match(m.group(1).lower())):
                name, args = m.group(1).lower().split(":")[-1], m.group(2).split()
            elif AUTH_CHAT.match(raw):
                words = raw.lstrip("/\\. ").split()
                name, args = words[0].lower(), words[1:]
            else:
                continue
            seen += 1
            secrets[user.lower()].update(auth_arguments(name, args))
    return secrets, seen


def creayun_header(stem, server, day):
    where = f"its server '{server}'" if server in NETWORK else "one of its servers (the export does not name which)"

    def head(part, parts):
        over = f" The first {OVERLAP} lines repeat the end of part {part - 1}, for context." if part > 1 else ""
        # The part text the readers read; its words stay as the reads answered them, so rebuilt parts stay byte-equal.
        return (f"# Dossier {stem}, part {part} of {parts}. Creayun (a Minecraft server network), {where}, {day}; "
                f"times are to the minute (seconds always :00).\n"
                f"# Creayun's own servers are {', '.join(NETWORK)}; players move between them with /server NAME.\n"
                f"# Line: id time channel speaker: text. Channels: chat (everyone on this server sees), msg>NAME "
                f"(private message to NAME). This is a selection of the chat, not all of it: lines between these may "
                f"be missing. Names are not identities: anyone can log in under any name.{over}\n\n")
    return head


def creayun_lines(rows, secrets, counts):
    """One server-day's table rows in time order (index = table row) as transcript lines. Returns ([(second, line)],
    the kept rows, {folded row: the row shown}, {dropped row: reason}). The table has minute timestamps and already
    folds a speaker's repeat of their previous message; the fold here (a repeat of any earlier copy within
    FOLD_SECONDS) applies on top. The id of a line is L<row>."""
    kept, folded, dropped, last_copy = [], {}, {}, {}
    for r in rows.itertuples():
        row = r.Index
        sec = int(r.ts[11:13]) * 3600 + int(r.ts[14:16]) * 60
        raw = r.text if isinstance(r.text, str) else ""
        if AUTH_CHAT.match(strip_colour(raw)) or AUTH_START.match(strip_colour(raw)):
            counts["authentication dropped"] += 1
            dropped[row] = "authentication command"
            continue
        mine = secrets.get(r.speaker.lower())
        if gives_secret(mine, strip_colour(raw)):
            # the word itself is replaced, so the rest of the line stays readable; a line that still holds it (glued
            # to other text in a way the replacement misses) is dropped
            raw = strip_colour(raw)
            for x in sorted(mine, key=len, reverse=True):
                raw = re.sub(r"(?<!\w)" + re.escape(x) + r"(?!\w)", "[secret]", raw)
            raw = " ".join("[secret]" if tok.strip(".,!?\"'") in mine else tok for tok in raw.split(" "))
            counts["typed authentication argument replaced"] += 1
            if gives_secret(mine, raw):
                counts["authentication dropped"] += 1
                counts["password typed as text dropped"] += 1
                dropped[row] = "typed authentication argument"
                continue
        text = redact(clean(hide_secrets(redact(raw, counts), counts)), counts)
        if not text:
            dropped[row] = "empty"
            continue
        target = r.target if isinstance(r.target, str) and r.target else "unknown"
        ch = "chat" if r.channel == "chat" else f"msg>{redact(target, counts)}"
        speaker = redact(r.speaker, counts)
        key = (speaker.lower(), ch, text)
        prev = last_copy.get(key)
        if prev and sec - prev[1] <= FOLD_SECONDS:
            prev[0]["repeats"] += 1
            prev[0]["until"] = sec
            last_copy[key] = (prev[0], sec)
            folded[row] = prev[0]["row"]
            counts["repeats folded"] += 1
            continue
        item = dict(sec=sec, row=row, ch=ch, speaker=speaker, text=text, repeats=1, until=sec)
        kept.append(item)
        last_copy[key] = (item, sec)
    lines = []
    for it in kept:
        rep = f" [sent {it['repeats']} times until {hms(it['until'])}]" if it["repeats"] > 1 else ""
        lines.append((it["sec"], f"L{it['row']} {hms(it['sec'])} {it['ch']} {it['speaker']}: {it['text']}{rep}"))
    return lines, [it["row"] for it in kept], folded, dropped


def creayun_rows(messages, units=None):
    """The rows of the Creayun message table, sorted by server, day, time and row, indexed by row; only the
    (server, day) `units` when given."""
    import pandas as pd
    df = pd.read_parquet(messages, columns=["row", "ts", "server", "channel", "speaker", "target", "text"])
    df["day"] = df.ts.str[:10]
    if units is not None:
        df = df[pd.MultiIndex.from_frame(df[["server", "day"]]).isin(pd.MultiIndex.from_tuples(sorted(units)))]
    return df.sort_values(["server", "day", "ts", "row"], kind="stable").set_index("row")


def creayun_transcripts(cfg, units=None):
    """{stem: transcript} of every Creayun server-day (or of `units`), each transcript with its parts, rows, folds and
    drops."""
    d = cfg["transcript"]
    secrets, seen = creayun_secrets(cfg.path(d["exports"]))
    log.info("authentication commands in the exports: %d; words kept for the leak test: %d from %d names", seen,
             sum(len(v) for v in secrets.values()), len(secrets))
    out = {}
    for (server, day), rows in creayun_rows(cfg.path(d["messages"]), units).groupby(["server", "day"], sort=True):
        counts = collections.Counter()
        lines, ids, folded, dropped = creayun_lines(rows, secrets, counts)
        if not lines:
            continue
        stem = f"creayun-{server}-{day}"
        out[stem] = dict(stem=stem, server=server, day=day, counts=counts, parts=split(lines, creayun_header(stem, server, day)),
                         ids=[f"L{x}" for x in ids], rows=[int(x) for x in ids],
                         folded={str(k): int(v) for k, v in folded.items()},
                         dropped={str(k): v for k, v in dropped.items()})
    return out


# ---------------------------------------------------------------- batches

def batch_dir(cfg, name):
    return cfg.path(cfg["transcript"]["batches"]) / name


def load_manifest(folder):
    p = folder / "manifest.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def part_files(folder):
    return sorted((folder / "parts").glob("*.part*of*.txt"))


def transcript_of(stem):
    return stem.split(".part")[0]


def built_parts(d):
    """{file name: text} of a built transcript."""
    return {f"{d['stem']}.part{k}of{len(d['parts'])}.txt": text for k, (text, _, _) in enumerate(d["parts"], 1)}


def part_bytes(text):
    """A part file's bytes: UTF-8 with CRLF line ends on every platform. Readers read it as text, which turns them
    back into LF."""
    return text.replace("\n", "\r\n").encode("utf-8")


def manifest_entry(d, entry=None):
    """A transcript's manifest entry, updating `entry` (keeping its order and any field the build does not make)."""
    entry = dict(entry or {})
    entry.setdefault("transcript", d["stem"])
    parts = [dict(file=f"{d['stem']}.part{k}of{len(d['parts'])}.txt", tokens=tokens(text), first=d["ids"][s],
                  last=d["ids"][e - 1]) for k, (text, s, e) in enumerate(d["parts"], 1)]
    entry.update(day=d["day"], record_merged=d.get("record", False), lines=len(d["ids"]), parts=parts,
                 tokens=sum(p["tokens"] for p in parts), auth_dropped=d["counts"]["authentication dropped"],
                 addresses_in_text_replaced=d["counts"]["addresses in text replaced"])
    if "server" in d:
        entry["server"] = d["server"]
        entry["secrets_replaced"] = {k: v for k, v in d["counts"].items() if k not in (
            "authentication dropped", "addresses in text replaced", "repeats folded", "password typed as text dropped")}
        entry.update(rows=d["rows"], folded=d["folded"], dropped=d["dropped"])
    return entry


def body_ids(text):
    """The ids of a part's body: its lines after the overlap's end marker (every line of a first part)."""
    lines = text.splitlines()
    mark = next((i for i, ln in enumerate(lines) if PART_MARK.match(ln)), -1)
    return [m.group(1) for ln in lines[mark + 1:] if (m := LINE_ID.match(ln))]


def saved_bodies(folder, entry):
    """The body ids of a manifest entry's saved parts, in part order, or None when the batch holds none of them."""
    files = [folder / "parts" / x["file"] for x in entry.get("parts") or []]
    if not files or not all(f.exists() for f in files):
        return None
    return [body_ids(f.read_text(encoding="utf-8")) for f in files]


def build_batch(cfg, manifest, folder=None):
    """[(manifest entry, transcript)] of every transcript the manifest names, built from its sources; a server-log transcript whose
    parts `folder` holds is cut where they were cut. A source that holds no transcript line gives no transcript (None)."""
    if manifest.get("kind") == "creayun":
        built = creayun_transcripts(cfg, {(m["server"], m["day"]) for m in manifest["transcripts"]})
        return [(m, built.get(m["transcript"])) for m in manifest["transcripts"]]
    logs, record = cfg.path(cfg["transcript"]["logs"]), cfg.path(cfg["transcript"]["record"])
    run_of = {f"{src}/{f}": r for src in {m["source"].split("/")[0] for m in manifest["transcripts"]}
              for f, r in runs(logs / src).items()}
    tails = {}

    def carry(source):
        """(the repeat keys and held alerts open at the end of the file before `source` in its run, the player state
        there), in `source`'s seconds; (None, None) for a file that starts a run."""
        before = run_of[source][2]
        if before is None:
            return None, None
        before = f"{source.split('/')[0]}/{before}"
        if before not in tails:
            d = server_log_transcript(logs / before, record, None, *carry(before))
            tails[before] = (d["day"], d["tail"], d["state"])
        day, tail, state = tails[before]
        shift = (date.fromisoformat(file_day(logs / source)) - date.fromisoformat(day)).days * 86400
        return ({k: (st, rid, t - shift) for k, (st, rid, t) in tail.items()
                 if k[0] is None or t - shift >= -FOLD_SECONDS}, carried_state(state, shift))

    out = []
    for m in manifest["transcripts"]:
        d = server_log_transcript(logs / m["source"], record, saved_bodies(folder, m) if folder else None, *carry(m["source"]))
        tails[m["source"]] = (d["day"], d["tail"], d["state"])
        if m["transcript"] and d["stem"] != m["transcript"]:
            raise SystemExit(f"{m['source']} builds as {d['stem']}; the manifest names it {m['transcript']}")
        out.append((m, d if d["parts"] else None))
    return out


def new_manifest(cfg, args):
    if args.creayun:
        return {"kind": "creayun", "transcripts": [{"transcript": s, "source": f"creayun-messages.parquet {d['server']} "
                                                 f"{d['day']}", "server": d["server"], "day": d["day"]}
                                                for s, d in creayun_transcripts(cfg).items()]}
    logs = cfg.path(cfg["transcript"]["logs"])
    files = sorted({p for g in args.files for p in logs.glob(g) if p.is_file()
                    and (DATED.match(p.name) or p.name == "latest.log")})
    if not files:
        raise SystemExit(f"no log file under {logs} matches {' '.join(args.files)}")
    return {"kind": "server_log", "transcripts": [{"transcript": None, "source": p.relative_to(logs).as_posix()} for p in files]}


# ---------------------------------------------------------------- the check

# Secret shapes searched in a Creayun batch, which has no server log to take typed passwords from per transcript.
# Placeholders the builder writes ([secret], [email], [id]) do not match.
SHAPES = {
    "e-mail address": re.compile(r"(?i)[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[a-z]{2,}"),
    "key in a link": re.compile(r"(?i)[?&#;](?:key|api_?key|apikey|token|access_?token|auth|sig|signature|secret|"
                                r"password|pass|pwd|passcode|code|session|sid|ticket)=(?!\[secret\])[^&\s#]+"),
    "webhook token": re.compile(r"(?i)/api/webhooks/\d+/"),
    "Discord token": re.compile(r"[MNO][\w-]{23,27}\.[\w-]{6}\.[\w-]{27,40}"),
    "JSON web token": re.compile(r"eyJ[\w-]{10,}\.[\w-]{10,}"),
    "API key": re.compile(r"(?:sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{30,}|xox[abprs]-[\w-]{10,}|AKIA[0-9A-Z]{16}|"
                          r"AIza[\w-]{30,})"),
    "password stated in a sentence": re.compile(r"(?i)\b(?:pass(?:word)?s?|pw|pwd|passcode)\s*(?:is|=|:|was)\s*"
                                                r"(?!\[secret\])[^\s\[]"),
    "card-shaped number": re.compile(r"(?<![\d-])\d(?:[ -]?\d){14,18}(?!\d)"),
    "long id in a link": re.compile(r"(?i)(?:(?:https?|wss?|ftp|file)://|www\.)\S*?(?<![A-Za-z0-9])(?=[A-Za-z0-9_-]*\d)"
                                    r"(?=[A-Za-z0-9_-]*[A-Za-z])[A-Za-z0-9_-]{20,}"),
    "key-like string": re.compile(r"(?<![\w/+=-])(?=[\w+/=-]*\d)(?=[\w+/=-]*[a-z])(?=[\w+/=-]*[A-Z])[\w+/=-]{32,}"),
}


def check_part(text, secrets, shapes):
    """(problems, mentions) of one part. A problem is an address, an authentication command as a line's text, a word
    the line's speaker typed to an authentication command (`secrets`, keyed by lower name), or with `shapes` a secret
    shape. Each problem names its line id and kind, never the text. Mentions are lines that name an authentication
    command in a sentence ("do /shib link"); they stay."""
    problems, mentions = [], 0
    for line in text.splitlines():
        lid = line.split(" ", 1)[0]
        if IPV4.search(line):
            problems.append(f"IPv4 address in {lid}")
        if IPV6.search(line):
            problems.append(f"IPv6-shaped token in {lid}")
        m = EVENT.match(line)
        if not m:
            continue
        speaker, body = m.group(3), m.group(4)
        if AUTH_START.search(body):
            problems.append(f"an authentication command as the text of {lid}")
        elif AUTH_MENTION.search(body):
            mentions += 1
        for s in secrets.get(speaker.rsplit("=", 1)[-1].lower(), ()):
            if re.search(r"(?<!\w)" + re.escape(s) + r"(?!\w)", body):
                problems.append(f"its speaker's authentication argument ({len(s)} chars) appears in {lid}")
        if shapes:
            for k, rx in SHAPES.items():
                for hit in rx.finditer(body):
                    if k != "card-shaped number" or card_number(hit.group(0)):
                        problems.append(f"{k} in {lid}")
    return problems, mentions


class Checker:
    """The check of one batch's parts, with the authentication words of each transcript's source read once."""

    def __init__(self, cfg, manifest):
        self.cfg, self.creayun = cfg, manifest.get("kind") == "creayun"
        self.source = {m["transcript"]: m["source"] for m in manifest["transcripts"]}
        self.cache = {}

    def secrets(self, transcript):
        key = "creayun" if self.creayun else self.source[transcript]
        if key not in self.cache:
            if self.creayun:
                self.cache[key] = creayun_secrets(self.cfg.path(self.cfg["transcript"]["exports"]))[0]
            else:
                self.cache[key] = typed_secrets(self.cfg.path(self.cfg["transcript"]["logs"]) / self.source[transcript])
        return self.cache[key]

    def part(self, name, text):
        return check_part(text, self.secrets(transcript_of(name)), self.creayun)


def check_batch(cfg, folder, manifest):
    checker = Checker(cfg, manifest)
    files = part_files(folder)
    problems, mentions = [], 0
    for p in files:
        found, m = checker.part(p.name, p.read_text(encoding="utf-8"))
        problems += [f"{p.name}: {x}" for x in found]
        mentions += m
    for x in problems[:50]:
        log.warning("%s", x)
    log.info("check: %d parts, %d problems; %d lines name an authentication command in a sentence (they stay): %s",
             len(files), len(problems), mentions, "PASS" if not problems else "FAIL")
    if problems:
        raise SystemExit(f"{folder.name}: {len(problems)} problems in its parts; no read may send them")


# ---------------------------------------------------------------- the command

def speakers_of(text):
    """{id: (channel, speaker)} of a part's body lines that have a speaker."""
    lines = text.splitlines()
    mark = next((i for i, ln in enumerate(lines) if PART_MARK.match(ln)), -1)
    return {m.group(1): (m.group(2), m.group(3)) for ln in lines[mark + 1:] if (m := EVENT.match(ln))}


def to_read_again(folder, manifest, want, differ):
    """The parts a rebuild changes for its reads: ({part file: body lines it gains}, {part file: body lines it still
    holds that show another speaker or addressee}, {transcript: lines no part holds any more}). A part that gains a line or
    shows a line's speaker or addressee otherwise is read again; one whose body changes otherwise keeps its answers."""
    old_ids, gained, lost, respoken = {}, {}, {}, {}
    for e in manifest["transcripts"]:
        for x in e.get("parts") or []:
            old_ids[x["file"]] = set(body_ids((folder / "parts" / x["file"]).read_text(encoding="utf-8")))
    was = collections.defaultdict(set)
    for f, ids in old_ids.items():
        was[f.split(".part")[0]] |= ids
    now = collections.defaultdict(set)
    for f, text in want.items():
        body = body_ids(text)
        now[f.split(".part")[0]] |= set(body)
        new = [x for x in body if x not in was[f.split(".part")[0]]]
        if new:
            gained[f] = new
    for d, ids in was.items():
        if ids - now[d]:
            lost[d] = sorted(ids - now[d], key=lambda x: (x[0], int(x[1:])))
    for f in differ:
        saved = folder / "parts" / f
        if saved.exists():
            old, new = speakers_of(saved.read_text(encoding="utf-8")), speakers_of(want[f])
            ids = [x for x in new if x in old and old[x] != new[x]]
            if ids:
                respoken[f] = ids
    return gained, respoken, lost


def rewrite(cfg, folder, manifest, entries, want, differ, name):
    """Writes the rebuilt parts over a batch whose reads answered the saved ones. The saved manifest and every saved
    part that differs go to [transcript] history/NAME/<batch>/; so do every read's answers to a part to read again
    (to_read_again: its body gains a line, or a line in it shows another speaker or addressee), which that read answers
    again (`uv run read`). Writes rewrite.json there: the parts rewritten, the parts to read again with the lines they
    gain and the lines that show another speaker, and the lines no part holds any more."""
    import shutil
    hist = cfg.path(cfg["transcript"]["history"]) / name / folder.name
    if hist.exists():
        raise SystemExit(f"{hist} exists; name another history folder")
    gained, respoken, lost = to_read_again(folder, manifest, want, differ)
    again = sorted(set(gained) | set(respoken))
    reads = sorted(p for p in (folder / "reads").iterdir() if p.is_dir()) if (folder / "reads").exists() else []
    moved = collections.defaultdict(list)
    (hist / "parts").mkdir(parents=True)
    shutil.copy2(folder / "manifest.json", hist / "manifest.json")
    for f in differ:
        shutil.move(folder / "parts" / f, hist / "parts" / f)
    for f in again:
        for r in reads:
            a = r / f.replace(".txt", ".json")
            if a.exists():
                (hist / "reads" / r.name).mkdir(parents=True, exist_ok=True)
                shutil.move(a, hist / "reads" / r.name / a.name)
                moved[r.name].append(a.stem)
    for f in differ:
        (folder / "parts" / f).write_bytes(part_bytes(want[f]))
    manifest["transcripts"] = entries
    write_text(folder / "manifest.json", json.dumps(manifest, indent=1))
    report = {"parts_rewritten": differ,
              "parts_to_read_again": {f[:-4]: {"gained": gained.get(f, []), "speaker": respoken.get(f, [])}
                                      for f in again},
              "answers_moved": dict(moved), "lines_no_longer_built": lost}
    write_text(hist / "rewrite.json", json.dumps(report, indent=1))
    log.info("rewrote %d parts of %s (the saved ones are in %s); %d parts to read again (%d gain %d lines, %d show "
             "another speaker on %d lines), their answers moved there (%s); %d lines in %d transcripts are no longer built",
             len(differ), folder.name, hist, len(again), len(gained), sum(len(v) for v in gained.values()),
             len(respoken), sum(len(v) for v in respoken.values()),
             ", ".join(f"{k} {len(v)}" for k, v in moved.items()) or "none", sum(len(v) for v in lost.values()),
             len(lost))


def body(cfg, args):
    folder = batch_dir(cfg, args.batch)
    manifest = load_manifest(folder)
    if args.check:
        if manifest is None:
            raise SystemExit(f"{folder} has no manifest.json")
        check_batch(cfg, folder, manifest)
        return
    if args.files or args.creayun:
        if manifest is not None:
            raise SystemExit(f"{folder} already holds a batch; `uv run transcript {args.batch}` rebuilds it")
        manifest = new_manifest(cfg, args)
    elif manifest is None:
        raise SystemExit(f"{folder} has no manifest.json; name its sources with --files or --creayun")
    entries, want = [], {}
    for m, d in build_batch(cfg, manifest, folder):
        if d is None:
            if m["transcript"] is None:
                log.info("%s holds no transcript line; left out", m["source"])
                continue
            raise SystemExit(f"{m['transcript']} builds to no lines")
        entries.append(manifest_entry(d, {**m, "transcript": d["stem"]}))
        want.update(built_parts(d))
    have = {p.name: p for p in part_files(folder)}
    same = [n for n in want if n in have and have[n].read_bytes() == part_bytes(want[n])]
    differ = sorted(set(want) - set(same))
    extra = sorted(set(have) - set(want))
    log.info("%s: %d transcripts, %d parts built; %d of them equal the saved parts, %d differ or are new, %d saved parts "
             "are not built", args.batch, len(entries), len(want), len(same), len(differ), len(extra))
    if have:
        old = manifest["transcripts"]
        changed = [e["transcript"] for o, e in zip(old, entries) if o != e]
        log.info("manifest: %d of %d transcript entries differ from the saved ones%s", len(changed) + abs(len(old) - len(entries)),
                 len(entries), f", first {changed[:3]}" if changed else "")
    if differ or extra:
        answered = [r for r in (folder / "reads").glob("*/*.json")] if (folder / "reads").exists() else []
        if answered and args.rewrite and not extra and not set(differ) - set(have):
            rewrite(cfg, folder, manifest, entries, want, differ, args.rewrite)
        elif answered:
            gained, respoken, _ = to_read_again(folder, manifest, want, [n for n in differ if n in have])
            log.info("--rewrite would read %d parts again: %d gain %d lines, %d show another speaker or addressee on "
                     "%d lines", len(set(gained) | set(respoken)), len(gained), sum(len(v) for v in gained.values()),
                     len(respoken), sum(len(v) for v in respoken.values()))
            check_batch(cfg, folder, manifest)
            raise SystemExit(f"{len(differ) + len(extra)} parts differ from the saved ones, and reads have answered "
                             f"the saved ones, so the saved parts stay (--rewrite writes them); first: "
                             f"{(differ + extra)[:5]}")
        else:
            (folder / "parts").mkdir(parents=True, exist_ok=True)
            for n in extra:
                have[n].unlink()
            for n in differ:
                (folder / "parts" / n).write_bytes(part_bytes(want[n]))
            manifest["transcripts"] = entries
            write_text(folder / "manifest.json", json.dumps(manifest, indent=1))
            log.info("wrote %d parts and manifest.json to %s", len(differ), folder)
    check_batch(cfg, folder, manifest)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("batch", help="the batch: a folder under [transcript] batches")
    p.add_argument("--files", nargs="+", metavar="GLOB", help="a new server-log batch from these log files, relative to "
                   "[transcript] logs (tf/2026-11-12-1.log.gz, \"tf/2026-11-1*.log.gz\")")
    p.add_argument("--creayun", action="store_true", help="a new Creayun batch: every server and day of the table")
    p.add_argument("--check", action="store_true", help="only check the saved parts")
    p.add_argument("--rewrite", metavar="NAME", help="write the rebuilt parts over parts reads have answered, the saved "
                   "ones and the answers to parts that gain lines or show a line's speaker or addressee otherwise kept "
                   "in [transcript] history/NAME")
    run("transcript", body, p)
