"""uv run ingest: read every configured source into one message table, one row per message, keyed by a stable id.

Every run reads all source files again, so the table depends only on which files are present, never on the order they
arrived in or on earlier runs. A message's id is built from what the message is (message_id). Messages already in the
table keep their row and id; new ones are appended, one batch per source. A stored message that the files no longer
produce stops the run before anything is written.

`--folder` (default: [ingest] folder) is the folder that holds messages.parquet and state/ingest.json.
"""
import argparse
import collections
import gzip
import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import pyarrow as pa
import pyarrow.parquet as pq

from .core import changed_inputs, fingerprint, log, progress, run, write_parquet, write_text

COLUMNS = ["id", "source", "server", "ts", "channel", "conv", "speaker", "target", "text", "repeats"]

# Commands read from the logs, by their last part (/essentials:msg is msg).
PRIVATE = {"msg", "tell", "w", "whisper", "m", "pm", "emsg", "epm", "etell", "ew", "ewhisper"}
REPLY = {"r", "reply", "er", "ereply"}
MAIL = {"mail", "email"}
ME = {"me", "eme"}
NICK = {"nick", "enick", "nickname", "enickname", "nickmm", "nickminimessage"}
SUDO = {"sudo", "esudo"}
# Read only to be counted as dropped: the module does not capture staff chat, /say, or the guild commands (guild chat
# is read from Module-Guilds' console line instead, which also covers the chat toggle).
DROPPED = {"o": "staff chat", "ac": "staff chat", "adminchat": "staff chat", "say": "say", "g": "guild command",
           "gc": "guild command", "guild": "guild command"}

# Server output that lands in chat text: a "* <command> ..." echo of a command run for a player (/sudo, rcon), and a
# message that is only target selectors.
ECHO = re.compile(r"^\* (?:c:|/|[\w.-]+:[\w.-]+(?:\s|$)|(?:me|say|bc|broadcast\w*|suicide|ekill|kill|deop|op|ban|"
                  r"pardon|kick|mute|msg|tell|tp|gamemode|gm[acs]?|gmsp|gamerule|nuke|skin|give|effect|clear|"
                  r"execute|summon|sudo|fly|god|heal)(?:\s|$))", re.I)
SELECTOR = re.compile(r"^(?:\s*@[aeprsn](?:\[[^\]]*\])?)+\s*$")
COLOUR = re.compile(r"§.?", re.S)


def message_id(server, ts, speaker, channel, target, raw, k):
    """A message's id: its server, time, author account, channel, private-message target, raw text, and its occurrence
    number k among identical such tuples in its file, in file order. Not the file name (latest.log rotates into a dated
    file) and not the conversation key, so the same message read from two files gets the same id."""
    key = "\x1f".join([server, ts, speaker.lower(), channel, (target or "").lower(), raw, str(k)])
    return hashlib.blake2b(key.encode("utf-8"), digest_size=16).hexdigest()


def strip_colour(text):
    """Text as the module receives it: Minecraft components carry colour as style, never as § codes."""
    return COLOUR.sub("", text)


def server_output(text):
    return "server output" if ECHO.match(text) or SELECTOR.match(text) else None


# ---------------------------------------------------------------- account names

LOOK = str.maketrans("ᴀʙᴄᴅᴇꜰɢʜɪᴊᴋʟᴍɴᴏᴘǫʀꜱᴛᴜᴠᴡʏᴢаеорсухкмтвніјѕԁɡαορινтκε",
                     "abcdefghijklmnopqrstuvwyzaeopcyxkmtbhijsdgaopivtke")
CODES = re.compile(r"[§&](?:#[0-9a-f]{6}|[0-9a-fk-orx])", re.I)
ARG_TAG = re.compile(r"<[^<>]*:[^<>]*>|<#[0-9a-f]{6}>", re.I)
BRACKETS = re.compile(r"\[[^\]]*\]")
EDGE = re.compile(r"^\W+|\W+$")


def fold(name):
    """A name compared across lookalikes: NFKC (fullwidth, mathematical letters), small capitals and Cyrillic or Greek
    lookalikes as Latin, lower case, letters, digits and _ only."""
    return re.sub(r"[^a-z0-9_]", "", unicodedata.normalize("NFKC", name).casefold().translate(LOOK))


def plain_name(text):
    """A display name or nickname without colour codes, MiniMessage tags or [tags]."""
    return BRACKETS.sub(" ", ARG_TAG.sub("", CODES.sub("", text))).strip()


class Players:
    """Who is on the server as the log goes, their nicknames, private-message partners and pending /sudo. Every answer
    uses only lines read so far, so a file that grows never changes what its earlier lines resolved to."""

    def __init__(self):
        self.online = {}      # lower account -> account
        self.known = {}       # every account seen
        self.folded = collections.defaultdict(set)   # folded account name -> accounts
        self.nicks = {}       # lower account -> its folded nickname, from /nick
        self.nicked = collections.defaultdict(set)   # folded nickname -> accounts that have it
        self.shown = {}       # lower account -> folded name it last spoke under
        self.together = {}    # folded display -> lower accounts online at every line it spoke
        self.partner = {}     # lower account -> account /r answers
        self.sudo = {}        # lower account -> (issuer, second, kind)
        self.speakers = collections.Counter()   # how chat lines' display names resolved
        self.targets = collections.Counter()    # how typed player names resolved

    def seen(self, account):
        self.online[account.lower()] = account
        if account.lower() not in self.known:
            self.known[account.lower()] = account
            self.folded[fold(account)].add(account)

    def nick(self, account, folded):
        """Records `account`'s /nick (folded; "" for off)."""
        self.nicked[self.nicks.pop(account.lower(), "")].discard(account)
        if folded:
            self.nicks[account.lower()] = folded
            self.nicked[folded].add(account)

    def speaker(self, display):
        """(account, how) of a chat line's display name: tags, prefixes, a nickname or lookalike letters. In order: an
        online account's name shown as is (EssentialsX marks a nickname with ~); the one online player whose /nick
        is the shown name; an online account's name anywhere in the display, [tags] included; the one online name the
        shown name matches after folding lookalikes; the one account, online or not, with that /nick or that folded
        name; a known account's name; the one player online at every line this display spoke. An unresolved display
        keeps its shown name."""
        plain = plain_name(display)
        tokens = [t for t in plain.split() if re.search(r"\w", t)] or display.split()
        last = tokens[-1] if tokens else display.strip()
        name = EDGE.sub("", last)
        low = name.lower()
        if not last.startswith("~") and low in self.online:
            return self.online[low], "name"
        f = fold(name)
        nicked = [a for k, a in self.online.items() if f and self.nicks.get(k) == f]
        if len(nicked) == 1:
            return nicked[0], "nickname"
        for token in reversed(re.split(r"[\s\[\]()<>|:@~«»]+", CODES.sub("", display))):
            if token.lower() in self.online:
                return self.online[token.lower()], "name"
        if f:
            for test in (lambda a: a == f, lambda a: a.startswith(f) or f.startswith(a),
                         lambda a: len(f) >= 3 and (f in a or a in f)):
                hits = [acc for acc in self.online.values() if test(fold(acc))]
                if len(hits) == 1:
                    return hits[0], "lookalike"
                if hits:
                    break
        if len(self.nicked.get(f, ())) == 1:
            return next(iter(self.nicked[f])), "nickname"
        if len(self.folded.get(f, ())) == 1:
            return next(iter(self.folded[f])), "lookalike"
        if low in self.known:
            return self.known[low], "name"
        key = fold(plain) or plain
        was = self.together.get(key)
        now = set(self.online)
        both = now if was is None else (was & now or now)
        self.together[key] = both
        if len(both) == 1:
            return self.online[next(iter(both))], "online alone"
        return name or display.strip(), "unresolved"

    def account(self, typed):
        """The account a typed player name means, as EssentialsX matches it: an exact name, else the one online name
        it starts, else the one online player whose name or nickname contains it, else an exact known name."""
        t = typed.lower().lstrip("~")
        if t in self.online:
            self.targets["exact"] += 1
            return self.online[t]
        hits = [a for k, a in self.online.items() if k.startswith(t)]
        if len(hits) == 1:
            self.targets["start of an online name"] += 1
            return hits[0]
        if not hits:
            f = fold(typed)
            hits = [a for k, a in self.online.items() if f and (f in fold(a) or f in self.shown.get(k, ""))]
            if len(hits) == 1:
                self.targets["online name or nickname"] += 1
                return hits[0]
        if t in self.known:
            self.targets["known offline name"] += 1
            return self.known[t]
        self.targets["ambiguous" if hits else "unresolved"] += 1
        return None

    def credit(self, account, second, kind, window):
        """The /sudo issuer a forced chat line or command of `account` belongs to, or None."""
        pending = self.sudo.get(account.lower())
        if pending and pending[2] == kind and 0 <= second - pending[1] <= window:
            del self.sudo[account.lower()]
            self.speakers["sudo"] += 1
            return pending[0]
        return None


def command(players, base, speaker, cmd, rest, second, channel_conv, msg_conv):
    """The record of a command line, or None when the command carries no message."""
    rest = rest or ""
    if cmd in DROPPED:
        return dict(base, channel="/" + cmd, speaker=speaker, target=None, raw=rest, conv=None, drop=DROPPED[cmd]) \
            if rest.strip() else None
    if cmd in SUDO:
        who, _, forced = rest.partition(" ")
        kind = "chat" if forced.startswith("c:") and not forced.startswith("c:/") else "command"
        targets = list(players.online.values()) if who in ("*", "**") else [players.account(who)]
        for t in targets:
            if t:
                players.sudo[t.lower()] = (speaker, second, kind)
        return None
    if cmd in PRIVATE or cmd in REPLY or cmd in MAIL:
        if cmd in MAIL:
            parts = rest.split(" ", 2)
            if len(parts) < 3 or parts[0].lower() != "send" or not parts[2].strip():
                return None
            typed, text = parts[1], parts[2]
        elif cmd in REPLY:
            typed, text = None, rest
        else:
            typed, _, text = rest.partition(" ")
        if not text.strip():
            return None
        if typed is None:
            target = players.partner.get(speaker.lower())
            if not target:
                return dict(base, channel="msg", speaker=speaker, target=None, raw=text, conv=None,
                            drop="reply without partner")
        else:
            target = players.account(typed) or typed
        players.partner[speaker.lower()] = target
        if cmd not in MAIL:
            players.partner[target.lower()] = speaker
        return dict(base, channel="msg", speaker=speaker, target=target, raw=text,
                    conv=msg_conv + "|".join(sorted((speaker.lower(), target.lower()))))
    if cmd in ME and rest.strip():
        return dict(base, channel="chat", speaker=speaker, target=None, raw=rest, conv=channel_conv)
    if cmd in NICK and rest.strip():
        first, _, more = rest.partition(" ")
        owner, nick = (players.online[first.lower()], more) if more.strip() and first.lower() in players.online \
            else (speaker, rest)
        if nick.strip().lower() == "off":
            players.nick(owner, "")
            return None
        shown = plain_name(re.sub(r"<[^<>]*>", "", nick)).split()   # /nickmm takes MiniMessage
        players.nick(owner, fold(shown[-1]) if shown else "")
        return dict(base, channel="nick", speaker=speaker, target=None, raw=nick, conv=channel_conv)
    return None


# ---------------------------------------------------------------- Paper logs

LINE = re.compile(r"^\[(\d\d):(\d\d):(\d\d)\] \[([^\]]*)\]: (.*)$")
DATED = re.compile(r"^(\d{4}-\d\d-\d\d)-(\d+)\.log(?:\.gz)?$")
CHAT = re.compile(r"^(.*?) » (.*)$", re.S)
VANILLA = re.compile(r"^<([^<>\s]+)> (.*)$", re.S)
GUILD = re.compile(r"^\[GUILD - (.+?):([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\] (\S+) (.*)$",
                   re.S)
CMD = re.compile(r"^(\S+) issued server command: /(\S+)(?: (.*))?$", re.S)
# Staff chat: "[ADMIN] <name> [<rank>]: text" or "[ADMIN] <name>: text". A public chat line can carry an [ADMIN] tag.
ADMIN = re.compile(r"^\[ADMIN\] ([\w.]+)(?: \[[^\]]*\])?: (.*)$", re.S)
LOGIN = re.compile(r"^(\S+?)\[/[^\]]*\] logged in with entity id")
LEAVE = re.compile(r"^(\S+) (?:lost connection: |left the game$|left$)")
ISO = re.compile(r"\b(20\d\d-\d\d-\d\d)T(\d\d):(\d\d):(\d\d)(?:\.\d+)?Z")
BOOT = "Starting minecraft server"


@dataclass
class PaperFile:
    source: str
    path: object
    day: str
    first: str      # time of its first line, which orders files of the same day across folders
    rank: int       # the source's place in [ingest] sources
    index: float    # n of YYYY-MM-DD-n; latest.log comes after every dated file of its folder


def log_lines(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            yield line.rstrip("\r\n")


def seconds(m):
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))


def first_time(path):
    for line in log_lines(path):
        m = LINE.match(line)
        if m:
            return f"{m.group(1)}:{m.group(2)}:{m.group(3)}"
    return ""


def latest_day(path, dated):
    """The day latest.log covers; it has no dated name and its modification time is not used. Each line carries only
    the time of day (UTC). A full timestamp a plugin printed within two minutes of its line's own time gives the day
    (Module-Guilds prints them). Otherwise the rotation does: Paper rotates latest.log at midnight and at startup and
    names the rotated file by the day it covers, so latest.log covers the newest dated file's day when it starts with a
    startup that came after that file's last line, and the next day otherwise."""
    first, boot = None, False
    for n, line in enumerate(log_lines(path)):
        m = LINE.match(line)
        if not m:
            continue
        t = seconds(m)
        first = t if first is None else first
        boot = boot or (n < 300 and BOOT in m.group(5))
        for im in ISO.finditer(m.group(5)):
            if abs(int(im.group(2)) * 3600 + int(im.group(3)) * 60 + int(im.group(4)) - t) <= 120:
                return im.group(1)
    if not dated:
        raise SystemExit(f"{path}: no dated log file beside it and no full date in it, so its day is unknown")
    newest = max(dated, key=lambda f: (f.day, f.index))
    last = None
    for line in log_lines(newest.path):
        m = LINE.match(line)
        if m:
            last = seconds(m)
    if boot and first is not None and last is not None and first >= last:
        return newest.day
    return (date.fromisoformat(newest.day) + timedelta(days=1)).isoformat()


def paper_files(cfg, sources):
    """The log files of one Paper server's sources, oldest first."""
    files = []
    for src in sources:
        folder = cfg.path(src["path"])
        rank = cfg["ingest"]["sources"].index(src)
        dated = []
        for p in sorted(folder.iterdir()):
            m = DATED.match(p.name)
            if m:
                dated.append(PaperFile(src["name"], p, m.group(1), first_time(p), rank, int(m.group(2))))
            elif p.name != "latest.log":
                log.warning("%s: %s is not a Paper log name (YYYY-MM-DD-n.log.gz or latest.log); skipped",
                            src["name"], p.name)
        files += dated
        latest = folder / "latest.log"
        if latest.exists():
            files.append(PaperFile(src["name"], latest, latest_day(latest, dated), first_time(latest), rank,
                                   float("inf")))
    return sorted(files, key=lambda f: (f.day, f.first, f.rank, f.index))


def paper_records(server, files, advance):
    """Every chat-like record of one Paper server's files, in log order, with its id, author account and conversation;
    records the module does not capture carry a `drop` reason. Accounts come from the log: login lines, commands and
    guild lines name them; chat lines show display names, which Players resolves. A file with the same content as one
    already read (latest.log beside the dated file it rotated into) is read once: reading it again would replay its
    lines into a state that has moved on."""
    players = Players()
    counts = collections.Counter()
    ids, contents = set(), set()
    for f in files:
        lines = list(log_lines(f.path))
        digest = hashlib.blake2b("\n".join(lines).encode("utf-8", "surrogatepass")).digest()
        if digest in contents:
            log.info("%s: %s/%s has the same content as a file already read; skipped", server, f.source, f.path.name)
            advance()
            continue
        contents.add(digest)
        day = f.day   # Paper rotates at midnight, so a file holds one day
        base_day = date.fromisoformat(day).toordinal() * 86400
        for n, line in enumerate(lines):
            m = LINE.match(line)
            if not m:
                continue
            t = seconds(m)
            thread, msg = m.group(4), m.group(5)
            ts = f"{day}T{m.group(1)}:{m.group(2)}:{m.group(3)}"
            second = base_day + t
            chat_conv = f"{server}/chat/{day}"
            base = dict(source=f.source, server=server, ts=ts, second=second, file=(f.source, f.path.name), line=n,
                        display=None, target=None, drop=None)
            rec = None
            g = GUILD.match(msg)
            if g:
                players.seen(g.group(3))
                rec = dict(base, channel="guild", speaker=g.group(3), display=g.group(3), raw=g.group(4),
                           conv=f"{server}/guild/{g.group(2)}/{day}")
            elif thread.startswith("Async Chat Thread"):
                a = ADMIN.match(msg)
                if a:
                    rec = dict(base, channel="chat", speaker=a.group(1), display=a.group(1), raw=a.group(2),
                               conv=None, drop="staff chat")
                else:
                    c = CHAT.match(msg) or VANILLA.match(msg)
                    if c:
                        speaker, h = players.speaker(c.group(1))
                        forced = players.credit(speaker, second, "chat", 2)
                        if not forced:
                            players.speakers[h] += 1
                            players.shown[speaker.lower()] = fold(c.group(1).split()[-1]) if c.group(1).split() \
                                else ""
                        rec = dict(base, channel="chat", speaker=forced or speaker, display=c.group(1),
                                   how="sudo" if forced else h, raw=c.group(2), conv=chat_conv)
            elif thread.startswith("Server thread"):
                c = CMD.match(msg)
                if c:
                    players.seen(c.group(1))
                    issuer = players.credit(c.group(1), second, "command", 2) or c.group(1)
                    rec = command(players, dict(base, display=c.group(1)), issuer, c.group(2).lower().split(":")[-1],
                                  c.group(3), second, chat_conv, f"{server}/msg/")
                else:
                    lm = LOGIN.match(msg)
                    if lm:
                        players.seen(lm.group(1))
                    else:
                        lv = LEAVE.match(msg)
                        if lv:
                            players.online.pop(lv.group(1).lower(), None)
                        elif msg.startswith(BOOT):
                            players.online.clear()
                            players.sudo.clear()
            if rec is not None:
                rec["text"] = strip_colour(rec["raw"]).strip()
                yield identify(rec, counts, ids)
        advance()
    log.info("%s: chat speakers by %s", server, ", ".join(f"{k} {v:,}" for k, v in players.speakers.most_common()))
    log.info("%s: private message targets by %s", server,
             ", ".join(f"{k} {v:,}" for k, v in players.targets.most_common()))


def identify(rec, counts, ids):
    """Gives a record its id (counting identical tuples per file in `counts`) and marks server output, empty text and
    a message already read from another file (`ids`)."""
    if rec["drop"] is None:
        rec["drop"] = server_output(rec["text"]) or (None if rec["text"] else "empty")
    key = (rec["server"], rec["ts"], rec["speaker"].lower(), rec["channel"], (rec["target"] or "").lower(), rec["raw"])
    counts[(rec["file"], key)] += 1
    rec["id"] = message_id(*key[:5], rec["raw"], counts[(rec["file"], key)])
    if rec["id"] in ids and rec["drop"] is None:
        rec["drop"] = "same message in another file"
    ids.add(rec["id"])
    return rec


# ---------------------------------------------------------------- Discord exports (Creayun "FLAGGED" embeds)

HEADER = re.compile(r"^\[(\d+/\d+/\d+ \d+:\d+[\s\u202f][AP]M)\] (.+)$")  # Discord puts U+202F before AM/PM
SEGMENT = re.compile(r"`((?:[^`\\]|\\.)*)`", re.S)
UNESCAPE = re.compile(r"\\(.)", re.S)
TRAILER = re.compile(r"^\{\w+\}$")     # {Reactions}, {Embed}, {Attachments} after the fields


def flagged_blocks(path):
    """(block index, time, username, server, message field) per FLAGGED embed of an export. The message field is every
    line from "- Message: " to the end of the embed. IP fields are never read."""
    index = 0
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.read().split("\n")
    i = 0
    while i < len(lines):
        m = HEADER.match(lines[i])
        if not m or i + 1 >= len(lines) or lines[i + 1].strip() != "## FLAGGED":
            i += 1
            continue
        j = i + 2
        while j < len(lines) and not HEADER.match(lines[j]):
            j += 1
        fields, message = {}, None
        for k in range(i + 2, j):
            line = lines[k]
            if line.startswith("- Message: "):
                body = []
                for rest in lines[k:j]:
                    if TRAILER.match(rest.strip()):
                        break
                    body.append(rest)
                message = "\n".join(body)[len("- Message: "):].rstrip()
                break
            for name in ("Username", "Server"):
                if line.startswith(f"- {name}: "):
                    fields[name] = line[len(name) + 4:].strip().strip("`")
        if message is not None:
            yield index, m.group(1), UNESCAPE.sub(r"\1", fields.get("Username", "")), \
                UNESCAPE.sub(r"\1", fields.get("Server", "") or "unknown"), message
            index += 1
        i = j


def strings(message):
    """The texts of a message field: each `...` segment unescaped; a JSON string or component that the export split
    over several segments is joined again."""
    segs = [UNESCAPE.sub(r"\1", s) for s in SEGMENT.findall(message)] or [message.strip("`")]
    out, buf, joined = [], None, 0
    for s in segs:
        buf, joined = (s, 1) if buf is None else (buf + "\n" + s, joined + 1)
        if joined < 40 and unfinished(buf):
            continue
        out.append(buf)
        buf = None
    if buf is not None:
        out.append(buf)
    return out


def unfinished(s):
    """Whether s starts a JSON string or component that has not ended yet."""
    if s.startswith('"'):
        return len(s) < 2 or not s.endswith('"')
    if s.startswith(("{", "[")):
        return not json_text(s)[1] and s.count("{") + s.count("[") > s.count("}") + s.count("]")
    return False


def json_text(s):
    """(plain text, is JSON) of a string that may be a JSON string or text component."""
    if not s.startswith(('"', "{", "[")):
        return s, False
    for attempt in (s, s.replace("\n", "\\n")):
        try:
            return component(json.loads(attempt)), True
        except (ValueError, RecursionError):
            pass
    return s, False


def component(c):
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(component(x) for x in c)
    if isinstance(c, dict):
        return str(c.get("text", "")) + "".join(component(x) for x in c.get("extra", []))
    return ""


def classify(parts):
    """The channel of a flagged message's texts. The export does not say what was flagged, so the shape decides. A first
    text in § colours or a JSON component is an item's name (then its lore). Any other single text is chat, unless it is
    a JSON string over several lines (a book and quill's page). Four short texts are a sign's lines. A first text in
    JSON quotes starts a book and quill's pages; otherwise a title or name followed by a text longer than a lore line
    is a book, and a name with short lines is an item."""
    texts = [json_text(p) for p in parts]
    plain = [strip_colour(t).strip() for t, _ in texts]
    if parts[0].startswith(("§", "{", "[")) and (texts[0][1] or parts[0].startswith("§")):
        return "item"
    if len(parts) == 1 and not (texts[0][1] and "\n" in parts[0]):
        return "chat"
    if len(parts) == 4 and all(len(p) <= 20 for p in plain):
        return "sign"
    if parts[0].startswith('"') and texts[0][1]:
        return "book"
    return "book" if any(len(p) > 60 for p in plain[1:]) else "item"


def discord_records(source, network, files, advance):
    """Every flagged message of a source's exports, oldest first, with its id and conversation; records the module
    does not capture carry a `drop` reason."""
    counts, ids, records = collections.Counter(), set(), []
    for rank, p in enumerate(files):
        records += [dict(r, rank=rank) for r in export_records(source, network, p, counts, ids)]
        advance()
    return sorted(records, key=lambda r: (r["second"], r["rank"], r["line"]))


def export_records(source, network, path, counts, ids):
    """The records of one export, in its order. Private-message names resolve against this export alone, so adding
    another export never changes them; a player counts as online for 10 minutes after their last message."""
    players = Players()
    recent = collections.OrderedDict()   # lower account -> (second, account), oldest first
    for index, when, username, sub, message in flagged_blocks(path):
        if not username:
            continue
        stamp = datetime.strptime(when.replace("\u202f", " "), "%m/%d/%Y %I:%M %p")
        name = path.name
        ts = stamp.isoformat()
        second = int((stamp - datetime(1970, 1, 1)).total_seconds())
        server, day = f"{network}/{sub}", ts[:10]
        recent.pop(username.lower(), None)
        recent[username.lower()] = (second, username)
        while second - next(iter(recent.values()))[0] > 600:
            recent.popitem(last=False)
        players.known[username.lower()] = username
        base = dict(source=source, server=server, ts=ts, second=second, file=(source, name), line=index,
                    display=None, target=None, drop=None)
        parts = strings(message)
        raw = "\n".join(parts)
        channel = classify(parts)
        cm = re.match(r"^/(\S+)\s*(.*)$", raw, re.S) if channel == "chat" else None
        if cm:
            players.online = {k: v[1] for k, v in recent.items()}
            speaker = players.credit(username, second, "command", 60) or username
            rec = command(players, base, speaker, cm.group(1).lower().split(":")[-1], cm.group(2), second,
                          f"{server}/chat/{day}", f"{network}/msg/")
            if rec is None:
                continue
            rec["text"] = strip_colour(rec["raw"]).strip()
        elif channel == "chat":
            speaker = players.credit(username, second, "chat", 60) or username
            rec = dict(base, channel="chat", speaker=speaker, raw=raw, text=strip_colour(raw).strip(),
                       conv=f"{server}/chat/{day}")
        else:
            text = "\n".join(t for t in (strip_colour(json_text(p)[0]).strip() for p in parts) if t)
            rec = dict(base, channel=channel, speaker=username, raw=raw, text=text, conv=f"{server}/{channel}/{day}")
        yield identify(rec, counts, ids)
    log.info("%s: private message targets by %s", path.name,
             ", ".join(f"{k} {v:,}" for k, v in players.targets.most_common()))


# ---------------------------------------------------------------- raids and floods

def settle(records, per_minute):
    """Marks raid accounts and flood repeats on one server group's records (in order) and returns the kept ones.

    Raid: an account whose every message falls in a minute with at least `per_minute` messages on its server (or the
    3 minutes around one). Flood: the deployed duplicate rule; a player's exact repeat in the same conversation within
    60 seconds of their previous copy is one message, and `repeats` counts the copies."""
    live = [r for r in records if r["drop"] is None]
    per_min = collections.Counter((r["server"], r["second"] // 60) for r in live)
    hot = {k for k, c in per_min.items() if c >= per_minute}
    window = {(s, m + o) for s, m in hot for o in range(-3, 4)}
    outside = {(r["server"], r["speaker"].lower()) for r in live if (r["server"], r["second"] // 60) not in window}
    for r in live:
        if (r["server"], r["speaker"].lower()) not in outside:
            r["drop"] = "raid account"
    last = {}
    kept = []
    for r in live:
        if r["drop"]:
            continue
        key = (r["speaker"].lower(), r["conv"])
        rep = last.get(key)
        if rep and rep[0]["raw"] == r["raw"] and r["second"] - rep[1] <= 60:
            rep[0]["repeats"] += 1
            r["drop"], r["into"] = "flood repeat", rep[0]["id"]
            last[key] = (rep[0], r["second"])
            continue
        r["repeats"] = 1
        last[key] = (r, r["second"])
        kept.append(r)
    return kept


# ---------------------------------------------------------------- the step

def source_files(cfg, src):
    folder = cfg.path(src["path"])
    if src["kind"] == "paper":
        return sorted(p for p in folder.iterdir() if p.is_file())
    if src["kind"] == "discord":
        return sorted(p for p in folder.iterdir() if p.is_file() and "flagged" in p.name)
    raise SystemExit(f"source {src['name']}: unknown kind {src['kind']!r}")


def read_sources(cfg):
    """(every record of every source, the kept messages in order, grouped by server)."""
    sources = cfg["ingest"]["sources"]
    per_minute = cfg["ingest"]["raid_messages_per_minute"]
    servers = {}
    for src in sources:
        servers.setdefault((src["kind"], src["server"]), []).append(src)
    everything, kept = [], []
    for (kind, server), group in servers.items():
        if kind == "paper":
            files = paper_files(cfg, group)
            with progress(f"{server}: reading", len(files), "files") as advance:
                records = list(paper_records(server, files, advance))
        else:
            if len(group) > 1:
                raise SystemExit(f"server {server}: one discord source per server")
            files = source_files(cfg, group[0])
            with progress(f"{server}: reading", len(files), "files") as advance:
                records = list(discord_records(group[0]["name"], server, files, advance))
        kept += settle(records, per_minute)
        everything += records
    return everything, kept


def file_prints(cfg):
    return {f"{s['name']}/{p.name}": fingerprint(p) for s in cfg["ingest"]["sources"] for p in source_files(cfg, s)}


def pending(cfg, state):
    """{source name: [files that are new or changed since the run that wrote `state`]}."""
    seen = (state or {}).get("files", {})
    return {s["name"]: [p.name for p in source_files(cfg, s) if seen.get(f"{s['name']}/{p.name}") != fingerprint(p)]
            for s in cfg["ingest"]["sources"]}


def body(cfg, args):
    folder = cfg.path(args.folder or cfg["ingest"]["folder"])
    out = folder / "messages.parquet"
    state_file = folder / "state" / "ingest.json"
    state = json.loads(state_file.read_text(encoding="utf-8")) if state_file.exists() else None
    if out.exists() and not state:
        raise SystemExit(f"{out} exists but {state_file} does not; it is not known what the table holds")
    if state and changed_inputs(cfg, state):
        raise SystemExit(f"{out} changed outside ingest; restore it, or delete it and {state_file} to rebuild")
    files = file_prints(cfg)
    if state and state["files"] == files:
        log.info("up to date: %s messages; no source file changed", f"{state['rows']:,}")
        return

    everything, kept = read_sources(cfg)
    fresh = {r["id"]: r for r in kept}
    stored = pq.read_table(out, columns=["id"]).column("id").to_pylist() if out.exists() else []
    gone = [i for i in stored if i not in fresh]
    if gone:
        raise SystemExit(f"{len(gone):,} stored messages are no longer produced by the source files (first: {gone[0]})."
                         " A source file was removed or changed other than by growing, or a latest.log was dated"
                         " differently from the dated file it rotated into. Restore the file, or delete the table"
                         " and its state to rebuild it.")
    held = set(stored)
    rows = [fresh[i] for i in stored]
    batches = list(state["batches"]) if state else []
    for src in cfg["ingest"]["sources"]:
        new = [r for r in kept if r["source"] == src["name"] and r["id"] not in held]
        if new:
            batches.append({"source": src["name"], "start": len(rows), "end": len(rows) + len(new),
                            "written": datetime.now().isoformat(timespec="seconds")})
            rows += new
    table = pa.table({c: [r[c] for r in rows] for c in COLUMNS},
                     schema=pa.schema([(c, pa.int32() if c == "repeats" else pa.string()) for c in COLUMNS]))
    write_parquet(table, out)
    held_as = str(out.relative_to(cfg.data)) if out.is_relative_to(cfg.data) else str(out)
    write_text(state_file, json.dumps({"finished": datetime.now().isoformat(timespec="seconds"),
                                       "inputs": {held_as: fingerprint(out)}, "rows": table.num_rows,
                                       "files": files, "batches": batches}, indent=1))

    drops = collections.Counter((r["source"], r["drop"]) for r in everything if r["drop"])
    for src in cfg["ingest"]["sources"]:
        mine = [r for r in rows if r["source"] == src["name"]]
        added = sum(1 for r in mine if r["id"] not in held)
        log.info("%s: %s messages (%s new): %s", src["name"], f"{len(mine):,}", f"{added:,}",
                 ", ".join(f"{k} {v:,}" for k, v in sorted(collections.Counter(r["channel"] for r in mine).items())))
        log.info("%s: dropped %s", src["name"],
                 ", ".join(f"{k} {v:,}" for (s, k), v in sorted(drops.items(), key=lambda x: -x[1]) if s == src["name"])
                 or "nothing")
    log.info("%s: %s messages, %s new", out, f"{table.num_rows:,}", f"{table.num_rows - len(stored):,}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--folder", help="folder for messages.parquet and state/ingest.json (default: [ingest] folder)")
    run("ingest", body, p)
