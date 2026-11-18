"""The chat-log parsing `dossier` reads the raw chat with: Paper log lines, the commands that carry a message, the
player tracking that resolves shown and typed names to accounts (Players), and the FLAGGED embeds of Creayun's Discord
exports."""
import collections
import json
import re
import unicodedata

from .inputs import CODES, SECTION

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


def strip_colour(text):
    """Text as the module receives it: Minecraft components carry colour as style, never as § codes."""
    return SECTION.sub("", text)


# ---------------------------------------------------------------- account names

LOOK = str.maketrans("ᴀʙᴄᴅᴇꜰɢʜɪᴊᴋʟᴍɴᴏᴘǫʀꜱᴛᴜᴠᴡʏᴢаеорсухкмтвніјѕԁɡαορινтκε",
                     "abcdefghijklmnopqrstuvwyzaeopcyxkmtbhijsdgaopivtke")
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
        self.nicks = {}       # lower account -> its folded nickname, from /nick
        self.shown = {}       # lower account -> folded name it last spoke under
        self.together = {}    # folded display -> lower accounts online at every line it spoke
        self.partner = {}     # lower account -> account /r answers
        self.sudo = {}        # lower account -> (issuer, second, kind)
        self.speakers = collections.Counter()   # how chat lines' display names resolved
        self.targets = collections.Counter()    # how typed player names resolved

    def seen(self, account):
        self.online[account.lower()] = account
        self.known.setdefault(account.lower(), account)

    def nick(self, account, folded):
        """Records `account`'s /nick (folded; "" for off)."""
        self.nicks.pop(account.lower(), None)
        if folded:
            self.nicks[account.lower()] = folded

    def match(self, display):
        """(account, how) of a chat line's display name by what it shows, or None: an online account's name shown as
        is (EssentialsX marks a nickname with ~), "name"; the one online player whose /nick is the shown name,
        "nickname"; an online account's name anywhere in the display, [tags] included, "name"; the one online name the
        shown name matches after folding lookalikes, "lookalike". Changes nothing."""
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
        return None

    def speaker(self, display):
        """(account, how) of a chat line's display name: tags, prefixes, a nickname or lookalike letters. In order:
        what the name shows (match); the one player online at every line this display spoke. Only an online player
        can speak, so an offline account is never the answer. An unresolved display keeps its shown name."""
        hit = self.match(display)
        if hit:
            return hit
        plain = plain_name(display)
        tokens = [t for t in plain.split() if re.search(r"\w", t)] or display.split()
        last = tokens[-1] if tokens else display.strip()
        name = EDGE.sub("", last)
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
LEAVE = re.compile(r"^(\S+) (?:lost connection: |left the game$|left$)")
BOOT = "Starting minecraft server"


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
