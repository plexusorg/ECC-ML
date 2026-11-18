"""Meaning normalization for model inputs. Source records, repeat keys and judge text keep their original form."""
import base64
import re
import unicodedata

RULE = "v3_meaning_v1"
UNICODE_VERSION = "15.1.0"
LINE_BREAKS = re.compile(r"\r\n|[\n\v\f\r\x1c\x1d\x1e\x85\u2028\u2029]")
RTL_RUN = re.compile(r"[\u202b\u202e\u2067]([^\u202c\u2069]*)[\u202c\u2069]?")
BASE64 = re.compile(r"[A-Za-z0-9+/]+={0,2}")
BASE64_MIN = 8

if unicodedata.unidata_version != UNICODE_VERSION:
    raise RuntimeError(f"model text requires Unicode {UNICODE_VERSION}; this Python uses {unicodedata.unidata_version}")


def clean(text):
    """Use v3's display order: NFKC, line breaks to spaces, reversed bidi runs, then remove format characters."""
    text = LINE_BREAKS.sub(" ", unicodedata.normalize("NFKC", text))
    text = RTL_RUN.sub(lambda match: match.group(1)[::-1], text)
    return "".join(char for char in text if unicodedata.category(char) != "Cf")


def decode_base64(text):
    """Use v3's exact ASCII-word criterion. Other base64-looking runs remain unchanged."""
    def decoded(match):
        token = match.group(0)
        body = token.rstrip("=")
        if len(body) < BASE64_MIN or len(body) % 4 == 1 or (body != token and len(token) % 4):
            return token
        raw = base64.b64decode(body + "=" * (-len(body) % 4))
        if 0x20 not in raw or any(byte < 0x20 or byte > 0x7E for byte in raw):
            return token
        if sum(0x41 <= byte <= 0x5A or 0x61 <= byte <= 0x7A or byte == 0x20 for byte in raw) * 10 < len(raw) * 7:
            return token
        return raw.decode("ascii")
    return BASE64.sub(decoded, text)


def normalize(text):
    return decode_base64(clean(text))
