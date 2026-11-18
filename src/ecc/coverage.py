"""Require complete current targets before scoring. Earlier context can still be cut to the model budget."""
import re

import numpy as np

from . import inputs as I

RULE = "complete_target_v1"


def source_shortened(raw_text, channel):
    """Whether text preparation would discard visible characters or part of a URL."""
    text = I.CODES.sub("", I.SECTION.sub("", raw_text)).replace("\r", " ").replace("\n", " / ").strip()
    text = re.sub(r"\s+", " ", text)
    if len(text) > I.limit(channel):
        return True
    return any(len(url.group(0)) > I.URL_OVER for url in I.URL.finditer(text))


def route(raw_text, channel, prepared_text, rendered_target, sh_tokenizer, c_tokenizer, c_tokens):
    """Return reject, c or cascade. Use the same rendered target as the C input builder, including its role header.

    A rejected target must not enter context, judgment, or the release queue. The caller can ask for shorter text.
    Route c bypasses the short encoder so a clipped short-encoder input cannot clear the full target.
    """
    if source_shortened(raw_text, channel):
        return "reject"
    c = c_tokenizer.encode(rendered_target, add_special_tokens=False).ids
    if len(c) + 3 > c_tokens:
        return "reject"
    sh = sh_tokenizer.encode(I.line_string(channel, prepared_text), add_special_tokens=False).ids
    return "c" if len(sh) + 2 > I.SH_LINE_TOKENS else "cascade"


def _contexts(df, c_tokenizer, c_tokens, sh_tokenizer=None):
    """Build causal context with the same rejection rule as serving, before remembering each line.

    Text must already be prepared and canonicalized. raw_text must retain the original source. Context-only and
    trusted records retain their existing exemption. The returned routes cover all rows, including context rows.
    """
    if "raw_text" not in df:
        raise ValueError("coverage needs the original raw_text, not an already shortened copy")
    who = I.keys(df)
    partner = df.partner.fillna("").str.lower().to_numpy(object)
    channel = df.channel.to_numpy()
    text = df.text.fillna("").to_numpy(object)
    raw = df.raw_text.fillna(df.text).to_numpy(object)
    exempt = ~df.judged.to_numpy(bool) | (channel == "discord")
    if "trusted" in df:
        exempt |= df.trusted.fillna(False).to_numpy(bool)
    routes = np.full(len(df), "context", object)
    c_lengths, sh_lengths = {}, {}

    def keep(i, previous):
        if exempt[i]:
            return True
        if source_shortened(raw[i], channel[i]):
            routes[i] = "reject"
            return False
        names = {who[i]: "t"}

        def name(key):
            if key not in names:
                names[key] = f"s{len(names)}"
            return names[key]

        for j in previous:
            name(who[j])
            if channel[j] == I.PRIVATE:
                name(partner[j])
        tag = f"[msg to {name(partner[i])}] " if channel[i] == I.PRIVATE else \
            ("" if channel[i] == "chat" else f"[{channel[i]}] ")
        target = f"{tag}t: {text[i]}"
        n = c_lengths.get(target)
        if n is None:
            n = len(c_tokenizer.encode(target, add_special_tokens=False).ids)
            c_lengths[target] = n
        if n + 3 > c_tokens:
            routes[i] = "reject"
            return False
        if sh_tokenizer is None:
            routes[i] = "c"
            return True
        line = I.line_string(channel[i], text[i])
        n = sh_lengths.get(line)
        if n is None:
            n = len(sh_tokenizer.encode(line, add_special_tokens=False).ids)
            sh_lengths[line] = n
        routes[i] = "c" if n + 2 > I.SH_LINE_TOKENS else "cascade"
        return True

    return I.contexts(df, accept=keep), routes


def contexts(df, c_tokenizer, sh_tokenizer, c_tokens):
    """Offline training routes, including the original SH diagnostics."""
    return _contexts(df, c_tokenizer, c_tokens, sh_tokenizer)


def c_contexts(df, c_tokenizer, c_tokens):
    """C-only serving routes. Check the complete target before adding it to any later context."""
    return _contexts(df, c_tokenizer, c_tokens)
