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


def contexts(df, c_tokenizer, c_tokens):
    """Causal context with the serving rejection rule, checked before each line is remembered: (contexts, routes).

    Text must already be prepared and canonicalized; raw_text must retain the original source. Context-only and
    trusted lines are exempt. Each route is `context` (exempt), `reject` (preparation would shorten the source, or
    the rendered target with its role header does not fit in c_tokens) or `c` (scored). A rejected target never
    enters a later line's context. The routes cover all rows, including context rows.
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
    lengths = {}

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
        n = lengths.get(target)
        if n is None:
            n = len(c_tokenizer.encode(target, add_special_tokens=False).ids)
            lengths[target] = n
        if n + 3 > c_tokens:
            routes[i] = "reject"
            return False
        routes[i] = "c"
        return True

    return I.contexts(df, accept=keep), routes
