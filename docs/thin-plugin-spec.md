# Thin plugin: ECC-Server decides, Module-ECC captures and acts (design spec)

Working document. When it lands, fold it into `docs/spec.md` sections 2 and 3 and delete it.

## Why

Module-ECC (the open-source Minecraft module) today holds the duplicate rule, the per-player queue, the trigger, the
pending state, the transcripts, the whole slow check (Kimi, quote validation, roles, decisions) and the daily record.
That puts decision logic and private policy text in a public plugin, splits one decision across two programs, and
means every rule change ships as a plugin release. ECC-Server, already the owner of the private bundle and the W&B and
Bedrock keys, becomes the owner of every decision. The plugin forwards what players say and carries out what the
server tells it.

## Split

| Module-ECC (Java, in game) | ECC-Server (Go, beside it) |
| --- | --- |
| Capture every channel it captures today, with true author, channel, conversation and target | Flood and duplicate rule, per-player queues, scoring, the trigger |
| Show a muted player's message to them alone (today's shadow display) | Pending state, transcripts, the slow check (Kimi through Bedrock, quote validation, ages, roles, `decide`), the held cap and re-checks |
| Mark a message `bypass` (staff permission) or `shadowed` (NUSH quarantine or muted) | The daily record (every message, score and decision) |
| Keep the current mute set, persisted, so a server outage never unmutes anyone | Which players are muted, banned or released, and when a banned player is kicked |
| Carry out actions: mute, unmute, ban (Plex indefinite-ban entry with name, UUID and IP, and the fixed ban screen), kick at the given time | All thresholds, prompts and keys |
| `/ecc status` shows the server's status | `/status` |

NUSH (raid detection, quarantine, WorldEdit limits) stays in the plugin unchanged.

Staff (`bypass`) and quarantined lines now reach the server, recorded and unscored, so conversation windows hold
every line players saw. That closes the old gap where the host's windows lacked staff chat.

## Protocol (loopback HTTP, plugin to server)

- `POST /messages` with one message: `{"id", "time", "channel", "conversation", "speaker_uuid", "speaker",
  "target_uuid", "target", "text", "shadowed", "bypass"}`. `id` is a UUID the plugin makes; `time` is RFC 3339 with
  milliseconds; `target_uuid` and `target` are set for private messages and mail, else null. The server answers 202
  once it has queued the message, and treats an `id` it has already queued as the same message (a resend after a lost
  answer is harmless). The plugin sends from one sender thread of its own, never a game thread, in capture order;
  while the server is unreachable it keeps a bounded per-player buffer (oldest dropped, count logged) and resends in
  order. Chat is never blocked or delayed.
- `GET /actions` long-polls (up to 30 s) and returns every unacknowledged action, oldest first, as soon as there is
  one: `{"id", "type": "mute" | "unmute" | "ban", "uuid", "name", "role", "kick_at"}`. `role` (`offender` or
  `at_risk`) and `kick_at` are on `ban` only. `kick_at` is an absolute time the server picked, a random 2 to 10
  minutes after the verdict; the plugin kicks the player then if they are online (the ban entry already keeps an
  offline player out). The server keeps every action until it is acknowledged, so a restart of either program resends
  what was not done.
- The plugin applies actions idempotently, by `id`: mute and unmute set the mute set; a ban writes the Plex
  indefinite-ban entry labelled `ecc-<id>` (skipped when that label exists) with name, UUID and the IP (the online
  player's address, else the last IP Plex holds), the fixed reason, and for `at_risk` the welfare screen as its
  message, then schedules the kick for `kick_at` on the player's entity scheduler. A player is muted from the ban until
  the kick, whether or not they were muted before (a participant banned by corroboration).
- `POST /actions/ack` with `{"id", "ok", "detail"}` once an action is carried out (for a ban: once the Plex entry is
  written; the kick is not acknowledged). A failed action is retried by the plugin with backoff; the server records
  every acknowledgement in the record.
- `GET /muted` returns the current mute set (`[{"uuid", "name"}]`), including banned players until their kick. The
  plugin replaces its persisted copy with it at start and whenever it reconnects after an outage. While the server is
  unreachable, the persisted copy decides who is muted.
- `GET /status`: model bundle, W&B, Bedrock, message backlog, pending players, unacknowledged actions.

A muted player's messages are still sent with `shadowed: true`: the server keeps them as later evidence for that
player's slow check (they can add evidence, never clear) and records them, and does not score them.

## Ownership of state

- ECC-Server persists pending state, unacknowledged actions and the action sequence in its state directory, written
  atomically, so a restart of either program loses nothing.
- The record moves to ECC-Server's state directory as daily `YYYY-MM-DD.jsonl.zst` files with the same retention
  rule; the pipeline ingests them as before.
- The plugin persists only the mute set (its existing module storage).

## Removed from the plugin

`Scorer`, `Pending`, `SlowCheck`, `Evidence`, `Bedrock`, `RecordLog`, `DuplicateKey`, `Decision`, `Verdict`, the model
host scoring client, and their config keys (`queue_size`, `record_retention_days`, `kick_delay_minutes`,
`held_cap_minutes`). Config keeps the chat safety switch, `server` and `model_host`. The pending table migration is
replaced by a mute-set table.

## Built in ECC-Server

The Java logic above, ported to Go: `Evidence` (itself a port of `src/ecc/redteam.py`) checked by replaying
`redteam.py`'s cached Kimi answers with zero mismatches, as the Java port was; the trigger, duplicate rule, queues,
pending state, held cap, kick timing and record as the Java code does them today, with the defects the last review
found fixed. Thresholds still come from the bundle; the kick delay range and held cap become ECC-Server config.

## Verification

Builds, the replay parity check, one independent review per project, and an end-to-end run on the game server by the
owner: a test account triggers, is muted at once on every channel, Kimi's verdict arrives, the ban entry and kick
happen, and an innocent trigger is released.
