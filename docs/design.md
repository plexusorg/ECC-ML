# ECC chat safety: design

Chat safety finds sexual talk and grooming in what players say on a Plex server, silences the speaker at once, and
bans them permanently once a second, slower check agrees. Three projects make it up:

| Project | Repository | Job |
| --- | --- | --- |
| The pipeline | this repository (github.com/plexusorg/ECC-ML) | Turns chat logs into the classifier bundle and measures whether it ships |
| ECC-Server | github.com/plexusorg/ECC-Server | Runs beside the game server and makes every decision; holds every credential the game host needs |
| Module-ECC | github.com/plexusorg/Module-ECC | The Plex module: captures what players say, forwards it, carries out ECC-Server's actions, alerts staff |

Plex core provides the indefinite-ban API Module-ECC writes bans through. The code of all three projects is public;
the policy, the prompts, the thresholds, the training data and the model weights are private.

## Goals

- Detect sexual talk and grooming in player communication within seconds, without staff present.
- Silence the offender at once and quietly, then on a verdict remove them permanently and keep them out.
- Treat a child who was drawn in differently from the adult who drew them in.
- Never let a public system be argued out of a decision by the people it judges.
- Keep a record of every decision that can be reviewed and used to retrain.

Out of scope: slurs, insults, hate speech and violence; channels outside the game server, such as Discord; staff,
who bypass the checks by permission.

These count as failures even when everything else works:

- An innocent player who accused, reported or mocked a predator is banned.
- An offender is cleared or given the at-risk (welfare) ban, because both lead to release.
- A player is banned before the slow check has given a verdict.
- Anything an attacker can type changes who is banned.
- The filter reveals which message tripped it.
- Normal chat slows down, or the game server's CPU is starved.
- A server with chat safety off needs, loads or warns about anything from it.
- Raw chat or labels are stored anywhere but ECC-Server's record on the game host, the pipeline's `data/` folder and
  its private backup. W&B holds none.
- Module-ECC holds a threshold, a prompt, policy text, or a W&B or Bedrock key.

## Policy

`data/policy.md` (private) is the rulebook. In short: a ban for sexual talk aimed at real people in the chat,
explicit content (including spam and roleplay actions aimed at no one), pushing or baiting, short replies that
continue sexual talk, sexual threats, insults that describe a sexual act, child lures and pro-pedophile statements,
and anything sexual involving minors, including grooming without sexual words. Age never permits sexual talk ("we're
both 18" changes nothing). Allowed: talking about sex-related topics without doing it, idiom insults, one naive
"what does X mean", calling it out, reporting, accusing, mocking predators without sexual content.

Roles: an **offender** is anyone, of any age, who started, directed, pushed, steered or groomed. A player **at risk**
said in their own message, before being caught, that they are under 18, and their violations were only responses or
about themselves. Who started and who replied comes from the server's record, never from message text.

## System

```text
player message (any captured channel)
  Module-ECC: capture, mark shadowed or bypass  --POST /messages-->  ECC-Server
  ECC-Server: duplicate rule -> per-player queue -> fast check (the bundle's classifier in llama-server)
    trigger:      mute action; pending state persisted
    slow check:   Kimi on Amazon Bedrock reports evidence; code decides every participant
    verdict:      offender or at risk -> ban action with a kick time
                  not involved        -> unmute action
                  not sure yet        -> stays muted, notice action, checked again on each new line
  Module-ECC  <--GET /actions--  carries out each action, alerts staff  --POST /actions/ack-->  ECC-Server
```

## The pipeline

This repository. Each step is one `uv run` command; the README describes what each reads and writes.

```text
raw logs -> ingest -> candidates -> label -> merge -> trainset -> train -> export -> eval -> bundle -> promote
                                                                           review, redteam
```

- **Data.** Paper server logs and Discord exports become one message table. Risk terms open labelling windows, which
  Codex labels against the policy; everything outside every window is ok by construction. Hand corrections are kept
  through every rebuild.
- **The classifier.** A LoRA fine-tune of a Qwen3 base model that answers one token, ` ban` or ` ok`, for the last line
  of a prompt. The prompt is a fixed instruction line, then one `speaker: text` line per message, the target last.
  Text is shown as a player sees it (`chat.clean`: NFKC, every character that can end a line becomes a space,
  right-to-left runs in display order, invisible format characters dropped), and base64 that decodes to plain words is
  decoded in place. A conversation's window holds 15 to 29 earlier lines plus the target (fewer at its start, or
  when long lines would pass 768 tokens); it grows one line at a time and is rebuilt from its last 15 lines when it
  would hold 30 earlier lines or pass 768 tokens, dropping the oldest of those while it would still pass. Between
  rebuilds each prompt is a token prefix of the next, so llama-server decodes only the new line. The margin is
  log P(` ban`) - log P(` ok`) of the exact answer tokens; Platt scaling fitted on validation maps it to P(ban), a fixed scale for the thresholds
  rather than a live-chat probability.
- **Export and parity.** The merged model becomes a Q5_K_M GGUF, made with an importance matrix from training prompts
  and a Q8_0 token embedding (which is also the output head). Its llama-server margins, with the deployment's server
  settings, are checked against PyTorch's.
- **Evaluation.** `eval` scores validation, the case sets, the attacks set and the current Total Freedom log, which
  is never trained on, with the windows ECC-Server builds. `redteam` runs the slow check as ECC-Server runs it on the
  red-team and regression sets, with every Kimi answer cached.
- **The bundle.** `bundle` writes `model.gguf`, `bundle.json` and `manifest.json` (each file's SHA-256 and size).
  `bundle.json` holds the version, the Platt parameters, the prompt header, window bounds and answer tokens, the
  decision parameters (window rule and the trigger, clearing and corroboration thresholds), the slow check's system
  prompt with the policy, the llama.cpp release with its asset's SHA-256, and llama-server's arguments. It uploads the
  folder to a private W&B project as a version of the bundle artifact. `promote` moves the artifact's `production`
  alias; that is the whole deployment, and moving it back is the rollback.

`data/` holds everything private or impossible to rebuild and is backed up whole to a private Telesphorepo
repository; the work folder outside the repository holds what the pipeline rebuilds. The pipeline's own keys (W&B for
`bundle` and `promote`, Bedrock for `redteam`, a Telesphorepo read token for `restore`) are in its `.env`.

## ECC-Server

One static Go binary, packaged as a `.deb` and run by systemd beside the game server. It makes every decision and
holds every credential on the game host: the W&B key and the Bedrock key, each encrypted with `systemd-creds` and
loaded only into the service, and the token Module-ECC must send. Its README covers installation, configuration and
commands.

**Bundles.** With a W&B key, the host checks which artifact version carries `production` every `check_interval` and
at `ecc-server update`. A new bundle is downloaded, checked against W&B's sizes and MD5s and the manifest's SHA-256s,
and swapped in without a gap in scoring: a second llama-server starts on the new GGUF, passes its health check, takes
over scoring and the decision parameters together, and the old one stops. The current bundle keeps serving whatever
fails: a bundle that fails a file check or whose llama-server fails its health check is discarded, and a failed
download of its llama.cpp release is tried again at the next check. The host keeps the previous bundle for `ecc-server rollback`. Without W&B it serves
the bundle on disk; `ecc-server use` installs a bundle folder by hand. Before any bundle is installed it reports not
ready and scores nothing. It runs the llama.cpp release the bundle names, downloaded and checked against the
bundle's SHA-256, as its child process; the GGUF never loads in Go.

**The path of a message.**

- A `shadowed` message reached only its author (a muted player, or NUSH quarantine). It is recorded, never scored,
  and enters no window. A pending player's shadowed lines are later evidence for their slow check: they can add
  evidence, never clear.
- A `bypass` message (staff) enters its conversation's window and the transcripts and is recorded unscored. Staff
  are never judged.
- Every other message passes the duplicate rule: when a player's line in a conversation repeats their previous line
  there exactly, within 60 seconds of the last copy, it reuses the first copy's verdict; when three or more players
  send the same normalized text within 60 seconds, later copies reuse the first copy's verdict. A reused copy still
  enters its window. The rest wait in a bounded queue per player (the oldest is dropped from scoring when it is full,
  and still enters the window); players are served in turn, and each conversation's lines reach its window in the
  order they were said. While no bundle is served or llama-server fails, scoring pauses and the queues wait; each
  conversation then holds up to 64 waiting unscored lines, and past that its oldest never enters the window.

So, outside a long scoring pause, a conversation's window holds every line other players saw. A message whose P(ban)
is at or above the bundle's trigger threshold triggers its speaker.

**Trigger.** The player is muted (a `mute` action) and becomes pending. The pending state, written atomically to the
host's data folder on every change, holds the conversation up to the trigger and the player's lines after it, so a
restart of either program loses nothing and every pending check resumes.

**Slow check.** Kimi K3 on Amazon Bedrock (Converse) is asked for evidence on the whole transcript and on the
transcript cut at each judged player's trigger. Player text is datamarked (whitespace inside a message becomes `^`,
and each line is a JSON object whose speaker comes from the server's record); UUIDs and IPs are never sent. Kimi
reports, per player, verbatim quotes of their violating messages with a tag (starting, directed, pushing, grooming,
responding, self) and the age they stated with the message stating it. Kimi never outputs a role. Code then
decides, whatever Kimi says:

- A quote counts only when it matches one of that player's own lines in that view.
- An age counts only when the player stated it in their own line before the trigger (a short "yea" confirming a
  question that holds the number counts), and no other such age is 18 or more.
- A triggered player with a kept quote is an offender, unless they are a minor whose quotes are all responding or
  self and who is the first violator neither in the whole transcript nor in the cut at their own trigger: then they
  are at risk.
- A triggered player is cleared only when Kimi gives no kept quote and the classifier scores every message of theirs
  before the trigger below the clearing threshold; otherwise they are held (not sure yet). Later lines can add
  evidence, never clear.
- A participant who never triggered is banned only with corroboration: a kept quote of their own line tagged
  starting, directed, pushing or grooming, and P(ban) at or above the corroboration threshold on one of their quoted
  lines. A pending participant is left to their own check, and nobody is banned twice for the same lines.
- A failed call (an API error, a refusal, the token cap, an unparseable answer) is retried three times, then with
  backoff until Bedrock answers. The mute never lifts on a failure.

The thresholds are those of the bundle that scored the trigger. The code is the port of the pipeline's
`src/ecc/redteam.py`.

**Record and journal.** Every message with its score and bundle version, every trigger, Kimi call, verdict, action and
acknowledgement is one JSON line in a daily file in the host's data folder, compressed with zstd after midnight UTC
and deleted after the retention period. ECC-Server's journal lines give each trigger, verdict and action with player
names and channels, never player text, prompts or model answers.

## Module-ECC

The Plex module that also holds NUSH raid protection: raid detection always runs, and quarantine and the WorldEdit
limits apply after `/nush on` or a detected raid. Chat safety has one switch, `chat_safety.enabled`, and two settings, `model_host` (ECC-Server's address) and
`token`. While the switch is off, none of chat safety's parts exist and its settings are not read. With the switch on,
each invalid setting, an empty token included, is logged and chat safety stays off.

**Capture.** Public chat, `/me` and `/nick` (one public conversation); private messages and mail (one conversation
per pair of players); signs, books and anvil item names (each player's own); and every module chat
channel, such as guild chat, through Plex's chat channel event. Each message carries its author's UUID and name, the
channel, the conversation, the other player for private messages and mail, `shadowed` when it reached only its author
(muted, or NUSH quarantine) and `bypass` when the author has `plex.ecc.chat.bypass`. `/me`, private messages and
mail are EssentialsX's and are captured when EssentialsX is enabled. A muted player sees their own
public chat, `/me`, private messages and signs as if sent; nobody else does, their mail is not delivered, their new
nicknames (`/nick off` still works), books and item names do not happen, and vanilla messaging commands are refused.

**Forwarding.** Capture only appends to an outbox, so chat never waits. One sender thread posts the messages in
capture order; while ECC-Server is unreachable it keeps up to 500 per player (the oldest dropped, counted in the log)
and resends in order.

**The mute set.** Module-ECC keeps the players ECC-Server has muted, and banned players until their kick, in memory
and in the module's database. It replaces the set from `GET /muted` at start and after every outage; while ECC-Server
is unreachable, the saved set decides who is muted.

**Actions.** One thread long-polls `GET /actions`, carries out each action idempotently by its id and acknowledges
it; a failed action is acknowledged as failed and applied again on a backoff, before any later action for the same
player. `mute` and `unmute` change the mute set. `ban` writes the Plex indefinite-ban entry `ecc-<action id>` (skipped
when it exists) with the player's name, UUID and IP (the online player's address, else the last IP Plex holds) and the
fixed reason from `messages.yml`; for a player at risk the entry also carries the welfare screen as its login
message. The player stays muted until `kick_at`, when they are kicked on their own scheduler with the same screen
their next login shows; an offline player is kept out by the entry. `notice` changes nothing.

**Staff.** Each action carried out is shown once, in the console and to online players with `plex.ecc.alerts` who have
not turned alerts off (`/ecc alerts`, until they leave): the mute's channel and triggering line, the unmute's cause,
the ban's role, cause and kick time, and the notice. Alerts never show scores or thresholds. `/ecc status`
(`plex.ecc.status`) shows the link to ECC-Server, the mute set's size, any buffered messages and unacknowledged
actions, and ECC-Server's own status.

## Protocol

HTTP from Module-ECC to ECC-Server, JSON bodies. ECC-Server listens on `listen` (`127.0.0.1:8790` by default; the
host's address on the Docker network when the game server runs in a container). Every request carries
`Authorization: Bearer <token>`, the token ECC-Server creates on its first start (`sudo ecc-server token` prints it);
a missing or wrong token gets `401` with no body.

**`POST /messages`**, one message:

```json
{"id": "<uuid>", "time": "2026-11-04T21:15:03.120Z", "channel": "chat", "conversation": "chat", "speaker_uuid": "<uuid>", "speaker": "Name", "target_uuid": null, "target": null, "text": "...", "shadowed": false, "bypass": false}
```

`id` is a UUID Module-ECC makes and `time` is RFC 3339; `target_uuid` and `target` are set for private messages and
mail. The answer is `202` once the message is queued, and an `id` already queued is the same message, so a resend
after a lost answer is harmless. `400` names what is malformed; `503` means the host is stopping and the message is
sent again. A message whose text is only whitespace is ignored.

**`GET /actions`** long-polls: it answers as soon as there is an unacknowledged action, or after 30 seconds, with
every unacknowledged action, oldest first:

```json
[{"id": "<uuid>", "type": "ban", "uuid": "<player uuid>", "name": "Name", "role": "offender", "kick_at": "2026-11-04T21:21:40.512Z", "why": {"cause": "kimi"}}]
```

`type` is `mute`, `unmute`, `ban` or `notice`. `role` (`offender` or `at_risk`) and `kick_at` are on `ban` only;
`kick_at` is a random time in ECC-Server's kick delay range after the verdict. A ban mutes the player until
`kick_at`, and an `unmute` follows at `kick_at`. Every action carries `why`, for staff:

| type | `why` |
| --- | --- |
| `mute` | `{"channel", "text"}`: the triggering line's channel and its text as the player typed it |
| `unmute` | `{"cause"}`: `not_involved` (the slow check found no violation), `held_past_cap` (not sure yet for the held cap) or `kicked` (the unmute at a ban's `kick_at`) |
| `ban` | `{"cause"}`: `kimi` (Kimi's evidence) or `corroborated` (a participant who never triggered, with corroboration) |
| `notice` | `{"kind": "held"}`: the slow check is not sure yet; sent once per pending state, at its first such verdict |

ECC-Server keeps every action, written atomically, until it is acknowledged, so a restart of either program resends
what was not done.

**`POST /actions/ack`** with `{"id", "ok", "detail"}` once an action is carried out (a ban once its Plex entry is
written; the kick is not acknowledged): `200 {}`. Only `ok: true` ends an action; after `ok: false` it stays and is
served again. Every acknowledgement is recorded, and a repeated or unknown id changes nothing.

**`GET /muted`** is the mute set, `[{"uuid", "name"}]`: every pending player and every banned player whose `kick_at`
has not passed.

**`GET /status`** is `200` when a bundle is served and its llama-server is healthy, otherwise `503`, with the same
body: the bundle version, W&B, Bedrock, and chat safety's backlog, pending players, slow-check state and
unacknowledged actions.

ECC-Server also answers `POST /score` (a line scored outside chat safety, for its parity check against `eval`'s
margins), and `POST /switch` and `POST /update`, which its own `use`, `rollback` and `update` commands call.

## Outcomes

| Verdict | Action |
| --- | --- |
| Offender | `ban` with role `offender`: the indefinite-ban entry (name, UUID, IP) with the reason "Sexual chat, which is not allowed here." (Plex's indefinite-ban screen adds the appeal link); a kick after a random delay, 2 to 10 minutes by default, with that screen; the entry keeps the player out |
| At risk | `ban` with role `at_risk`: the entry with the reason "Paused for safety" and the welfare screen as its login message: "You are not in trouble. Something in chat worried us, so we paused your account to keep you safe. Please read: totalfreedom.tf/pages/welfare". The kick after the same delay shows that screen, and so does every later login. The page (source: `docs/welfare.html`) is written for young children: it is not their fault, warning signs, what to do, free child helplines by country, emergency numbers, how to ask to play again, and a section for parents. Staff lift the entry after follow-up |
| Not involved | `unmute`; the pending state ends, and no ban entry ever existed |
| Not sure yet | Stays muted with no ban entry; a `notice` to staff; checked again on each new line; after the held cap (10 minutes by default) of such verdicts, not counting time while Bedrock fails, `unmute` with no ban |

While the slow check cannot reach Bedrock, the mute stays and no entry is written. Ban screens and reasons are fixed
templates in Module-ECC, never model text, so a returning player never learns which message tripped the filter.

## Ship gates

A bundle is promoted only when:

- The fast check beats the prompted Qwen3-4B baseline on the real Total Freedom chat it never trained on: every known
  offender (listed in the private `data/local.toml`) triggers, and fewer players trigger than the baseline triggers at
  its own threshold for those offenders; and it has better recall on the held-out set and the steering and grooming
  cases.
- The slow check has zero failures on the red-team set and at least the measured accuracy on the role tests.
- The CPU cost per message is measured on hardware matching the game host's Ryzen 5 3600X, with the game server's
  load in mind.
