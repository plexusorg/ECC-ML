# Module-ECC: chat safety for Plex servers (design spec)

Working document. Delete it when the work lands; move what stays true into each project's docs.

## Why

On 2026-11-02 an adult player openly admitted to being a pedophile, solicited girls, and drew
players, one of whom said he was 13, into sexual talk for twenty minutes on the Total Freedom server before a staff
member acted. The ban he got was a 24-hour `/gtfo`, and he logged back in. Real chat logs show the same thing
happened before without any action (solicitation, explicit image spam, sexual harassment of a self-declared
15-year-old). This must not happen again.

## Goals

- Detect sexual talk and grooming in player communication within seconds, without staff present.
- Silence the offender at once and quietly, then on a verdict remove them permanently and stop them coming back.
- Treat a child who was drawn in differently from the adult who drew them in.
- Never let a public, open-source system be argued out of a decision by the people it judges.
- Keep a record of every decision that can be reviewed and used to retrain.

## Non-goals

- Slurs, insults, hate speech, violence. Out of scope; placement in Plex or elsewhere is decided later.
- Moderating Discord or any channel outside the game server.
- Staff alerts. The outcome is a permanent ban or no ban.
- Moderating staff. Staff bypass the filter by permission.

## Anti-goals (failure even if it "works")

- An innocent player who accused, reported or mocked a predator gets banned.
- An offender gets cleared or gets the at-risk (welfare) ban, because both lead to release.
- A player is banned or refused at login before the slow check has given a verdict.
- Anything an attacker can type changes who is banned.
- The filter reveals which message tripped it.
- Normal chat slows down or the game server's CPU is starved.
- A server with chat safety off needs, loads or is warned about anything from chat safety.
- Raw chat or labels are stored anywhere but the game server's record, the pipeline machine's `data/` directory and the
  owner's private Telesphorepo data repository. W&B holds none.
- The Minecraft plugin has a W&B dependency, key or call, or the plugin's config or source holds a threshold.
- A label, correction or train/val assignment ends up on the wrong message after a re-ingest.

## Policy

`data/policy.md` (private) is the rulebook. In short: a ban for sexual talk aimed at real people in the chat,
explicit content (including spam and roleplay actions aimed at no one), pushing or baiting, short replies that
continue sexual talk, sexual threats, insults that describe a sexual act, child lures and pro-pedophile statements,
and anything sexual involving minors, including grooming without sexual words. Age never permits sexual talk ("we're
both 18" changes nothing). Allowed: talking about sex-related topics without doing it, idiom insults, one naive
"what does X mean", calling it out, reporting, accusing, mocking predators without sexual content.

Roles: **offender** is anyone, of any age, who started, directed, pushed, steered or groomed. **At risk** is a player
who said in their own message, before being caught, that they are under 18, and whose violations were only
responses or about themselves. Who started and who replied comes from the server's record, not from message text.

## System

```text
player message (any channel)
  -> ECC capture -> normalize -> flood/duplicate rule -> per-player queue
  -> fast check (model host -> local llama-server, fine-tuned Qwen3 classifier) -> per-player window rule
  -> trigger: shadow mute + persisted pending state (no ban entry)
  -> slow check (Kimi K3 on Amazon Bedrock, verified by code)
  -> offender | at risk: write the permanent ban entry (UUID, name, IP) and reason, kick after a random delay
     not involved: lift shadow mute
     not sure yet: stay shadow-muted, re-check on new messages, no ban after the cap
```

### 1. Plex core: indefinite-ban API

Plex gains a public API to create, read, update and delete entries in `indefbans.yml` (label, reason, users, UUIDs,
IPs). A write updates the file and the in-memory indefinite-ban list in one step, so the ban is enforced at the next
login without `/plex reload`. Login enforcement is unchanged (`BanListener` checks UUID, IP and username). The API
validates input once at the boundary (label and message rules in `Config.validateIndefiniteBans`). Module-ECC
writes an entry only when the slow check returns a verdict that bans; it never writes one at the trigger.

An entry may also carry an optional `message` (MiniMessage), set and read through the same API. When it is set, the
login screen shows it in place of the `messages.yml` indefinite-ban template; without it the template is used as
before. The screen belongs to the entry, so nothing else records which player sees which screen, and deleting the
entry removes both. (Plex 2.0.1.)

### 2. Module-ECC (Java): NUSH plus chat safety in one module

Module-NUSH becomes Module-ECC (Extra Chat Controls): one module, two features. NUSH stays the name and branding of
the raid feature (raid detection, quarantine, WorldEdit restrictions; `/nush`, `plex.nush.*` and its config are
unchanged). The second feature is chat safety (its name is still to be chosen; "chat safety" below). The module's own
name, main class and build coordinates change to ECC without compatibility shims (unreleased Plex 2.0). Chat safety
shares the module because NUSH already owns the state it needs: shadow
muting (quarantine echoes a message only to its sender and cancels it, which also keeps it off DiscordSRV), raid
state and session trust.

**Modes.** NUSH is always on; it is core to the module and has no enable switch, so an existing NUSH config keeps
working unchanged. Chat safety has one switch in the module's config, off unless the config says otherwise.

- **Off.** Module-ECC never contacts the model host, and none of chat safety's parts exist at runtime: no chat
  listeners, no tasks, no queue, no checks, no JSONL record, no data folder for it, and none of its settings are read
  or validated. A server with chat safety off needs no model host, no llama-server, no Bedrock credentials, no `.env`
  keys for chat safety and no XenForo settings, and nothing logs a warning about their absence.
- **On, at startup.** Chat safety validates its own settings: model host endpoint, Bedrock region, model id and
  credentials, and the XenForo URL, key and node. If any is missing or invalid, the log names each one and chat safety
  stays off. Otherwise chat safety runs, and the model host being down is handled by the one rule under Fast check
  below, at startup or at any later time. NUSH keeps running either way.
- Chat safety uses NUSH's shadow muting, raid state and session trust.
- NUSH's names, `/nush` command, `plex.nush.*` permissions and config keys do not change.

ECC runs on the main server only; players chat nowhere else (proxy and auth are login steps), so the local
`indefbans.yml` on main is the enforcement point.

**Capture.** Public chat (`AsyncChatEvent`), private messages (EssentialsX `/msg`, `/tell`, `/w`, `/r`, `/mail`),
guild chat, `/me`, `/nick`, signs, books, anvil item names. Each message is recorded with its true author, time,
channel and, for private messages, the other player; a guild's chat is one conversation. Guild chat is sent two
ways: `/guild chat <message>`, and plain chat while a member's guild chat toggle is on, which Module-Guilds cancels
and redirects. Both go through `Guilds.sendChat`, so Module-Guilds fires a guild chat event there (the sending
player, the guild's id and the message), and ECC listens to it. Watching the command alone would miss the toggle.
Bypass: `plex.ecc.chat.bypass` (staff); bypassing players are logged, never checked.

**Normalize.** Lowercase, strip color codes, map lookalike characters (fullwidth, Cyrillic, leetspeak), collapse
repeated characters, drop random-looking tokens (hex, digit runs, mixed gibberish). Used for the duplicate rule.

**Flood and duplicates.** When three or more players send the same normalized text within 60 seconds, the first copy
is checked with its context and later copies reuse its verdict (measured: 86 skips in 80 days of normal chat, 2,885
during raids). A player's exact repeats within 60 seconds reuse that player's last verdict. There is no general
text-to-verdict cache, because context changes meaning.

**Queue.** One bounded queue per player, served fairly, so one flooding player cannot delay others; the oldest
messages drop when a player's queue is full. A raid does not pause checking: quarantined players' chat is shadowed and
recorded unscored, and every other player is checked as usual.

**Fast check.** A fine-tuned Qwen3 single-token classifier (`ban`/`ok`) run by the model host (section 3) on the game
host (CPU). Module-ECC talks only to the model host, over local HTTP; it never calls llama-server, and the plugin has
no W&B dependency, key or call. Input: the conversation's window, which is the fixed instruction header followed by
one `speaker: text\n` line per message. Text is shown as a player sees it: NFKC, so styled and fullwidth letters
are plain; every line-ending character a player can type (including U+2028, U+2029 and NEL) becomes a space, so
a message can never start a line of its own; right-to-left override runs are put in display order; and invisible
format characters are dropped (`src/ecc/chat.py` clean). The target is
the window's last line and is scored at its end. A window holds 15 to 29 earlier lines plus the target: it grows until
it would hold 30 earlier lines and is then rebuilt from its last 15, or earlier when it would pass 768 tokens. Each
message appends to its conversation's window, so the host processes only that line's tokens. Score: the host returns
P(ban): log P(" ban") - log P(" ok") of the exact answer tokens, mapped by Platt scaling fitted on validation. That
calibration follows validation's class balance (about 5% ban), not live chat's, so P(ban) is a fixed scale for the
thresholds rather than a live-chat probability; the thresholds come from evaluation. A per-player window rule turns scores into a trigger (rule and threshold set from evaluation;
current measurement favours a single message above a high threshold). The window rule, the trigger threshold, the clearing threshold and the
corroboration threshold are the decision parameters of the deployed bundle. Module-ECC reads them from the model host
and keeps them in memory only; they appear in no plugin file. Every score carries the bundle version it came from,
and Module-ECC reads the decision parameters again when that version changes, so a score and the thresholds applied
to it always belong to the same bundle. The weights, thresholds and training data are private.

Whenever the model host is unreachable or not ready, at startup or later, scoring pauses and Module-ECC keeps
retrying in the background. Chat is never blocked, capture and the record continue, and scoring resumes on its own
when the host answers:
- **Queue:** unscored messages wait in the per-player bounded queue and are scored, in order, when the host returns, so a trigger fires late rather than never. A player's oldest unscored messages are the first dropped when the queue is full, and the record notes how many were dropped.
- **Warnings:** the console warns once when scoring stops and once when it resumes.
- **Status:** `/ecc status` shows the host state and the backlog.

**Trigger.** The player is shadow-muted on every captured channel at once (they see their own messages; nobody else
does). Module-ECC records the player as pending in its own persisted state, which holds the shadow mute and the data
the slow check needs. The state survives disconnect, rejoin and server restart: a pending player who rejoins is
shadow-muted again, and the slow check of every pending player resumes after a restart.
The pending state, including the shadow mute, is stored in the database and applied from there. It does not depend
on the model host. No ban entry exists, so the player can log in and play; no one hears them. Nothing visible happens, and other players' chat does not change.

**Slow check.** Called on demand for each triggered player; retried with backoff until Bedrock answers; no local
fallback. The player stays shadow-muted, with no ban entry, while it waits.

- Kimi K3 (`us.moonshotai.kimi-k3`) through the AWS Java SDK (Bedrock Converse), credentials from the module's
  `.env`.
- Input: the conversation transcript with player text datamarked (spotlighting), names aliased, IPs and UUIDs never
  sent.
- Output (strict JSON schema): per judged player, verbatim quotes of that player's own violating messages, a tag per
  quote (starting, directed, pushing, grooming, responding, self), and the age the player stated with the message
  that states it. Kimi never outputs a role.

Code then guarantees, independent of anything Kimi says:

- Authorship comes from the server record. A quote counts only if it matches one of that player's own messages.
- Roles are derived in code from validated quotes, tags and message order (first directed violation, steering).
  An age counts only if the player stated it before the trigger (a short "yea" confirming a question with the number
  counts).
- Temporal cutoff: clearing a player uses only messages up to their trigger. Later messages can add evidence, never
  clear.
- Two-model clearing: a shadow mute is lifted only if Kimi returns no validated violation and the fast-check
  classifier scores every pre-trigger message below its clearing threshold.
- Participants who did not trigger: Kimi judges every participant in the transcript. A non-triggered player is banned
  only with corroboration: a validated quote of their own message tagged starting, directed, pushing or grooming,
  and the classifier scoring at least one of their quoted messages above a lower corroboration threshold. This
  catches a groomer who avoided sexual words while the child who replied tripped the filter. A participant banned
  this way gets the same verdict-time entry and random-delay kick; they were never shadow-muted.
- Failure (API error, refusal, token cap, unparseable output) means retry; the shadow mute never lifts on failure.
- Ban screens come from fixed templates, never model text. The `reason` in `indefbans.yml`, which Plex shows on the
  login screen, is the template text only, so a returning player never learns which message tripped the filter.

**Evidence for staff.** For each final verdict, ECC posts to a staff-only node on the XenForo forum through the
XenForo REST API (key in the module's `.env`): player, role, the verified quotes, and the transcript excerpt. Posts are
made from code with fixed formatting; quotes are the players' own validated messages. A failed post is retried and
never affects the ban.

Measured on 75 red-team conversations (direct and hidden injections, fake system and staff lines, framing,
self-reclassification, obfuscation, role reversal, steering, rule-aware evasion, adult claims) in two independent
runs: 0 offenders cleared, 0 offenders turned at-risk, 0 innocents framed, 0 benign controls harmed, with
datamarking plus the code guarantees. CoVe-style verification and an injection-attempt veto changed no outcome and
are not used.

**Outcomes.**

| Verdict | Action |
| --- | --- |
| Offender | Write the indefinite-ban entry (UUID, name, current IP) with the offender reason ("Sexual chat, which is not allowed here."; Plex's indefinite-ban screen adds the appeal link) through the Plex API, then kick after a random 2-10 minute delay (at once if the player is offline) with the offender screen; the entry blocks future logins |
| At risk | Write the indefinite-ban entry through the Plex API with the reason "Paused for safety" and the welfare screen as its `message`, then kick after the random delay (at once if offline) with the same screen. Welfare screen: "You are not in trouble. Something in chat worried us, so we paused your account to keep you safe. Please read: totalfreedom.tf/pages/welfare". Every later login shows that screen, never the indefinite-ban template. The page (source: `docs/welfare.html`) is written for young children: it is not their fault, warning signs, what to do, free child helplines by country, emergency numbers, how to ask to play again (register on the forum and start a thread), and a section for parents. Staff lift the entry after follow-up in that thread |
| Not involved | Lift the shadow mute silently and end the pending state; no ban entry ever existed |
| Not sure yet | Stay shadow-muted with no ban entry; re-run on each new message; after a 10-minute cap of real verdicts (outage time not counted), end the pending state and lift the shadow mute with no ban |

During an outage of the slow check the shadow mute stays and no entry is written.

**Record.** Every captured message, its scores, bundle (model and prompt) version, and every decision with its evidence, is
appended as one JSON line to the current day's file in the module's data folder; shadowed messages carry a
`shadowed` flag. At midnight the day's file is closed and compressed with zstd (`YYYY-MM-DD.jsonl.zst`); files older
than the configured retention are deleted. Estimated volume at Total Freedom's ~2,000 messages a day: under 1 MB a
day uncompressed, roughly 40 MB a year compressed. The module writes line-oriented files because appends are cheap,
a crash loses at most the last line, and they need no heavy Java dependency; the Python pipeline converts them to
Parquet when it ingests them.

**Configuration (private, per deployment).** Chat safety's switch; NUSH has none, and its keys are unchanged. Chat
safety's settings are read only when its switch is on: model host endpoint, Bedrock region and
model id, XenForo URL and node, kick delay range, caps. The model, window bounds, window rule, thresholds and Platt
parameters are not plugin settings; they come from the model host. Chat safety's secrets (Bedrock credentials, XenForo
key) live in the module's `.env`, which is read only when chat safety is on. Nothing in the module's config, source or
`.env` is a W&B key.

### 3. Model host (ECC-Server, a separate project)

A small Go program, one static binary run by systemd on the game host. It runs `llama-server` (llama.cpp, CPU only)
as its child process. Go never loads the GGUF itself; llama-server does, and the host talks to it over local HTTP. It
is its own project, separate from the training pipeline in this repository, in
`C:\Users\telesphoreo\GolandProjects\ECC-Server`.

- **What it owns.** The pipeline produces classifier bundles; the model host only consumes them. It fetches the
  deployed classifier, verifies it, runs it, and tells Module-ECC what it needs to act on the scores.
- **Who talks to it.** Module-ECC talks only to the model host, over local HTTP on the loopback interface, never to
  llama-server.
- **Install.** An operator installs the binary, a config file (listen address, W&B key, the W&B project and artifact
  name, thread count) and a systemd unit; `ecc-server install` writes the unit.
- **llama-server.** The host downloads the llama.cpp release the bundle names (today b12059,
  `llama-b12059-bin-ubuntu-x64.tar.gz`, the Linux CPU build of the release every parity check ran) and checks its
  SHA-256 before running it. On a machine without systemd (development), the config can name a local llama-server
  instead.
- **CPU budget.** Thread count and window bounds are tuned against the Ryzen 5 3600X, so the game server keeps its CPU.

**Parity.** The host builds the same windows as the pipeline (`src/ecc/chat.py`):
- the same header and line format;
- the same text cleanup (`chat.clean`: NFKC, line breaks to spaces, right-to-left runs in display order, format
  characters dropped);
- the same token counting, through llama-server's `/tokenize` with the same rules as `chat.window_sizes`;
- the same rebuild points.

The end-to-end check replays the case sets and a val sample through the real host and llama-server, and compares
margins with `uv run eval`'s: Pearson at least 0.999, and the same trigger decisions.

**Before W&B is set up.** `ecc-server use <bundle folder>` serves a bundle copied onto the host. It is the same
on-disk bundle the W&B update writes, so nothing else differs.

**The bundle.** The classifier bundle holds the GGUF, the Platt parameters, the prompt format version, the window
bounds, and the decision parameters (trigger, clearing and corroboration thresholds and the window-rule settings).
The pipeline uploads each bundle to W&B as a versioned artifact, and the host serves the artifact version that
carries the `production` alias.

The pipeline writes a bundle as a folder: the GGUF, `bundle.json` (Platt parameters, prompt format version and
header, window bounds, decision parameters, llama.cpp release) and a manifest of SHA-256s.

**Key.** The host fetches bundles from W&B with an ordinary W&B API key, kept in the host's own configuration. W&B has
no Go SDK, so the host resolves the alias and downloads the files through W&B's HTTP API; a spike against a real
artifact proves that before the update path is built. It is
the only W&B credential on the game server, and the plugin has none. The host never uses Telesphorepo. That is how
the weights and thresholds stay private while the code is open source.

**Update.** On a schedule, and at once on a command, the host asks W&B which version carries `production`. For a new
version it downloads the bundle, checks its hash, then swaps it in without a gap in scoring: it starts a second
`llama-server` on the new GGUF, health-checks it, switches scoring and the served decision parameters over together,
then stops the old one. A bundle that fails the hash check or the health check is discarded, the current one keeps
serving, and the host logs why. The host keeps the one previous version on disk for rollback and deletes older ones. A
command switches back to the previous version, which also works while W&B is unreachable; moving the `production`
alias back in W&B has the same effect on the next check.

**Offline.** If W&B is unreachable, the host serves the bundle already on disk and keeps retrying. An unreachable W&B
never stops scoring. Before any bundle has been fetched the host reports not ready and scores nothing.

**Interface.** Over local HTTP the host does two things:

- Scores a message. The request carries the conversation id, the speaker and the text. The host appends the line to
  that conversation's window (prompt format and window bounds from the bundle), runs the classifier, and returns the
  Platt-mapped P(ban) with the bundle version. It runs `llama-server -t 6 -np 4 -c 4096 --slot-prompt-similarity 0.5
  --cache-ram 1024 --no-cache-idle-slots` (4 slots of 1,024 tokens; a prompt is at most 768) and sends `/completion`
  with `cache_prompt`, `n_predict` 1 and `n_probs` 20, so each message reuses its conversation's cached prefix
  (measured: 99.9% cache hits replaying 3 hours of Total Freedom chat). The margin is read from the logprobs of the
  exact " ban" and " ok" token ids; an answer token missing from the top 20 gets the lowest listed logprob.
- Serves the decision parameters and the bundle version that go with the scores it returns.

How it handles traffic:
- Messages of one conversation are scored in order, one at a time; different conversations run in parallel up to the
  slot count.
- A conversation idle for 30 minutes drops its window, and its next message starts a new one.

Setup of the host, its W&B key and `llama-server` is documented for operators.

### 4. Module-ECC training pipeline (this repository)

A reproducible pipeline, one command per step, inputs in their native formats. It produces classifier bundles and
holds no serving code; the model host (section 3) is a separate project.

1. Ingest the inbox (below) into one Parquet message table (id, source, author, time, channel, conversation, text),
   dropping raid accounts; appending adds new messages and never changes an existing message's id.
2. Candidate windows (risk terms plus surrounding lines); everything else is ok by construction.
3. Labeling shards for Codex (gpt-6.1-sol), validated and resumable; regrade sets for known error groups.
4. Merge: first pass < regrade < `corrections.jsonl` (manual fixes, never lost on rebuild), all keyed on message id.
5. Training set (history up to 29 lines, ambiguous lines dropped; current Total Freedom logs, eval case files and
   red-team cases never trained on; the `[trainset] extra` synthetic conversations go to the training split only).
6. Train (LoRA on Qwen3, single-token classifier), merge, export the GGUF, check llama-server parity, fit Platt, and
   assemble the classifier bundle the model host serves (GGUF, Platt parameters, prompt format version, window
   bounds, decision parameters). The deployed GGUF is Q5_K_M with an importance matrix from training prompts and a
   Q8_0 token embedding (which is also the output head). Thresholds are measured with the AVX2 llama.cpp build the
   3600X runs, because builds for different CPUs give different margins.
7. Evaluate: validation, case files, held-out set, red-team set, and triggers per day on real Total Freedom chat.
8. Confident-learning review queue: labeled lines the trained model strongly disagrees with.
9. Slow-check red-team harness against Bedrock.
10. Drop the module's JSONL records into the inbox for the next round.

Code, policy, prompts and synthetic test cases live here. The working copies of data, labels, model weights,
thresholds (the decision parameters in the bundle come from the evaluation results in `data/`), `local.toml`,
`.env` and any file with real chat live in the gitignored `data/` directory of this repository. Telesphorepo
holds the stored copy of data and pipeline artifacts, and W&B holds runs and model bundles (below).

#### Productionalizing the pipeline

**Machine transfer from a blank clone.** On a new Windows machine with any recent NVIDIA GPU (5090, 5070 and
similar), setup is `uv sync` plus `uv run setup`.

- `uv run setup` downloads the pinned llama.cpp release build (release tag and per-file SHA-256 live in the repo) and
  the base model (pinned revision, files hash-verified), checks each hash, and writes `local.toml` with what it
  detected: GPU name and free memory, CUDA, and the paths of the llama.cpp build and the base model. Running it again
  skips whatever is present and verified.
- Steps read the committed pipeline settings and `local.toml` as one configuration. No path is edited by hand.
- `train` measures the GPU's free memory when it starts and sizes the token budget per batch and gradient
  checkpointing from it, so a smaller card trains without config edits. The sizes it chose are logged with the run.
- The pipeline's W&B API key and the owner's Telesphorepo read token go in `.env`. Uploads to Telesphorepo go
  through a device enrolled with `telesphorepo login`, so no write secret is stored in the repository. `setup` names
  each one that is missing, including an unenrolled device. Only the training pipeline uses Telesphorepo.

**Drop-in data inbox.** Raw inputs go into one folder, `data/inbox/`, and stay there. They are Paper server log folders
or archives (current Total Freedom 2026 logs; old Total Freedom 2023-2024 logs), the Creayun export, and Module-ECC's
own daily `YYYY-MM-DD.jsonl.zst` chat records. Each new file is also stored in Telesphorepo (see Telesphorepo below).

- `uv run ingest` walks the inbox, opens archives (`.zip`, `.tar.gz`, `.gz`), and detects each file's format from its
  first non-empty lines, never its name or path. Detectors run in this fixed order, and a file is checked against all
  of them:
  1. **Module record:** every sampled line is a JSON object with the record schema's required keys. This also reads
     `.zst` files.
  2. **Paper log:** the lines carry the Log4j prefix `[HH:MM:SS] [<thread>/<LEVEL>]: `.
  3. **DiscordChatExporter text export (Creayun):** `[M/D/YYYY h:mm AM] <author>` header lines, followed by
     `- Username:`, `- Server:` and `- Message:` field lines.

  The signatures are disjoint. A file that matches two detectors is not guessed at: it is reported with both format
  names, skipped and left in the inbox, and ingest exits non-zero after finishing the other files. A file that matches
  none is reported as an unknown format, with its first lines, and skipped the same way.
- **The source is the server a message was said on, not the export it came in:**
  - Module records name their server in every record.
  - A Creayun export names its server in its `- Server:` fields.
  - Paper logs do not name their server, so the inbox subfolder decides (`data/inbox/<server>/`, for example
    `data/inbox/tf/`). A Paper log directly in `data/inbox/` is reported and skipped, naming the configured Paper
    servers as candidates.
  
  Current and old Total Freedom logs are one server, `tf`. Overlapping exports and a module record of a message that
  is also in a Paper log therefore collapse into one message through the message id.
- **Train/test split:** which messages may train is a per-server date rule in config. For example, `tf` messages from
  `test_from` onward are test-only; this is how current Total Freedom chat stays the real-chat test set. It does not
  depend on which export a message arrived in.
- A file whose content hash is already in the ingest manifest is skipped, so re-dropping a file changes nothing. New
  files append to the message table.
- A file in an unknown format is reported with what was seen, skipped and left in the inbox. It is never guessed at.
- The manifest records each file's hash, source, parser, row count and dropped-account count.

**Stable message identity.** Every message gets an id derived from what the message is, not where it was read from:
source, timestamp to the second, conversation, speaker, text, and an occurrence number for identical messages with
the same values. The same message read from two files (an overlapping export, or a `latest.log` later seen again as
its rotated dated file) gets the same id, and ingest skips ids it already holds. Labels, corrections, windows, roles
and train/val membership key on that id, never on a row number or position. Re-ingesting on another machine, in another
order or with more data leaves every label attached to the same message. Labels and corrections hold ids, labels,
rules and roles but no message text. Case files are labelled conversations and so contain text: the synthetic and
red-team ones live in the repository, and the ones built from real chat live in `data/` with the raw inputs.

**Nothing irreplaceable.** Every artifact is rebuilt by a pipeline step from the raw inputs, or pulled from
Telesphorepo or W&B. The raw inputs themselves are stored in Telesphorepo.

| Artifact | Made by | On a new machine |
| --- | --- | --- |
| Raw logs and exports (the inbox files) | Dropped in by hand; `ingest` stores them | Pulled from Telesphorepo into the inbox |
| Message table, candidate windows, merged labels, training set, review queue | `ingest`, `candidates`, `merge`, `trainset`, `review` | Rebuilt from the inbox and the pulled labels |
| Labels (Codex output, shard indexes, policy and prompt) and `corrections.jsonl` | `label`, by hand | Pulled from Telesphorepo. Rebuilding labels is a full Codex run |
| Case files built from real chat | By hand | Pulled from Telesphorepo |
| LoRA adapter, classifier bundle (GGUF, Platt parameters, prompt format version, window bounds, decision parameters) | `train`, `export` | Pulled from W&B, or rebuilt by `train` and `export` |
| Eval result files | `eval`, `redteam` | Pulled from Telesphorepo, or rebuilt |

A fresh clone plus Telesphorepo and W&B access reproduces the deployable model. `uv run pull` fetches every missing
item: data, labels, corrections, case files and eval results from Telesphorepo, and the adapter and bundles from W&B.
`uv run status` lists what is missing and, for each item, the step or `pull` that makes it.

**Telesphorepo.** The owner's private artifact server (source in `C:\Users\telesphoreo\GolandProjects\Telesphorepo`,
authoritative reference `SPEC.md` there) holds all data and pipeline artifacts: the raw inputs (Paper logs, the
Creayun export, Module-ECC's JSONL records), the Codex labeling outputs and shard indexes with the policy and prompt
used, the corrections, case files built from real chat, and eval results.

- They live in a private raw repo, `ecc-data`. Raw repos are hosted only and keep one file per path with no
  versioning, so every path is immutable and carries its version: raw inputs by content hash, labels under
  `labels/<policy-hash>/…`, and every other file under its content hash or the run that made it. The repo's
  overwrite policy is `deny`. A "latest" pointer file is the one path a glob rule allows to be overwritten, and it
  changes only by `If-Match` compare-and-swap; every other upload uses `If-None-Match: *`, so a path is written once.
- The Codex labeling run is the most expensive artifact and is stored in full: every shard's raw Codex output,
  including any rationales that quote chat, the regrade outputs, the shard index, and `policy.md` and the labeling
  prompt exactly as used, with their hashes. A label version is tied by its path to the policy and prompt that
  produced it. A new machine pulls it instead of relabeling. `label` refuses to mix outputs made under different
  policy hashes into one version.
- The pipeline downloads with the owner's read token from `.env` (`GET /raw/{repo}/{path}`, listing by
  `GET /api/repos/{name}/assets?prefix=`). It uploads through the `telesphorepo` CLI (`telesphorepo publish`) or a
  publish session minted the same way from the enrolled device. Telesphorepo has no long-lived write token: a
  session lasts 15 minutes by default, and the device re-enrolls with 2FA every 90 days by default.
- Each upload checks the returned `ETag` against the file's SHA-256. Files of 64 MiB or more go up as resumable
  chunked uploads, which Telesphorepo verifies by SHA-256 and size when they finish; Cloudflare in front of it refuses
  single request bodies over 100 MB. Uploads are capped by `max_upload_size`, an admin setting that is 2 GiB by
  default.
- `ingest` uploads each new inbox file once, keyed by its content hash, and `pull` restores the files into the inbox.
  While Telesphorepo is unreachable, every step still runs on the local copies, `status` lists the files with no
  stored copy, and `pull` uploads them first, before fetching.

**Weights & Biases.** The project is private and holds run tracking and model bundles, never raw data, labels or
chat. Every `train`, `export` and `eval` run is logged with its config, metrics, loss curve, eval summaries, git
commit (and whether the tree was dirty) and data fingerprints: the Telesphorepo path and SHA-256 of each stored file
it used, and the hashes of the message table and training set built from them. Lineage is complete without uploading
data to W&B. Eval summaries and logged eval output hold ids and scores, not lines. These steps need W&B and stop at
the start with a clear message if the key is missing.

When the key is present but W&B cannot be reached, a run logs offline to `<work>/wandb/` instead of failing.
`uv run status` lists the unsynced runs, and `uv run pull` syncs them first, before fetching. Artifacts from an
offline run are marked unsynced and cannot be tagged for production until they reach W&B. (This default was chosen
while the owner was away; the owner can overturn it.)

- Versioned W&B artifacts: the LoRA adapter and the classifier bundle.
- `export` uploads the classifier bundle as a versioned artifact. The deployed classifier is the version carrying the
  `production` alias. Moving that alias is an explicit action after the ship gates pass, and it is the whole
  deployment: the model host on the game server picks the new version up from W&B on its next check (section 3).
  `export` records the artifact version and run it wrote, and `status` shows which version is in production.

**Continuous evaluation, manual retraining.** New in-game JSONL records and new log exports are dropped into the
inbox. `ingest`, `candidates`, `label`, `merge`, `trainset`, `train`, `export` and `eval` then process only what is
new, using the manifest and message ids to tell. Retraining is a manual command; the pipeline never starts it.

**Developer experience.** The requirements below apply to every step, including `setup` and `pull`. The README walks a
new machine from clone to trained model: `uv sync`, `uv run setup`, `telesphorepo login`, drop files in the inbox,
`uv run pull` if Telesphorepo or W&B holds earlier work, then each step.

**Order of work.** Productionalizing starts after the classifier's base model and recipe are settled, so it does not
delay the model. Until then the spike runs as it is.

Developer experience is a requirement, not polish:

- Each step is `uv run <step>` (`setup`, `ingest`, `candidates`, `label`, `merge`, `trainset`, `train`, `export`,
  `eval`, `review`, `redteam`, `pull`, plus `uv run status`). Steps read paths and settings from one config file; a step needs no
  flags for the normal run, and has at most a few for real choices (which model, which source).
- `uv run status` shows each step's state: done, partly done (with counts), stale (an input changed since), missing
  (with the step or `pull` that makes it), or not started.
- Progress bars with counts, rate and time left for every long step; concise console output; a full log file per
  run in the data folder.
- Ctrl+C stops cleanly: the current item finishes or is discarded whole, state is saved, and the step says how to
  continue. Running the step again resumes where it stopped. Long steps (labeling, training, real-log eval) also
  resume after a crash or a killed process, including a half-written last record.
- A README that walks a new reader from raw logs to a deployed model, one section per step, with what each step
  reads, writes and takes.

### 5. Documentation site

The Plex docs site (`C:\Users\telesphoreo\WebstormProjects\plex-docs`, sources in `src/content/docs`) changes with
the module: `modules/nush.md` becomes the Module-ECC page (`modules/ecc`), covering NUSH raid protection and the chat
safety feature (NUSH always on; chat safety's one switch, that it needs nothing when off; what happens while the model host is down; what it captures, what players experience (shadow mute while a check is pending, a ban only at a verdict), configuration, permissions, the model host (ECC-Server: its setup beside llama-server, its W&B key, how it picks up a new
bundle and works offline) and the Bedrock setup, the `.env` keys, the XenForo evidence posting). The modules index and any links to `modules/nush` are updated.
The Plex API pages document the indefinite-ban API. Generated `dist/` output is not edited by hand.

### 6. Release and migration

Plex 2.0 is released; this ships in Plex 2.0.1 and Module-ECC's first release with exactly one migration for
existing servers, covering everything that moves: the module's data folder and config from Module-NUSH to Module-ECC
(NUSH's own keys, permissions and command are unchanged; the migrated config has chat safety off), and any Plex-side
change the indefinite-ban API needs.
NUSH needs no compatibility shims because none of its names change.

### 7. What exists today (spike) and moves here

The measurements in this spec come from a spike in the Claude session scratch folder
(`...\scratchpad\spike`). It moves into this repository (code, `policy.md`, prompts, synthetic and red-team cases)
and into the gitignored `data/` directory (messages Parquet, labels, Codex outputs, real-chat case files, real-log results, model
files) once the training run that uses those paths has finished. The baseline numbers and the red-team result are
re-checked from the moved copy before any module work depends on them.

## Ship gates

- Fast check beats the prompted Qwen3-4B baseline on real Total Freedom chat it never trained on: all five known
  offenders (listed in the private local.toml) triggered, and fewer players triggered over 80 days than the baseline
  triggers at its own threshold for those five; better recall on the held-out set and the steering/grooming cases.
- Slow check: zero failures on the red-team set; at least the measured accuracy on the role tests.
- Measured CPU cost per message on hardware matching the 3600X, with the game server's load in mind.

## Decided

Policy and roles as above; staff bypass; fast check local, slow check Kimi K3 on Bedrock with no local fallback;
shadow mute at trigger with no ban entry until a verdict; the pending state kept and persisted by Module-ECC; at a
verdict, permanent bans by UUID, name and IP through `indefbans.yml` and a random-delay kick; no chat clearing (clearing
chat tips off the player and everyone else); slurs out of scope; code open source, weights/thresholds/data private; JSON (not TOON) for model output; datamarking always on;
non-triggered participants bannable with corroboration; staff evidence posted to a XenForo staff node, never in the
player-visible reason; ECC on main only; anvil names and `/nick` captured in the first release; pipeline developer
experience as specified; NUSH always on with no switch; chat safety's one switch, needing nothing when off; one rule for the model
host being down at any time (scoring pauses, messages queue, background retry); the training pipeline
(this repository) and the model host (ECC-Server) are separate projects, the pipeline producing bundles and the host
only consuming them; Telesphorepo (private raw repo `ecc-data`, overwrite denied, every path immutable and versioned
in its path) for all data and pipeline artifacts: raw inputs, Codex labeling outputs with their policy and prompt,
corrections, real-chat case files and eval results; Weights & Biases (private project) for run tracking and model
bundles with the `production` alias, never raw data or labels, each run recording the Telesphorepo paths and SHA-256
of its data; only the training pipeline uses Telesphorepo; the Minecraft plugin never touches W&B; the model host
fetches the `production` classifier bundle from W&B with an ordinary W&B API key, verifies and swaps it, serves scores
and the decision parameters to Module-ECC over local HTTP, serves offline from the bundle on disk, and keeps one
previous version for rollback; one drop-in inbox with content-based format detection in a fixed order, a two-detector match or an unknown
format reported and skipped, and Paper logs' server named by their inbox subfolder; sources are servers (current and
old Total Freedom are one server, `tf`) with a per-server date rule for test-only data; stable message ids;
W&B runs fall back to offline logging and sync on `pull` (an owner-overturnable default); retraining manual.

## Left to the builder

Package layout and names inside each project; the config key name of chat safety's switch; the message id hash and
manifest formats; the number of lines each detector samples; how `train` maps free memory to a token budget; the database schema for the pending state; JSONL schema details; how the Java side calls the model host (local HTTP) and Bedrock (Java SDK); the model host's
package layout, port, update-check interval, command interface and bundle layout on disk; the path layout inside
`ecc-data` beyond the rules above, and the "latest" pointer file format;
whether the pipeline uploads through the `telesphorepo` CLI or mints publish sessions itself; the exact normalization tables; retry backoff; message templates' wording beyond the meaning above.

## Open

- Window rule, thresholds and window bounds: from the fine-tune evaluation.
- How long the module keeps its JSONL records.
- The Telesphorepo instance URL. The owner creates the private raw repo `ecc-data`.
- Raising Telesphorepo's `max_upload_size` above 2 GiB if a stored file, such as a large log archive, exceeds it.

## Later (not in the first release)

- Automatic retraining when enough new data arrives. Retraining stays a manual command.
- Periodic slow checks of high-risk conversations that did not trigger (private messages with a self-declared
  minor), as a backstop if classifier weights ever leak.
