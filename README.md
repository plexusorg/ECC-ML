# Module-ECC pipeline

This pipeline takes raw Minecraft chat logs and produces the fast-check classifier that ECC-Server runs for
Module-ECC on the game server, plus the measurements that decide whether it ships. Each step is one command.
`uv run status` shows where you are.

```text
raw logs -> transcript -> read -> line-labels -> labels       (the labels)
trainset -> train -> export -> eval -> bundle -> promote
```

## Setup

1. Install [uv](https://docs.astral.sh/uv/), then run `uv sync`. The project requires Python 3.13 and its Unicode 15.1 tables.
2. Private data lives in `data/` (gitignored, backed up whole). On a new machine, `uv run restore` fills it from the
   newest backup (see backup), including `data/local.toml`. The chat logs are in `data/raw/`. What the pipeline
   rebuilds (models, bundles, eval scores, the train and val tables, logs, tools) goes to a work folder outside the
   repository, never backed up. `data/local.toml` holds the machine's paths:

   ```toml
   [paths]
   work = 'C:\path\to\ECC-ML-work'          # what the pipeline rebuilds
   ```

3. Copy `.env.example` to `.env` (gitignored) and fill in the W&B key for `bundle` and `promote`. `backup` and
   `restore` use the telesphorepo CLI as the account this computer is logged in as (`telesphorepo login`).
4. For `read`, install the Codex CLI and log in (`codex`).

Every setting lives in `ecc.toml`, and `data/local.toml` overrides it. Each run writes a full log to `<work>/logs/`.
Ctrl+C stops a step and keeps all finished work. Run the same command again to continue from that point.

## Words

- **The classifier**: the model that scores each chat line, XLM-RoBERTa-large with eight outputs: the seven
  categories and hold.
- **Categories**: the seven problem types a line can show (ask_contact, give_contact, ask_personal, give_personal,
  off_server, sexual, romantic).
- **Server log**: a log file of the Total Freedom server (Paper); **Creayun** chat comes from Discord exports.
- **Transcript**: the readable chat built from one server log file, or one Creayun server on one day.
- **Session**: one player's lines in one transcript, `<transcript>|<account>`.
- **Line labels**: each line's reader marks, merged into `lines-labelled.parquet`. **Hold labels**: what the
  classifier's hold output learns, in each batch's `hold.parquet`.
- **Coverage**: whether a line fits the classifier's input: `scored` or `rejected` (preparation would shorten it, or
  it does not fit in the token budget); `context` for a line that is only context.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/ecc/` | The steps. |
| `data/` | Private: gitignored, backed up whole to Telesphorepo. Everything the pipeline cannot rebuild. |
| `data/raw/` | The chat logs: the Total Freedom server logs (`tf/`, `tf-old/`) and Creayun's Discord exports (`creayun/`). |
| `data/policy.md` | The rulebook. The slow check reads it. |
| `data/v4/` | The v4 reader batches (`batches/`: parts, reads, hold labels), the selected line-label tables, the readers' questions, and what the bundle is built from besides the model: its settings and the judge's texts (`data/v4/README.md`). |
| `data/models/base/<repo>@<sha>/` | The classifier's base model at its pinned commit (base model, below). |
| `data/v4/models/ranker/` | The trained classifier that ranks the training lines (`[train] ranker`); its `input.json` holds its base revision and `trained_tokens`. |
| `data/rulings/` | The owner's saved rulings, their reviewed derivatives and the scripts that built the ruling pages. Select the judge's rulings with `[bundle] rulings`. |
| `data/local.toml` | This machine's paths and private settings. |
| `<work>/` | `[paths] work`, outside the repository, never backed up: models, bundles, eval scores, the train and val tables, logs, tools, and notes. |

## transcript

`uv run transcript BATCH` builds the parts of a reader batch from the raw chat and checks them. A batch is a folder
under `[transcript] batches`: `manifest.json` (its `kind`, `server_log` or `creayun`, and `transcripts`: its sources
and what was built from them), `parts/`, `reads/<read>/` and, once labelled, `hold.parquet`.
`--files GLOB...` starts a server-log batch from log files under `[transcript] logs` (`tf/2026-11-1*.log.gz`);
`--creayun` starts a Creayun batch of every server and day of `[transcript] messages`
(`data/v4/creayun-messages.parquet`): Creayun's chat and private messages read from its Discord exports, one row per
message with its time to the minute, each numbered by `row`.

- **A transcript** is one server log file (one uptime or one day), or one Creayun server on one day, one line per
  event: `<id> <time> <channel> <speaker>: <text>`. `L<n>` is line n of the log file (for Creayun, the message whose
  `row` is n); `R<n>` is line n of that day's ECC-Server record (`[transcript] record`, v3 or v4), from which signs,
  books, item names and lines typed while muted come (a sign or item name set by command, which a v4 record also
  holds, is read from the log). Ids stay the same across rebuilds. A part's header calls its transcript a "Dossier",
  the word the readers' questions use.
- **Muted and held lines.** NUSH logs a quarantined player's chat line as `[NUSH] <line>`, hiding some and marking the
  next with ` (+N more)`; it is a `muted` line by the account, the mark removed, unless the day's record holds the
  same line. A chat or guild line Module-ECC held reaches the log only when it is delivered, on the module's
  `Plex-ECC-Actions` or `Plex-ECC-Asker` thread; it is read at its `Held <account>'s public chat: <text>` alert, where
  it was held, with the name the delivery shows. When its file holds no delivery (a line never delivered, or delivered
  after midnight), it is the line ECC-Server received as the v4 record holds it, with the shown name and raw text the
  judge was given (the record's line of the same channel and text near the alert, by the account's UUID from the
  log's `UUID of player` line); only where the record does not hold it is it read from the alert alone, its account
  as the name and the alert's text.
- **Text.** Each text is prepared as `src/ecc/inputs.py` states, the rules ECC-Server applies to the raw text the
  plugin sends (bundle.json `text`): colour codes removed, line breaks as ` / `, whitespace collapsed, links over 60
  characters and texts over their channel's limit cut, addresses as `[ip]`. A text empty once prepared makes no line.
- **Speakers.** The name as shown (colour codes, icon codes and `[Not Secure]` removed), with `=account` when the shown
  name does not contain the account it resolved to through the log's player tracking (`src/ecc/parse.py`; only an
  online player is ever the answer). A log file that continues a run starts from the player state at the end of the
  file before (who is online, nicknames, message partners, pending `/sudo`, the accounts' UUIDs), so a player online
  since before midnight keeps their account; a file that starts a run starts with nobody known. A chat name ends at
  the first ` » ` unless it shows no player and a name ending at a later one does (a tag ending in `»`). A chat line
  the day's record holds takes the record's account. Module-Guilds' guild chat is read from its
  `[GUILD - <guild>:<id>]` console line on any thread; a `[Guild]` tag before a chat name is part of the shown name of
  a public chat line.
- **Commands.** EssentialsX's `/msg`, `/r`, `/mail` and `/me` are read under every alias its plugin.yml gives them, and
  a typed player name resolves as EssentialsX's `matchUser` resolves it (UUIDs, name prefixes, nicknames). A command
  EssentialsX sends to no one (a `/msg` to an offline or unknown player, a `/r` with no one to answer) makes no line,
  and neither does a `me`, `say`, `msg`, `reply` or `mail` command NUSH cancelled for a quarantined player (its feed
  line `[NUSH] <account>: <command>` follows it, or the feed showed one of that player's lines in the 10 seconds
  before). `/mail sendall` is one line to `*`.
- **Addresses.** None is written. A join shows `from addr-NN`, an alias numbered by first appearance within the
  transcript; IPv4 and IPv6 addresses inside any shown field become `[ip]`.
- **Authentication.** Every login, register, password or `/shib` command (typos and plugin prefixes included) is
  dropped with its arguments, as is any line whose text is such a command, and any line in which its speaker repeats a
  word they typed to one of those commands in the same log (for Creayun, in any export: the word is replaced by
  `[secret]`, and the line is dropped if that misses it). For Creayun, e-mail addresses, keys and ids in links, tokens,
  passwords stated in a sentence, card numbers and key-like strings are replaced by placeholders. Only counts are kept.
- **Left out:** staff chat, console output, moderation notices. A speaker's exact repeats on one channel within 60
  seconds fold into one line, `[sent N times until T]`; a log file that continues a run (`transcript.runs`: a file
  without the startup line continues the uptime of the file before it) folds copies of the file before's lines too.
- **Parts** hold at most 12,000 tokens (characters / 3), cut at the longest quiet gap in the last 40% of each part's
  room; a later part starts with the 30 lines before its cut and a `--- part N begins here ---` marker.

On a batch that has parts, the step rebuilds them, cut where the saved parts were cut, and compares them byte for
byte with the saved ones. It writes only when the batch has no parts or no read has answered them; otherwise a part
that differs is reported and the saved parts stay, with the count of parts `--rewrite` would read again.
`--rewrite NAME` writes them anyway: the saved parts and manifest, and every read's answers to a part whose body gains
a line or shows one of its lines with another speaker or addressee, go to `[transcript] history/NAME/<batch>/` with
`rewrite.json` (the parts rewritten, the parts to read again with the lines they gain and the lines that show another
speaker, the lines no longer built); then run `uv run read` to answer the parts whose answers moved. A part whose
body changes only in text keeps its answers. Every run ends with the check, and `--check` runs only the check: no
IPv4 or IPv6 address, no authentication command as a line's text, no authentication word in its typist's own lines,
and for Creayun none of the secret shapes. It prints line ids and lengths, never text. The server-log batches take
seconds, Creayun about a minute.

## read

`uv run read BATCH READ` has the Codex CLI read every part of a batch that has no answer from that read yet, each part
in its own ephemeral session (read-only sandbox, an empty working folder in `<work>/read/`, no user config or rules),
`--parallel N` at a time up to the reader's `max_parallel`. A read is named `<reader>-<copy>` (`codex4-a`, `codex4-b`):
every copy of a reader sends the question its `[read.readers.<reader>]` entry names in `[read] questions`, then the
part. A "marks" reader answers the seven categories on any line of the part, each mark's category in its `kind`
field; a "details" reader (`codex5`) is asked, per part, only the lines and categories the batch's line-label table
marks. Each answer is checked (JSON only, ids of the part, known categories and details) and saved to
`reads/<read>/<part>.json`; every attempt goes to `reads/<read>/log.jsonl` and an answer that fails the check to
`reads/<read>/invalid/`. A part that fails the transcript check is never sent. After `[read] stop_after` usage-limit
answers in a row the run starts nothing new; run it again later to continue.

- `--disputed` reads only the parts the batch's `disputed.json` lists: the third read (`codex4-c`).
- `--parts a,b`, `--limit N` narrow the run; `--out DIR` saves the answers elsewhere.
- `--dry-run` writes each prompt beside its working folder and shows the command, without calling Codex.

## line-labels

`uv run line-labels BATCH...` builds each batch's line-label table from its reads: `lines-labelled.parquet` and
`disputed.json`. Select the table versions with `[line_labels] tables`. Keep the reader parts and manifests under
`[transcript] batches`. To change labels, use `--out NEW_FOLDER` for all selected batches, verify the new tables, then
select that folder. The command keeps an identical existing table and refuses to replace a changed one.

Each row represents one player line with the seven categories, the account, the session
(`<transcript>|<account>`), up to 5 earlier lines of its conversation within 5 minutes, and the split. A server-log
row also has its run (one server uptime over its log files; rows are in run order) with `time` counting from the
midnight of the run's first day, and what the plugin sends for the line: `raw_text`, `raw_shown` and the `copies`
folded into it. Per category, from reads a and b (`[line_labels] reads`): 1 when both marked the line, -1 when one
did, 0 when neither; on a part the third read (`[line_labels] third`) answered, two of three decide. A line a read
listed as unsure and no read marked is -1 on every category; off_server is 0 on a line whose links are all on
`[line_labels] allowlist` after meaning normalization. The original reader value remains in `off_server_raw`. A line
holding a text of `[line_labels] left_out` is -1 on every category. `disputed.json` lists the parts, and their lines,
a third read should cover.

A server-log batch needs both reads on every part. Its transcripts are rebuilt from the logs for each line's account,
raw text and fields, and the build stops unless every rebuilt part is byte for byte the part the reads read (run
`uv run transcript BATCH --rewrite NAME` to write the rebuilt parts). A Creayun batch takes the parts both reads have
answered, rebuilt from the message table and compared byte for byte; the account is the shown name, and its rows are
in transcript order, each transcript's parts by part number. The split is the one period rule, `[periods]`: a log file
of the `owner` batch (`full`) is `train` before `val_from`, `val` before `test_from` and `test` from then on, by its
date; every line of the other server-log batches is `train`; a Creayun transcript is `test_creayun` when a hash of its
name falls in `[line_labels] test_share` (15%), else `train`. `--out DIR` writes the tables to `DIR/<batch>/`. All four
batches take about three minutes.

## labels

`uv run labels [BATCH...]` makes the hold labels: from the line-label tables of the `[labels] batches`, the cases a
judge reads and the sessions the owner ruled, then from them and the judge's reads each batch's
`data/v4/batches/<batch>/hold.parquet` (transcript, id, session, session_ruling, session_source, hold, ban_here). It
writes `<work>/labels/cases.json`, `<work>/labels/owner.parquet` and `<work>/labels/sessions.parquet` (every session's
ruling and `by` line). The `hold.parquet` it replaces is first copied to
`data/label-history/hold-<the run's time>/<batch>/`; run `uv run backup` after.

- **The owner's sessions.** `[labels] rulings` lists his rulings files in order; the first that rules a session gives
  its ruling (block and warn are hold, no ban is not ban) and its `by` (when it names none, the next file with the same
  ruling that names one). An entry lands on the session of each of its case's own lines, the player's lines of the
  case as the owner ruled it (the ruling page, or the rendered case the entry's `cases` names); a case whose lines are
  now on two sessions rules both, and the step lists such cases. An entry with no case lands on the judged lines whose
  speaker has the name in its key. A file with `take = "trusted"` gives only its trusted sessions, one with
  `take = "unsure"` only its unsure ones; an unsure entry of any other file leaves the session to the next file. The
  step stops when a rulings file is missing or an entry lands on no judged line.
- **The cases.** One per session with a judged line a category marks (1), except the owner's own, those whose judged
  lines are all spam raids (`left_out`) and those any rulings file rules. A case shows each marked or disputed line of
  the player with the lines around it, as `data/v4/bundle.toml [judge.render]` sets for the live judge (a private
  message within its own conversation), the player's lines `>>`, tagged by `[labels] tags`: the readers' categories
  and disputed categories with whom the line is aimed at, or no tags. A case over the render's `max_chars` is cut to
  its marked lines, then to a narrower window, then to as many marked lines, spread evenly, as fit. The owner's ruled
  cases dated after `[labels] examples_until` are added as `[labels] cases` renders them, with id `owner|<case id>`;
  they measure the judge and label nothing. The saved judge reads are bound to the hash of the cases as written, so a
  case keeps its fields `dossier`, `kinds` and `kind`.
- **The examples** are the owner's ruled cases `uv run bundle` packs, dated up to `[labels] examples_until`. The step
  stops when one is dated on or after `[periods] test_from` or its lines are on a session that is a case.
- **The reads.** `[labels] answers` holds a folder per read of `[labels] reads`, each answer (ruling, by, sure) keyed by
  the SHA-256 of the case text the judge read. Answers are added as new files and never changed; a later answer to the
  same text stands. The step stops, naming them, while a case lacks an answer in any read.
- **The session rulings.** The owner's own sessions are `own`; a session he ruled takes his ruling (`trusted`,
  `unsure`, `nothing`, `hold`, `ban`, `not ban`); a session whose judged lines are all spam raids is `spam raid`; any
  other the reads' ruling where they agree, else `split`. A ban's `by` is a valid explicit owner line, or a line all
  selected reads name exactly, with every answer sure and `ban`. The line must be a judged line of the session.
  An inferred line has `by_from = "judge reads"`, including when the owner ruled ban without a line. For those
  owner bans, select `[labels] reads` for his rendered cases that are cases here (after `examples_until`), and
  `[labels] by_reads` for the others (his examples). The step stops while an answer is missing. A separate
  `censor_after` marks possible intervention: use the valid explicit owner line, else the earliest valid line from
  those reads, else the first line marked contact, sexual or romantic, else the first marked line. If an explicit
  owner line is invalid, use the marked-line fallback for censoring. The censor does not establish a ban deadline.
  `censor_from` records its source. For each judge-derived boundary, retain the case text hashes in `by_hashes` or
  `censor_hashes` so the frozen reads can be checked.
- **The hold labels**, one per judged line: `nothing` and clean sessions hold 0; `hold` and `ban` sessions 1 on lines
  with a category at 1, -1 on lines whose only non-zero categories are -1, else 0; a ban session -1 on every line after
  its `censor_after`. Mark only a supported `by` line as `ban_here`; every other session -1 on lines with a non-zero
  category, else 0; spam-raid lines (`left_out`) -1. A -1 line is out of the hold loss.

The three batches take under a minute.

## base model

The classifier starts from `Roblox/roblox-pii-classifier-v2` (`[train.classifier]`), pinned to a commit sha.
`uv run python -m ecc.base fetch` downloads its pinned files when it has no copy in `data/models/base/<repo>@<sha>/`,
through a cache of its own in the work folder, so the Hugging Face cache is not written; it checks each file against
the hub's size and SHA-256 and writes `source.json` (repo, revision, each file's size and SHA-256). Training reads the
copy, never the hub (`ecc.base.path`), and refuses a copy of another repo or revision or with a file of another size.
`uv run python -m ecc.base check` hashes the copy against its `source.json` and checks that the ranker's `input.json`
names the pinned base.

## trainset

`uv run trainset` builds the inputs the classifier trains on with `src/ecc/inputs.py`, the code export and the bundle
build the served inputs with, into `<[train] out>/inputs/` (`<work>/train/inputs/`), about 10 minutes of CPU.

- **The server-log table.** Every row of the `[train] owner` batches (full, old, old2) in one table in arrival order
  (by run, its log files in log order, each file's rows as stored), so a run whose log files went to two batches is
  one stream, as ECC-Server receives it. The step stops when a batch's stored order is not its runs' arrival order
  (time running backwards, a log file out of order or in two blocks), when a merged run's time runs backwards, and when
  two log files of one name (from two logs folders) would share a transcript or a run. A categories-only batch
  (`[train] categories_only`, Creayun) is its own table in its line-label table's order. The model text uses the same
  meaning normalization as serving, then redacts revealed addresses and replaces allowed links with the canonical one.
  Source records, repeat keys and judge text keep their prepared form.
- **What it writes.** `server-log.parquet` (the server-log table's rows: batch, transcript, id, run, coverage);
  `lines.parquet`, the lines the classifier trains and is scored on in training row order (the judged server-log lines
  in server-log order, then every categories-only line), with batch, run, the row in the server-log table
  (`server_log_row`), split, coverage, the seven categories and the masks (`category_mask`: out of the category loss);
  the classifier's input ids of every line of `lines.parquet` at each length of `[train.classifier] tokens` and at the
  length the ranker (`[train] ranker`) was trained at (`classifier-<N>.npy`). A ragged array is one flat `.npy` and
  its `.offsets.npy`: row k is `ids[offsets[k]:offsets[k + 1]]`.
- **Coverage.** Each row's coverage is `scored`, `rejected` or `context`. The source raw text decides whether
  preparation would shorten a target; Creayun raw text comes from its stable source row. A target whose rendered
  string with its role header does not fit in the coverage length is rejected too. Rejected rows keep their keys,
  train no loss, and never enter later context.
- The tokenizer is the pinned base model's copy (base model, above). `inputs.json` records what the inputs were made
  from, the source code hash, the input manifests, the lengths, the primary `coverage_tokens` and the counts. The
  primary length is the first `[train.classifier] tokens` entry. The step reuses current inputs; use `--force` to
  build them again. Ranking, exported graphs and torch scores also record their builder's source hash.
- **The ranking.** `uv run trainset --rank` also scores the candidates for hard negatives with the ranker
  (`[train] ranker`, a pinned trained classifier) at the length it was trained at, bf16 on the GPU: the train lines
  with every category at 0 and not category-masked, outside `[periods] owner` (old, old2 and Creayun's train split).
  The score is the larger of the hold logit and the largest category logit; `<work>/train/rank.parquet` holds each
  candidate's row of `lines.parquet`, its logits and score. Rejected rows are not candidates.
- **The training lines.** Once every server-log batch has its `hold.parquet` (labels), `uv run trainset` writes
  `<work>/train/training-lines.parquet` (row of `lines.parquet`, batch, transcript, id, reason, hold, score) in
  training row order: every train line of `[periods] owner` (`all`); of the other server-log batches every hold = 1
  line (`hold`), every unmasked line with a category at 1 or -1 (`non-zero`), the top `[train] share` (10%) of each
  batch's candidates by score (`hard`) and a random share of its other candidates (`random`); of each categories-only
  batch the same with `[train] categories_share` (5%), every line at hold -1 (out of the hold loss). A line out of
  both losses (category-masked, hold -1) is dropped. The random draws are `default_rng([train] seed)` over each
  batch's other candidates in row order, one stream for the server-log batches in `[train] owner` order and one for
  each categories-only batch. The step stops on a training line outside the train split, a candidate without a score,
  a judged line without a hold label, or a `rank.parquet` made from other inputs or another ranker.

## train

`uv run train classifier --tokens N` trains the classifier on the trainset's inputs cut to N tokens, the length it
will serve at (N must be the input set's recorded `coverage_tokens`), into `<[train] out>/classifier-<N>/`. Use the
GPU for training. To select another serving length, change the first `[train.classifier] tokens` entry and rebuild
the inputs.

- **The model.** `Roblox/roblox-pii-classifier-v2` (the pinned copy, base model above) with a fresh head of eight
  sigmoid outputs, the seven categories and hold. A category at -1 (one read marked it, or a read was unsure)
  contributes no loss for that output. A line with `category_mask` is out of the category loss; a line at hold -1 is
  out of the hold loss.
- **The lines.** Those `<[train] out>/training-lines.parquet` names: the owner's lines with their hold label from each
  `[train] owner` batch's `hold.parquet`, which must label every judged line in the inputs, then the categories-only
  lines (Creayun) at hold -1. The step refuses a training line outside the train split and inputs that are not
  current.
- **The recipe** (`[train.classifier]`): AdamW, a linear warm-up then a linear decay to 0, gradients clipped, bf16
  autocast over fp32 weights, batches of length-sorted chunks, a fixed number of epochs and one seed. The run records
  GPU use and warns about slow step windows. Non-finite losses or gradients stop training before another optimizer
  update.
- **What it writes.** After the last epoch the classifier scores the owner's validation and test lines and Creayun's
  test_creayun lines in fp32, as export does: `scores.parquet` and `creayun.parquet` (transcript, id, period,
  coverage, `classifier_<out>`), which `uv run eval` reads. Beside them the model (`config.json`,
  `model.safetensors`, `tokenizer.json`), `train.json` (each epoch's loss, time and memory, the GPU readings, the
  validation hold metric) and `input.json`: the run's description, `max_tokens` (which export and the bundle cut the
  input to), the base revision, the settings and what the run was made from.
  The run saves the model and `trained.json` before scoring. If scoring fails, run the same command again to finish
  from the saved model. The run checks the data, settings, source code and each saved model file's size and SHA-256.
  A changed or incomplete saved model is refused. Use `--force` only to replace that run and train again.
  After each epoch, `epoch-<N>/` holds the model, optimizer, scheduler and random state, with file hashes in
  `complete.json`. Run the same command to resume an interrupted training run from its last verified epoch.
  The trainer preserves incomplete checkpoint files beside the completed epochs.
- `--epochs E` and `--limit N` make a smoke run (`-smoke`). `--limit` selects up to N training lines, divided between
  server-log and categories-only rows when available, and scores the first 2,000 lines of each scored set. `--force`
  trains again over a finished run.

## export

Run `uv run export` to convert the completed classifier (`[export] classifier`) into one OpenVINO graph. It preserves
the trained input: raw preparation, model meaning, allowed links, complete-target checks, causal context, role names
and token budget. It scores every covered target with the classifier and rejects an incomplete target before folding,
context or judgment.

Each training run has a slot at `<[export] out>/<run folder name>/`:

1. `models/classifier.xml`, `classifier.bin` and `graphs.json`: fp32 weights, dynamic batch and sequence dimensions,
   and graph metadata. The conversion checks real Torch outputs on several sequence lengths.
2. `torch.parquet`: all eight logits (`classifier_<out>`) for every covered judged validation and test key. Rejected
   keys remain in the table with absent logits. The selected immutable table supplies the full causal runs.
3. `thresholds.json`: the hold cuts `src/ecc/calibrate.py` sets, on validation only. `1pct` and `2pct` hold that
   share of the covered validation lines. `offenders` is the highest cut that holds, in every validation session with
   a hold-positive line, a positive line or its ban line at or before the session's first ban line, less the allowed
   CPU logit error. `recall`, the served default, is the lower of `2pct` and `offenders`; `ceiling_ok` is false when
   `offenders` is under `2pct`, and bundle then refuses the slot. Category thresholds are hints. They do not clear a
   hold. The file records exact model, table and hold-label sources. It reports model-held, input-rejected and total
   traffic counts separately.
4. `parity.json`: the actual CPU graph outputs against Torch on identical inputs, for all eight outputs and all named
   thresholds. Run `--full` for all validation and test keys. Use `--sample N` to change the sample size or `--files N`
   to select the first N files. Use `--no-parity` to skip this check. The default sample includes every covered target
   near a threshold and a reproducible random sample.

Export and bundle refuse smoke runs, research models, missing hold-positive training evidence and changed model
bytes. Completed training records must match the model files and training source. The saved inputs must match the
served token budget. Input, score and threshold records detect changed source data or input rules. Keep numerical
proof fixtures separate from release artifacts.

## eval

`uv run eval RUN [RUN ...]` measures runs on the owner's validation and test weeks and on Creayun's held-out lines,
and writes `<[eval] out>/<run>[+<run>...]/tables.md` and `eval.json`. A run is a folder of scores, named under
`[export] out` or `[train] out`, or given as a path (a name found under both is refused):

- an export slot: `torch.parquet` and `thresholds.json`. The score and threshold provenance must match. Each covered
  target uses its hold logit at the diagnostic or recall thresholds. Rejected targets are blocked without a model
  decision. The report names the classifier's folder.
- a training run, a folder of `train classifier`: `scores.parquet` and, when it scores them, `creayun.parquet`. The
  run holds a line when its hold logit reaches the score that holds each share of `[eval] holds` of the run's own
  validation lines.

Each owner period must include scores for every judged line in its line-label table. Evaluation refuses incomplete
scores, including smoke runs, even when their filenames match a full run.

The report shows each run after the first beside its difference from the first, paired on the same resamples.
List the export slot first to compare with it. Use `--labels` to select a label set. By default, measure each
set whose files exist. The `forward` set uses the frozen labels and ban timing from `[eval] forward`. The `new` set
uses the hold labels in each `[eval] batches` folder and derives supported ban timing from the configured rulings
and saved reads. Evaluation checks that the resolved keys, hold labels and session fields match `hold.parquet`.
The batches' line-label tables supply the splits, categories, category masks and line order.

The report separates input rejections, model holds and total blocks. It gives model recall on all positive lines and
on covered positive lines, plus total blocked recall. A target blocked before scoring is never counted as a model
detection. The other measures, per period and threshold: blocked recall of the hold = 1 lines (all, in owner-ruled
sessions, by category), the share of lines held, hold = 0 lines held per 1,000, the marked hold = 0 lines held, clean
sessions of 20 or more lines with a line (and with 5 lines) held, the owner's ban, hold, nothing and not-ban sessions
with a line held, and his ban sessions whose ban line, or a line by it, is held. On Creayun's held-out lines, for runs
that score them: each category's lines over the hold thresholds and over the category's own output at the score that
flags `[eval] category_share` of the owner's covered validation lines, and the unmasked all-zero lines over each.
Creayun tables show total blocks and model holds separately; each category's own-output table uses covered lines
only. Beside them, the hold labels of the training period's lines with each category at 1. Intervals are 95% over
`[eval] resamples` resamples of the log files (Creayun: of its transcripts), seeded with `[eval] seed`, thresholds
fixed.

`--validation` reports the validation weeks only: every choice between runs (such as the classifier's length) is made
there, and the test weeks are reported once, after it. A run takes seconds, on the CPU.

## bundle

Run `uv run bundle --no-upload` to build and check the bundle in `<work>/bundles/<version>/`. The build reads
calibrated thresholds from the export slot, and the private judge settings and texts from `[bundle] private`.

Before creating or reusing a bundle, the build refuses a slot whose `ceiling_ok` is false: a classifier that must hold
more than 2% of validation lines to hold every validation offender in time. It then checks the required accounts in
private `[eval] known_offenders`. Each account must have judged owner-positive hold targets in the selected export
batch. Every owner-ruled session of theirs must have a line, a positive line or the first ban line, held by the real
CPU graph at the calibrated recall threshold at or before that ban line. Missing bindings, rejected inputs, non-finite
logits or a missed session block the build and its upload. Read the private result in the export slot's
`known-offenders.json`.

- `models/`: `classifier.xml`, `classifier.bin` and `classifier_tokenizer.json`.
- `bundle.json`: the graph and tokenizer (key `classifier`) with their hashes, the eight ordered outputs, the
  `categories`, the exact preparation and input rules (`context`, `classifier`, `text`, `links`, `coverage`,
  `model_text`) and their hash (`rules.sha256`), the validation thresholds (`thresholds.hold`,
  `thresholds.categories`) and their provenance, the judge's settings (`judge.checkpoints`, `judge.render`) and a
  worked example. ECC-Server uses `recall` by default.
- `judge/question.md` and `judge/examples.json`: the current private judge texts and owner rulings.
- `README.md`: how to use the bundle. `manifest.json` records every other file's SHA-256 and size.

The version includes a hash of the content. The build copies files into unique staging, checks their bytes and
preserves independent owned copies. The reference loader checks the bundle's layout, input-rule hash, output order
and model-file hashes.

The sibling `<version>-vectors/` folder contains real private wire records, folded copies, preparation examples and
expected outputs. The checks compare exact token ids and contexts with the training builder. They compare the real
CPU graph outputs with Torch and check all named threshold decisions. Run the independent scorer with
`uv run python -m ecc.reference BUNDLE IN.jsonl OUT.jsonl`; each output record carries its `coverage` (`scored` or
`rejected`), the `classifier` logits, the `held_<threshold>` decisions, the `verdict` and the `categories` marked.

Without `--no-upload`, the command uploads the verified bundle to the private W&B artifact project. A separate
promotion selects the served version.

## replay-export

`uv run replay-export` writes the test weeks as `ecc-server replay` reads them, into `[replay] out` (chat text, in the
work folder only), to measure a bundle's classifier and judge together before it ships:

- `lines.jsonl`: every run of `[export] batch` that holds a test file, as one stream each: the join and leave lines of
  its log files as `presence` events, and every line as the plugin sends it, prepared and raw, each copy the
  transcript folded into a line sent as a line of its own (`copy_of`). A line's player is its resolved account, so a
  player who shows several names is one player. Every line carries its `coverage` (`scored`, `rejected`, `context`,
  `fold` or `dropped`). A judged line of a test file carries its `session`, the verdict and `categories` of the torch
  scores in the export slot of `[export] classifier` at the thresholds `export` set there (`hold` when the classifier
  reaches the validation recall threshold; `reject` for a rejected line); the owner's own sessions (`[replay] owner`)
  are trusted; every other line is context only. The step stops when a judged test line has no torch score, or when
  the slot's thresholds were set from another classifier or different score inputs.
- `judge/`: a `-fold` folder: the question with its added sentence, the answer format, and the owner's ruled cases
  dated before `[periods] test_from` with his rulings (`data/v4/judge/`, `[bundle] rulings`), no saved answers.

## promote

`uv run promote [version]` moves the W&B `production` alias to an uploaded bundle (default: the one `bundle` built
last). That is the whole deployment: every ECC-Server picks the new bundle up at its next check or at
`ecc-server update`, verifies it and swaps it in. Promoting an earlier version rolls back the same way.

## backup

`uv run backup` copies the whole `data/` folder to the private `ecc-data` raw repository on Telesphorepo
(`telesphorepo.toml`) as gzipped tars and `manifest.json` with each file's SHA-256, size and modification time. A
folder of at most 64 MiB is one tar; a larger one is split into its subfolders by the same rule, one tar per file over
64 MiB, and one for its other files, so a change uploads only the tar that holds it. The work folder and `.env` are
never included. The repository keeps one file per path and a publish replaces it, so each backup goes to its own
`backups/<UTC time>/` folder, uploaded with `--if-new` so no backup is ever written over; `latest.txt` names the
newest and is published last.

Archives are built deterministically, so unchanged files give the same bytes, and an archive whose SHA-256 equals
its entry in the manifest `latest.txt` names is not uploaded again: the new manifest names the backup folder that
holds its bytes. Each manifest still restores the whole `data/` folder, so a backup folder stays on the server while a
later manifest names it. A backup of an unchanged `data/` uploads only its manifest and `latest.txt`; `--full`
uploads every archive.

An archive over 256 MiB goes up in parts of 256 MiB (`<archive>.000`, `.001`, ...), one publish each, listed in the
manifest with their SHA-256: a publish session ends 15 minutes after it starts (the server's `publish.session_ttl`)
and one upload may not exceed 2 GiB (`max_upload_size`), so a large archive in one publish fails with "Unauthorized:
publishing needs a publish session". Every line telesphorepo prints goes to the backup's log. Run it after every
change to `data/`. `--dry-run` builds the archives and the manifest in `<work>/backup-out/` and logs what it would
upload, without publishing.

`uv run restore` brings the newest backup back (`--backup backups/<UTC time>` for another one). It downloads with
`telesphorepo get`, as the account this computer is logged in as, each archive from the backup folder its manifest
entry names. It joins an archive's parts, checking each, checks every archive and file against the manifest, writes
only missing files, and stops without writing when a file in `data/` differs. Restored files get their recorded
modification times, so every step that was current at the backup is current again. `uv run restore --check`
downloads and verifies every archive the manifest names, wherever its bytes are, and compares it with `data/` without
writing anything.

## Deploying

ECC-Server (github.com/plexusorg/ECC-Server) serves the bundle that `promote` selects. It runs the classifier's graph
in-process on the game host's CPU and uses the bundle's question and cases when it calls the judge.
Module-ECC, the Plex module, asks for a verdict on each line before delivery and carries out the ruling.
ECC-Server's README documents the protocol.
