# Module-ECC pipeline

This pipeline takes raw Minecraft chat logs and produces the fast-check classifier that ECC-Server runs for
Module-ECC on the game server, plus the measurements that decide whether it ships. Each step is one command.
`uv run status` shows where you are.

```text
raw logs -> dossier -> read -> truth -> labels       (the labels)
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

## Repository layout

| Path | Contents |
| --- | --- |
| `src/ecc/` | The steps. |
| `data/` | Private: gitignored, backed up whole to Telesphorepo. Everything the pipeline cannot rebuild. |
| `data/raw/` | The chat logs: the Total Freedom server's Paper logs (`tf/`, `tf-old/`) and Creayun's Discord exports (`creayun/`). |
| `data/policy.md` | The rulebook. The slow check reads it. |
| `data/v4/` | The v4 reader batches (`batches/`: parts, reads, truth tables), the readers' questions, and what the v4 bundle is built from besides the models: its thresholds, settings and the judge's texts (`data/v4/README.md`). |
| `data/models/base/<repo>@<sha>/` | The base models of C and of SH's line encoder at their pinned commits (base models, below). |
| `data/v4/models/C-s2/` | The shipped C, which ranks the training lines (`[train] ranker`); its `input.json` holds its base revision and `max_tokens`. |
| `data/rulings/` | The owner's saved rulings, their reviewed derivatives and the scripts that built the ruling pages. Select the judge's rulings with `[bundle] rulings`. |
| `data/local.toml` | This machine's paths and private settings. |
| `<work>/` | `[paths] work`, outside the repository, never backed up: models, bundles, eval scores, the train and val tables, logs, tools, and notes. |

## dossier

`uv run dossier BATCH` builds the parts of a reader batch from the raw chat and checks them. A batch is a folder under
`[dossier] batches`: `manifest.json` (its sources and what was built from them), `parts/`, `reads/<read>/` and
`truth.parquet`. `--files GLOB...` starts a Paper batch from log files under `[dossier] logs` (`tf/2026-11-1*.log.gz`);
`--creayun` starts a Creayun batch of every server and day of `[dossier] messages`
(`data/v4/creayun-messages.parquet`): Creayun's chat and private messages read from its Discord exports, one row per
message with its time to the minute, each numbered by `row`.

- **A dossier** is one Paper log file (one uptime or one day), or one Creayun server on one day, one line per event:
  `<id> <time> <channel> <speaker>: <text>`. `L<n>` is line n of the log file (for Creayun, the message whose `row` is
  n); `R<n>` is line n of that day's ECC-Server record (`[dossier] record`, v3 or v4), from which
  signs, books, item names and lines typed while muted come (a sign or item name set by command, which a v4 record
  also holds, is read from the log). Ids stay the same across rebuilds.
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
  online player is ever the answer). A log file that continues a run starts from the player state at the end of the file before (who is
  online, nicknames, message partners, pending `/sudo`, the accounts' UUIDs), so a player online since before midnight
  keeps their account; a file that starts a run starts with nobody known. A chat name ends at the first ` » ` unless
  it shows no player and a name ending at a later one does (a tag ending in `»`). A chat line the day's record holds
  takes the record's account. Module-Guilds' guild chat is read from its `[GUILD - <guild>:<id>]` console line on
  any thread; a `[Guild]` tag before a chat name is part of the shown name of a public chat line.
- **Commands.** EssentialsX's `/msg`, `/r`, `/mail` and `/me` are read under every alias its plugin.yml gives them, and
  a typed player name resolves as EssentialsX's `matchUser` resolves it (UUIDs, name prefixes, nicknames). A command
  EssentialsX sends to no one (a `/msg` to an offline or unknown player, a `/r` with no one to answer) makes no line,
  and neither does a `me`, `say`, `msg`, `reply` or `mail` command NUSH cancelled for a quarantined player (its feed
  line `[NUSH] <account>: <command>` follows it, or the feed showed one of that player's lines in the 10 seconds
  before). `/mail sendall` is one line to `*`.
- **Addresses.** None is written. A join shows `from addr-NN`, an alias numbered by first appearance within the
  dossier; IPv4 and IPv6 addresses inside any shown field become `[ip]`.
- **Authentication.** Every login, register, password or `/shib` command (typos and plugin prefixes included) is
  dropped with its arguments, as is any line whose text is such a command, and any line in which its speaker repeats a
  word they typed to one of those commands in the same log (for Creayun, in any export: the word is replaced by
  `[secret]`, and the line is dropped if that misses it). For Creayun, e-mail addresses, keys and ids in links, tokens,
  passwords stated in a sentence, card numbers and key-like strings are replaced by placeholders. Only counts are kept.
- **Left out:** staff chat, console output, moderation notices. A speaker's exact repeats on one channel within 60
  seconds fold into one line, `[sent N times until T]`; a log file that continues a run (`dossier.runs`: a file
  without the startup line continues the uptime of the file before it) folds copies of the file before's lines too.
- **Parts** hold at most 12,000 tokens (characters / 3), cut at the longest quiet gap in the last 40% of each part's
  room; a later part starts with the 30 lines before its cut and a `--- part N begins here ---` marker.

On a batch that has parts, the step rebuilds them, cut where the saved parts were cut, and compares them byte for
byte with the saved ones. It writes only when the batch has no parts or no read has answered them; otherwise a part
that differs is reported and the saved parts stay, with the count of parts `--rewrite` would read again.
`--rewrite NAME` writes them anyway: the saved parts and manifest, and every read's answers to a part whose body gains
a line or shows one of its lines with another speaker or addressee, go to `[dossier] history/NAME/<batch>/` with
`rewrite.json` (the parts rewritten, the parts to read again with the lines they gain and the lines that show another
speaker, the lines no longer built); `uv run read` then answers the parts whose answers moved. A part whose body
changes only in text keeps its answers. Every run ends with the check, and `--check` runs only the check: no IPv4 or IPv6 address, no
authentication command as a line's text, no authentication word in its typist's own lines, and for Creayun none of the
secret shapes. It prints line ids and lengths, never text. The Paper batches take seconds, Creayun about a minute.

## read

`uv run read BATCH READ` has the Codex CLI read every part of a batch that has no answer from that read yet, each part
in its own ephemeral session (read-only sandbox, an empty working folder in `<work>/read/`, no user config or rules),
`--parallel N` at a time up to the reader's `max_parallel`. A read is named `<reader>-<copy>` (`codex4-a`, `codex4-b`):
every copy of a reader sends the question its `[read.readers.<reader>]` entry names in `[read] questions`, then the
part. A "marks" reader answers the seven kinds on any line of the part; a "details" reader (`codex5`) is asked, per
part, only the lines and kinds the batch's `truth.parquet` marks. Each answer is checked (JSON only, ids of the part,
known kinds and details) and saved to `reads/<read>/<part>.json`; every attempt goes to `reads/<read>/log.jsonl` and an
answer that fails the check to `reads/<read>/invalid/`. A part that fails the dossier check is never sent. After
`[read] stop_after` usage-limit answers in a row the run starts nothing new; run it again later to continue.

- `--disputed` reads only the parts the batch's `disputed.json` lists: the third read (`codex4-c`).
- `--parts a,b`, `--limit N` narrow the run; `--out DIR` saves the answers elsewhere.
- `--dry-run` writes each prompt beside its working folder and shows the command, without calling Codex.

## truth

`uv run truth BATCH...` builds each batch's line table from its reads: `truth.parquet` and `disputed.json`.
Select the table versions with `[truth] tables`. Keep the reader parts and manifests
under `[dossier] batches`. To change labels, use `--out NEW_FOLDER` for all selected batches, verify the new tables,
then select that folder. The command keeps an identical existing table and refuses to replace a changed one.

Each row represents one player line with the seven kinds, the account, the sitting (`<dossier>|<account>`), up to 5 earlier lines of its
conversation within 5 minutes, and the split. A Paper row also has its run (one server uptime over its log files;
rows are in run order) with `time` counting from the midnight of the run's first day, and what the plugin sends for
the line: `raw_text`, `raw_shown` and the `copies` folded into it. Per kind, from reads a and b (`[truth] reads`): 1 when both marked the
line, -1 when one did, 0 when neither; on a part the third read (`[truth] third`) answered, two of three decide. A line
a read listed as unsure and no read marked is -1 on every kind; off_server is 0 on a line whose links are all on
`[truth] allowlist` after meaning normalization. The original reader value remains in `off_server_raw`.
A line holding a text of `[truth] left_out` is -1 on every kind. `disputed.json` lists the parts,
and their lines, a third read should cover.

A Paper batch needs both reads on every part. Its dossiers are rebuilt from the logs for each line's account, raw
text and fields, and the build stops unless every rebuilt part is byte for byte the part the reads read
(`uv run dossier BATCH --rewrite NAME` writes the rebuilt parts). A Creayun batch takes the parts both reads have answered, rebuilt from the message table and compared byte
for byte; the account is the shown name, and its rows are in dossier order, each dossier's parts by part number. The
split is the one period rule, `[periods]`: a log file of the `owner` batch (`full`) is `train` before `val_from`,
`val` before `test_from` and `test` from then on, by its date; every line of the other Paper batches is `train`; a
Creayun dossier is `test_creayun` when a hash of its name falls in `[truth] test_share` (15%), else `train`.
`--out DIR` writes the tables to `DIR/<batch>/`. All four batches take about three minutes.

## labels

`uv run labels [BATCH...]` makes the hold labels: from the truth tables of the `[labels] batches`, the cases a judge
reads and the sittings the owner ruled, then from them and the judge's reads each batch's
`data/v4/batches/<batch>/hold.parquet` (dossier, id, sitting, sitting_ruling, sitting_source, hold, ban_here). It
writes `<work>/labels/cases.json`, `<work>/labels/owner.parquet` and `<work>/labels/sittings.parquet` (every sitting's
ruling and `by` line). The `hold.parquet` it replaces is first copied to
`data/label-history/hold-<the run's time>/<batch>/`; run `uv run backup` after.

- **The owner's sittings.** `[labels] rulings` lists his rulings files in order; the first that rules a sitting gives
  its ruling (block and warn are hold, no ban is not ban) and its `by` (when it names none, the next file with the same
  ruling that names one). An entry lands on the sitting of each of its case's own lines, the player's lines of the
  case as the owner ruled it (the ruling page, or the rendered case the entry's `cases` names); a case whose lines are
  now on two sittings rules both, and the step lists such cases. An entry with no case lands on the judged lines whose
  speaker has the name in its key. A file with `take = "trusted"` gives only its trusted sittings, one with
  `take = "unsure"` only its unsure ones; an unsure entry of any other file leaves the sitting to the next file. The
  step stops when a rulings file is missing or an entry lands on no judged line.
- **The cases.** One per sitting with a judged line a kind marks (1), except the owner's own, those whose judged lines
  are all spam raids (`left_out`) and those any rulings file rules. A case shows each marked or disputed line of the
  player with the lines around it, as `data/v4/bundle.toml [judge.render]` sets for the live judge (a private message
  within its own conversation), the player's lines `>>`, tagged by `[labels] tags`: the readers' kinds and disputed
  kinds with whom the line is aimed at, or no tags. A case over the render's `max_chars` is cut to its marked lines,
  then to a narrower window, then to as many marked lines, spread evenly, as fit. The owner's ruled cases dated after
  `[labels] examples_until` are added as `[labels] cases` renders them, with id `owner|<case id>`; they measure the
  judge and label nothing.
- **The examples** are the owner's ruled cases `uv run bundle` packs, dated up to `[labels] examples_until`. The step
  stops when one is dated on or after `[periods] test_from` or its lines are on a sitting that is a case.
- **The reads.** `[labels] answers` holds a folder per read of `[labels] reads`, each answer (ruling, by, sure) keyed by
  the SHA-256 of the case text the judge read. Answers are added as new files and never changed; a later answer to the
  same text stands. The step stops, naming them, while a case lacks an answer in any read.
- **The sitting rulings.** The owner's own sittings are `own`; a sitting he ruled takes his ruling (`trusted`,
  `unsure`, `nothing`, `hold`, `ban`, `not ban`); a sitting whose judged lines are all spam raids is `spam raid`; any
  other the reads' ruling where they agree, else `split`. A ban's `by` is a valid explicit owner line, or a line all
  selected reads name exactly, with every answer sure and `ban`. The line must be a judged line of the sitting.
  An inferred line has `by_from = "judge reads"`, including when the owner ruled ban without a line. For those
  owner bans, select `[labels] reads` for his rendered cases that are cases here (after `examples_until`), and
  `[labels] by_reads` for the others (his examples). The step stops while an answer is missing. A separate
  `censor_after` marks possible intervention: use the valid explicit owner line, else the earliest valid line from
  those reads, else the first line marked contact, sexual or romantic, else the first marked line. If an explicit
  owner line is invalid, use the marked-line fallback for censoring. The censor does not establish a ban deadline.
  `censor_from` records its source. For each judge-derived boundary, retain the case text hashes in `by_hashes` or
  `censor_hashes` so the frozen reads can be checked.
- **The line labels**, one per judged line: `nothing` and clean sittings hold 0; `hold` and `ban` sittings 1 on lines
  with a kind at 1, -1 on lines whose only non-zero kinds are -1, else 0; a ban sitting -1 on every line after its
  `censor_after`. Mark only a supported `by` line as `ban_here`; every other sitting -1 on lines with a non-zero kind,
  else 0; spam-raid lines (`left_out`) -1. A -1 line is out of the hold loss.

The three batches take under a minute.

## base models

C starts from `Roblox/roblox-pii-classifier-v2` (`[train.c]`) and SH's line encoder from
`intfloat/multilingual-e5-small` (`[train.sh]`), each pinned to a commit sha. `uv run python -m ecc.base fetch`
downloads the pinned files of a base model that has no copy into `data/models/base/<repo>@<sha>/`, through a cache of
its own in the work folder, so the Hugging Face cache is not written; it checks each file against the hub's size and
SHA-256 and writes `source.json` (repo, revision, each file's size and SHA-256). Training reads the copy, never the hub
(`ecc.base.path`), and refuses a copy of another repo or revision or with a file of another size.
`uv run python -m ecc.base check` hashes every copy against its `source.json` and checks that the ranker's
`input.json` names the pinned C base.

## trainset

`uv run trainset` builds the inputs C and SH train on with `src/ecc/inputs.py`, the code export and the bundle build
the served inputs with, into `<[train] out>/inputs/` (`<work>/train/inputs/`), about 10 minutes of CPU.

- **The Paper table.** Every row of the `[train] owner` batches (full, old, old2) in one table in arrival order (by
  run, its log files in log order, each file's rows as stored), so a run whose log files went to two batches is one
  stream, as ECC-Server receives it. The step stops when a batch's stored order is not its runs' arrival order (time
  running backwards, a log file out of order or in two blocks), when a merged run's time runs backwards, and when two
  log files of one name (from two logs folders) would share a dossier or a run. A kinds-only batch
  (`[train] kinds_only`, Creayun) is its own table in its truth order. The model text uses the same meaning
  normalization as serving, then redacts revealed addresses and replaces allowed links with the canonical one.
  Source records, repeat keys and judge text keep their prepared form.
- **What it writes.** `paper.parquet` (the Paper table's rows: batch, dossier, id, run, coverage); `lines.parquet`, the lines
  the models train and are scored on in training row order (the judged Paper lines in Paper order, then every
  kinds-only line), with batch, run, the row in the Paper table, split, coverage, the seven kinds and the masks (`kmask`: out of
  the kind loss); SH's line ids of every Paper row (`sh-tokens.npy`) and SH's features of every judged Paper line
  (`sh-features.npz`, context lines as Paper rows); C's input ids of every line of `lines.parquet` at each length of
  `[train.c] tokens` and at the length the ranker (`[train] ranker`) was trained at (`c-<N>.npy`). A ragged array is
  one flat `.npy` and its `.offsets.npy`: row k is `ids[offsets[k]:offsets[k + 1]]`.
- The tokenizers are the pinned base models' copies (base models, above). `inputs.json` records what the inputs were
  made from, the source code hash, the input manifests, the lengths, the primary `coverage_tokens` and the counts.
  The primary length is the first `[train.c] tokens` entry. Each row has a coverage route: `cascade`, `c` (bypass the
  SH gate), or `reject`. Rejected rows retain their keys, train no loss, and never enter later context. The source
  raw text determines whether preparation would shorten a target; Creayun raw text comes from its stable source row.
  The step reuses current inputs.
  Use `--force` to build them again. Ranking, exported graphs and torch scores also record their builder's source hash.
- **The ranking.** `uv run trainset --rank` also scores the candidates for hard negatives with the ranker (`[train]
  ranker`, a pinned trained C) at the length it was trained at, bf16 on the GPU: the train
  lines with every kind at 0 and not kind-masked, outside `[periods] owner` (old, old2 and Creayun's train split).
  The score is the larger of the hold logit and the largest kind logit; `<work>/train/rank.parquet` holds each
  candidate's row of `lines.parquet`, its logits and score. Rejected rows are not candidates. The ranking serves the
  selected coverage length and its teacher folds.
- **The training lines.** Once every Paper batch has its `hold.parquet` (labels), `uv run trainset` writes
  `<work>/train/pick.parquet` (row of `lines.parquet`, batch, dossier, id, reason, hold, score) in training row order:
  every train line of `[periods] owner` (`all`); of the other Paper batches every hold = 1 line (`hold`), every
  unmasked line with a kind at 1 or -1 (`non-zero`), the top `[train] share` (10%) of each batch's candidates by score
  (`hard`) and a random share of its other candidates (`random`); of each kinds-only batch the same with `[train]
  kinds_share` (5%), every line at hold -1 (out of the hold loss). A line out of both losses (kind-masked, hold -1) is
  dropped. The random draws are `default_rng([train] seed)` over each batch's other candidates in row order, one stream
  for the Paper batches in `[train] owner` order and one for each kinds-only batch. The step stops on a training line
  outside the train split, a candidate without a score, a judged line without a hold label, or a `rank.parquet` made
  from other inputs or another ranker.

## train

`uv run train c --tokens N` trains C on the trainset's inputs cut to N tokens, the length it will serve at (N must be
the input set's recorded `coverage_tokens`), into `<[train] out>/c-<N>/`. Use the GPU for training. To select another
serving length, change the first `[train.c] tokens` entry and rebuild the inputs.

- **The model.** `Roblox/roblox-pii-classifier-v2` (the pinned copy, base models above) with a fresh head of eight
  sigmoid outputs, the seven kinds and hold. A kind at -1 (one read marked it, or a read was unsure) contributes no
  loss for that output. A line with `kmask` is out of the kind loss; a line at hold -1 is out of the hold loss.
- **The lines.** Those `<[train] out>/pick.parquet` names: the owner's lines with their hold label from each `[train]
  owner` batch's `hold.parquet`, which must label every judged line in the inputs, then the kinds-only lines (Creayun)
  at hold -1. The step refuses a pick outside the train split and inputs that are not current.
- **The recipe** (`[train.c]`): AdamW, a linear warm-up then a linear decay to 0, gradients clipped, bf16 autocast
  over fp32 weights, batches of length-sorted chunks, a fixed number of epochs and one seed. The run records GPU use
  and warns about slow step windows. Non-finite losses or gradients stop training before another optimizer update.
- **What it writes.** After the last epoch C scores the owner's validation and test lines and Creayun's test_creayun
  lines in fp32, as export does: `scores.parquet` and `creayun.parquet`, which `uv run eval` reads. Beside them the
  model (`config.json`, `model.safetensors`, `tokenizer.json`), `train.json` (each epoch's loss, time and memory, the
  GPU readings, the validation hold metric) and `input.json`: the run's description, `max_tokens` (which export and the
  bundle cut C's input to), the base revision, the settings and what the run was made from.
  C saves the model and `trained.json` before scoring. If scoring fails, run the same command again to finish from
  the saved model. The run checks the data, settings, source code and each saved model file's size and SHA-256.
  A changed or incomplete saved model is refused. Use `--force` only to replace that run and train again.
  After each epoch, `epoch-<N>/` holds the model, optimizer, scheduler and random state, with file hashes in
  `complete.json`. Run the same command to resume an interrupted training run from its last verified epoch.
  The trainer preserves incomplete checkpoint files beside the completed epochs.
- `--fold K` trains teacher fold K of `[train.c] folds` into `c-<N>-fold<K>/`: the Paper runs are dealt into folds with
  `[train.c] fold_seed` (a run split between two batches is in one fold), and fold K's model trains on the owner's
  selected training lines of the other folds and the selected kinds-only lines. It then scores, as SH's teacher,
  only cascade-route train-split Paper lines of fold K (bf16, as the ranking scores) into `teacher.parquet`, and writes
  the Paper runs it trained on to `trained-runs.json`; it refuses before training when a line it would score is of a
  run it trains on. A teacher fold does not score validation or test lines. `--epochs E` and `--limit N` make a smoke
  run (`-smoke`). `--limit` selects up to N training lines, divided between Paper and kinds-only rows when available,
  and scores the first 2,000 lines of each scored set. `--force` trains again over a
  finished run.

`uv run train sh --c <run>` trains SH, the student, into `<[train] out>/sh-<run>/`, after `<run>` (`c-<N>`) and its
teacher folds (`c-<N>-fold<K>`). GPU, about 2 minutes an epoch.

- **The model.** `models.py`'s SH (the module export loads) over the pinned e5-small copy, `[train.sh] layers` context
  layers, the head's bias at each output's log-odds among the training lines.
- **The lines.** Cascade-route train-split judged Paper lines, with the trainset's SH line ids and features, kinds and
  hold label. Each unknown kind is out of its output's loss, `kmask` is out of the kind loss, and hold -1 is out of
  the hold loss. Accepted forced-C rows can remain in context; SH does not learn their unused gate decisions.
- **The teacher.** Each line's eight logits from the fold model of `<run>` that did not train on its run
  (`teacher.parquet` of each fold). The step refuses a fold trained on other inputs, labels, lines or length than
  `<run>`, a fold with different settings or coverage, a fold whose scored lines share a run with `trained-runs.json`,
  and a cascade training line scored by no fold or by
  two. `--teacher h` takes H instead (the models in `[train.sh] h`, at the length they were trained at, bf16, their mean;
  cached in `<[train] out>/teacher-h.parquet`) into `sh-h/`, to check the folds' SH against an H-taught one.
- **The recipe** (`[train.sh]`): the soft binary cross-entropy of SH's logits against the teacher's at `temperature`
  (times its square) on all eight outputs, plus the hard one on the labels, divided by the number of terms; batches of
  chunks of consecutive lines of one log file; AdamW with three learning rates (e5-small, the new layers, the head),
  warm-up then linear decay over `epochs`; bf16 autocast.
- **The epoch.** After each epoch SH scores covered validation lines in fp32. The epoch rule uses only cascade
  validation lines: at the configured pass share, SH must pass as many as possible of the lines C holds at the
  configured hold share. Forced-C rows already bypass that gate. Training stops `patience` epochs after the best.
- **What it writes.** SH's model folder (`model.safetensors`, `meta.json`, e5-small's `config.json` and
  `tokenizer.json`; `[export] sh` can name it), `scores.parquet` (dossier, id, period, `sh_<out>` of the kept epoch on
  every validation and test line, fp32, as export scores), `train.json` (each epoch's loss, time, memory and
  validation measure) and `input.json` (what the run is, which bundle.json `models.sh` carries, its teacher, epoch,
  settings and what it was made from).
  Every epoch has a verified model, optimizer, scheduler and random-state checkpoint. Run the same command to resume
  interrupted training. After training, the job copies the selected epoch's model and scores every validation and test
  key. Rejected rows have zero diagnostic logits and are identified by their coverage field; no model runs on them.
  `--limit N` (N training lines drawn with seed 0) and `--max-steps N` (N steps an epoch, two epochs) make a smoke run
  (`-smoke`). An explicitly limited C supplies its first 2,000 validation and test keys to a smoke SH's epoch rule.
  A full C must supply every validation and test key. Smoke models cannot be exported or bundled.

## export

Run `uv run export` to convert the completed C classifier into one OpenVINO graph. The export needs no SH model or
SH tokenizer. It preserves the trained C input: raw preparation, model meaning, allowed links, complete-target
checks, causal context, role names and token budget. Score every covered target with C. Reject an incomplete target
before folding, context or judgment.

Each C run has a slot at `<[export] out>/<C folder name>/`:

1. `models/c.xml`, `c.bin` and `graphs.json`: fp32 weights, dynamic batch and sequence dimensions, and graph metadata.
   The conversion checks real Torch outputs on several sequence lengths.
2. `torch.parquet`: all eight C logits for every covered judged validation and test key. Rejected keys remain in the
   table with absent logits. The selected immutable table supplies the full causal runs.
3. `thresholds.json`: diagnostic `1pct` and `2pct` thresholds and a named `recall` threshold. Calibration uses validation
   only. Recall is the minimum known covered hold-positive validation logit minus the allowed CPU logit error.
   Kind thresholds are hints. They do not clear a hold. The file records exact model, table and hold-label sources.
   It reports model-held, input-rejected and total traffic counts separately.
4. `parity.json`: the actual CPU graph outputs against Torch on identical inputs, for all eight outputs and all named
   thresholds. Run `--full` for all validation and test keys. Use `--sample N` to change the sample size or `--files N`
   to select the first N files. Use `--no-parity` to skip this check. The default sample includes every covered target
   near a threshold and a reproducible random sample.

Export and bundle refuse smoke runs, teacher folds, research models, missing hold-positive training evidence and
changed model bytes. Completed training records must match the model files and training source. The saved C inputs
must match the served token budget. Input, score and threshold records detect changed source data or input rules.
Keep numerical proof fixtures separate from release artifacts.

## eval

`uv run eval RUN [RUN ...]` measures runs on the owner's validation and test weeks and on Creayun's held-out lines,
and writes `<[eval] out>/<run>[+<run>...]/tables.md` and `eval.json`. A run is a folder of scores, named under
`[export] out` or `[train] out`, or given as a path (a name found under both is refused):

Each owner period must include scores for every judged line in its truth table. Evaluation refuses incomplete scores,
including smoke runs. Export and bundle also refuse smoke models, teacher folds and unverified training records.

- a production run, the C export slot: `torch.parquet` and `thresholds.json`. The score and threshold provenance must
  match. Each covered target uses its C hold logit at diagnostic or recall thresholds. Rejected targets are blocked
  without a model decision. The report names the C folder.
- a C run, a folder of `train c`: `scores.parquet` (dossier, id, period and C's eight logits, `c_<out>`, of every
  judged validation and test line) and, when it scores them, `creayun.parquet` (the same of Creayun's test_creayun
  lines). C alone holds a line when its hold logit reaches the score that holds each share of `[eval] holds` of the
  run's own validation lines.

The report shows each run after the first beside its difference from the first, paired on the same resamples.
List the production run first to compare with it. Use `--labels` to select a label set. By default, measure each
set whose files exist. The `forward` set uses the frozen labels and ban timing from `[eval] forward`. The `new` set
uses the hold labels in each `[eval] batches` folder and derives supported ban timing from the configured rulings
and saved reads. Evaluation checks that the resolved keys, hold labels and sitting fields match `hold.parquet`.
The batches' `truth.parquet` supplies the splits, kinds, kind masks and line order.

Every run must score the complete judged key set for the requested periods. Evaluation refuses partial score tables,
even when their filenames match a full run. A `coverage` field distinguishes model scores from rejected targets.

The report separates input rejections, model holds and total blocks. It gives model recall on all positive lines and
on covered positive lines, plus total blocked recall. A target blocked before scoring is never counted as a model
detection. The other measures, per period and threshold: blocked recall of the hold = 1 lines (all, in owner-ruled sittings, by kind), the
share of lines held, hold = 0 lines held per 1,000, the marked hold = 0 lines held, clean sittings of
20 or more lines with a line (and with 5 lines) held, the owner's ban, hold, nothing and not-ban sittings with a line
held, and his ban sittings whose ban line, or a line by it, is held. On Creayun's held-out lines, for runs that score
them: each kind's lines over the hold thresholds and over the kind's own output at the score that flags `[eval]
kind_share` of the owner's covered validation lines, and the unmasked all-zero lines over each. Creayun tables show
total blocks and model holds separately; each kind's own-output table uses covered lines only. Beside them, the hold labels of
the training period's lines with each kind at 1. Intervals are 95% over `[eval] resamples` resamples of the log files
(Creayun: of its dossiers), seeded with `[eval] seed`, thresholds fixed.

`--validation` reports the validation weeks only: every choice between runs (such as C's length) is made there, and
the test weeks are reported once, after it. A run takes seconds, on the CPU.

## bundle

Run `uv run bundle --no-upload` to build and check the C-only bundle in `<work>/bundles/<version>/`. The build reads
calibrated thresholds from the C export slot. It keeps the owner's private threshold file and old README template
unchanged. It preserves the private judge settings and texts.

Before creating or reusing a bundle, the build checks the required accounts in private `[eval] known_offenders`.
Each account must have judged owner-positive hold targets in the selected export batch.
Every such target must be covered and pass the real C CPU graph at the fixed calibrated recall threshold.
Missing bindings, rejected inputs, nonfinite logits or missed positives block the build and its upload workflow.
Read the private result in the export slot's `known-offenders.json`.

- `models/`: only `c.xml`, `c.bin` and `c_tokenizer.json`.
- `bundle.json`: `scoring.mode=c_only`, graph and tokenizer hashes, the eight ordered outputs, kind hints, exact
  preparation and input rules, their hash, validation thresholds and provenance. Use `recall` by default.
- `judge/question.md` and `judge/examples.json`: the current private judge texts and owner rulings.
- `README.md`: the C-only artifact instructions. `manifest.json` records every other file's SHA-256 and size.

The version includes a hash of the content. The build copies files into unique staging, checks their bytes and
preserves independent owned copies. The reference loader checks the explicit contract, input-rule hash, output
order and model-file hashes. Old cascade bundles fail the C-only contract.

The sibling `<version>-vectors/` folder contains real private wire records, folded copies, preparation examples and
expected outputs. The checks compare exact C token ids and contexts with the training builder. They compare the real
CPU graph outputs with Torch and check all named threshold decisions. Run the independent scorer with
`uv run python -m ecc.reference BUNDLE IN.jsonl OUT.jsonl`.

Without `--no-upload`, the command uploads the verified bundle to the private W&B artifact project. A separate
promotion selects the served version.

## replay-export

`uv run replay-export` writes the test weeks as `ecc-server replay` reads them, into `[replay] out` (chat text, in the
work folder only), to measure a bundle's models and judge together before it ships:

- `lines.jsonl`: every run of `[export] batch` that holds a test file, as one stream each: the join and leave lines of
  its log files as `presence` events, and every line as the plugin sends it, prepared and raw, each copy the dossier
  folded into a line sent as a line of its own (`copy_of`). A line's player is its resolved account, so a player who
  shows several names is one player. A judged line of a test file carries its sitting, the verdict and kind marks of
  the torch scores in the export slot of `[export] c` at the thresholds `export` set there (`hold` when C reaches the validation recall threshold); the owner's own sittings (`[replay] owner`) are trusted; every
  other line is context only. The step stops when a judged test line has no torch score, or when the slot's
  thresholds were set from another C model or different score inputs.
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

ECC-Server (github.com/plexusorg/ECC-Server) serves the bundle that `promote` selects. It runs one C graph
in-process on the game host's CPU and uses the bundle's question and cases when it calls the judge.
Module-ECC, the Plex module, asks for a verdict on each line before delivery and carries out the ruling.
ECC-Server's README documents the protocol.
