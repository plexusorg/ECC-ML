# Module-ECC pipeline

This pipeline takes raw Minecraft chat logs and produces the fast-check classifier that ECC-Server runs for
Module-ECC on the game server, plus the measurements that decide whether it ships. Each step is one command.
`uv run status` shows where you are.

```text
raw logs -> ingest -> candidates -> label -> merge -> trainset -> train -> export -> eval -> bundle -> promote
                                                                           review, redteam (slow check vs. Bedrock)
```

## Setup

1. Install [uv](https://docs.astral.sh/uv/), then run `uv sync`.
2. Private data lives in `data/` (gitignored, backed up whole). On a new machine, `uv run restore` fills it from the
   newest backup (see backup), including `data/local.toml`. The chat logs are in `data/raw/<source>/`, one folder per
   `[ingest]` source. What the pipeline rebuilds (models, bundles, eval scores, the train and val tables, logs, tools)
   goes to a work folder outside the repository, never backed up. `data/local.toml` holds the machine's paths:

   ```toml
   [paths]
   work = 'C:\path\to\ECC-ML-work'          # what the pipeline rebuilds
   models = 'C:\path\to\gguf\files'       # base models for the baseline
   llama = 'C:\path\to\llama.cpp\build'    # holds llama-server.exe and llama-quantize.exe
   llama_src = 'C:\path\to\llama.cpp'      # source checkout of the same build, for convert_hf_to_gguf.py

   [eval]
   known_offenders = ["name1", "name2"]    # must trigger on the real log
   ```

3. Copy `.env.example` to `.env` (gitignored) and fill in the credentials you use: the W&B key for `bundle` and
   `promote`, the Bedrock key for `redteam`, the Telesphorepo read token for `restore`.
4. For `label`, install the Codex CLI and log in (`codex`).

Every setting lives in `ecc.toml`. `export` writes the classifier's GGUF and tokenizer to `data/classifier.toml` and
`eval` adds its Platt parameters, the decision thresholds are in `data/decision.toml`, and `data/local.toml` overrides
all of them. Each run writes a full log to `<work>/logs/`.
Ctrl+C stops a step and keeps all finished work. Run the same command again to continue from that point.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/ecc/` | The steps. |
| `docs/` | The system's design (`design.md`: the pipeline, ECC-Server, Module-ECC and their protocol) and the welfare page. |
| `data/` | Private: gitignored, backed up whole to Telesphorepo. Everything the pipeline cannot rebuild. |
| `data/raw/<source>/` | The chat logs of each `[ingest]` source. |
| `data/policy.md` | The rulebook. Labeling and the slow check both read it. |
| `data/labeling/` | Codex's labeling instructions and the term lists that choose which messages are labelled. |
| `data/prompts/` | The slow check's system prompt (`slow-check.txt`, with the policy in place of `{policy}`) and the prompted baseline models' prompts. |
| `data/cases/<set>/` | One folder per eval and red-team set: its case files (`.txt`) and expected roles (`.json`). `attacks/` builds its set in code. |
| `data/synthetic/` | Written training conversations (`[trainset] extra`); `data/briefs/` holds the briefs they were written from. |
| `data/decision.toml` | The deployed decision thresholds. |
| `data/local.toml` | This machine's paths and private settings. |
| `<work>/` | `[paths] work`, outside the repository, never backed up: models, bundles, eval scores, the train and val tables, the review queue, logs, tools, and notes. |

## ingest

Reads every source in `[ingest] sources`: Paper server logs (`YYYY-MM-DD-n.log.gz`, `latest.log`) and Discord
"flagged" exports. Writes `messages.parquet` and `state/ingest.json` in `[ingest] folder` (`--folder` overrides it),
one row per message: id, source, server, time, channel, conversation, speaker, target, text and repeats.

- **Ids.** A message's id is a hash of its server, time, author account, channel, private-message target, raw text and
  its occurrence number among identical messages in its file. Every run reads all files again, so the same files give
  the same ids in any order; a grown `latest.log` keeps the ids of its earlier lines, and a `latest.log` seen again as
  its rotated dated file adds nothing. Stored rows keep their place and new ones are appended. A stored id that the
  files no longer produce stops the run.
- **Dates.** A Paper log's day comes from its file name. `latest.log` takes the day of a full timestamp printed in it
  (Module-Guilds prints them), else the day its rotation implies.
- **Captured:** public chat; private messages (`/msg`, `/tell`, `/w`, `/r`, `/mail send`, partial names resolved
  against the players online, so a pair is one conversation); `/me` (in public chat's conversation); `/nick` (beside
  public chat); guild chat from Module-Guilds' `[GUILD - <guild>:<id>]` console line (one conversation per guild and
  day). From Creayun exports also item names and lore (`item`), books (`book`, title and pages) and signs (`sign`).
- **Speakers are accounts.** Paper chat lines show display names; ingest resolves them to accounts from login lines,
  `/nick` commands, lookalike letters and who is online. A `/sudo <player> c:` line is credited to the player who ran
  it.
- **Dropped:** staff chat (`/o`, `/ac`, `/adminchat`, `[ADMIN]` lines), `/say`, the `/g`, `/gc` and `/guild`
  commands, server output in chat (`* me …` echoes of forced commands, messages that are only target selectors), raid
  accounts, and § colour codes in text. A player's exact repeats in a conversation within 60 seconds are one message
  whose `repeats` counts the copies, as the deployed duplicate rule treats them.

Takes about a minute for 2.2M messages.

## candidates

Reads `messages.parquet`. Finds risk-term hits (`risk` in `data/labeling/terms.toml`) and the rows listed in
`data/audit/rows.jsonl` (one `{"row": n}` per line, such as high-scoring rows of the eval set `outside`), and merges
each with 8 lines before and 4 after into conversation windows. `data/windows.jsonl` only grows: every hit outside all
existing windows gets a new window, so new messages and new risk terms add windows without changing the labelled ones. A
new window may share context lines with an existing one; merge keeps the existing label for those lines. New windows go
into new Codex shards (`data/label/shards/*.txt`, 50 windows each, indexed in `shards.json`), and `uv run label` labels
only those. Lines outside every window are ok without labeling. The step refuses to run when a window's line no longer
matches its row in `messages.parquet`. Takes about a minute.

## label

Runs `codex exec` on every shard (`gpt-6.1-sol`, medium effort, 8 at a time) and writes `data/label/out/<shard>.jsonl`.
Each output is checked against its shard: every window once, rows inside the window, known labels and rules. The step
retries shards that fail and "model is at capacity" answers. A rerun labels only what is missing or invalid.

- `uv run label --check` checks the outputs without running Codex.
- `uv run label --select-regrade 2026-11-02T21:30` builds regrade sets (`data/label/regrade/`) from shards labelled
  before that time: likely misses (lures, pro-pedophile phrasing) and likely over-bans (sexual rules on lines with no
  sexual words). The next `uv run label` labels these into `data/label/regrade_out/`.
- `uv run label --select-unsure` builds a regrade set (`data/label/unsure/`) of every window whose current label is
  marked unsure, shuffled, and the next `uv run label` labels it into `data/label/unsure_out/`. Codex sees the same
  instructions as in the first pass and not the earlier labels.
- `uv run label --limit 8` labels at most 8 of the shards left, for a trial batch.

The full corpus takes a few hours.

## merge

Reads the shard outputs, the regrade outputs and `data/corrections.jsonl`. When sources disagree, a correction beats the
unsure regrade, which beats the regrade sets, which beat the first pass. Writes `data/labels.parquet` (row, label
`ban`/`amb`/`ok`, rule, window) and `data/roles.jsonl`. Hand fixes in `corrections.jsonl` are kept every time the labels
are rebuilt. Takes seconds.

## trainset

Reads `messages.parquet` and `labels.parquet`. Writes `<work>/training/train.parquet` and `<work>/training/val.parquet`.
Each example is a message with up to 29 earlier lines of its conversation, the most a classifier prompt holds. `amb`
lines are dropped. A sample of out-of-window ok lines is added. Validation gets 5% of conversations. The case files in
the `[trainset] extra` folders (written conversations in `data/synthetic/`) go to the training split only. The current
Total Freedom logs and the eval case files never go into training. Takes seconds. Before written conversations go into
an extra file, `uv run python -m ecc.leakage FILE` lists their lines that equal or nearly equal an eval-set line or a
val text, and speakers who are real players (`--drop` removes the leaking lines).

## train

Needs a CUDA GPU. Fine-tunes a LoRA on a base model (a Hugging Face id from the local cache, or a model folder) as a
single-token classifier. The prompt (`classifier_prompt` in `src/ecc/chat.py`) is a fixed instruction line, then one
`speaker: text` line per message ending in a newline, the target message last; the answer is the one token after the
target's newline, ` ban` or ` ok`. Speaker and text are shown as a player sees them (`chat.clean`): NFKC, every
character a player can type that ends a line becomes a space, right-to-left override runs are put in display order, and
invisible format characters are dropped. Base64 in a message is decoded in place when it decodes to plain words
(`chat.decode_base64`). A conversation's prompt only grows by whole lines, so each prompt is a token prefix of the
conversation's next one and llama-server decodes only the new line. The model host builds the prompts and calls
llama-server; it keeps a window per conversation that holds 15 to 29 earlier lines plus the target: it grows until it
would hold 30 earlier lines and is then rebuilt from the last 15, or earlier when the prompt would pass 768 tokens
(`WINDOW_*` in `chat.py`). 768 tokens is also the cap on every prompt in training and scoring. The model host must build
exactly the same text. Training reads whole windows: each conversation is cut into the windows the model host builds,
and one forward pass over a window gives every labelled line in it its loss at the token after its own newline. Causal
attention lets that position see only the line's own prompt, so this is the same as training on each prompt alone, with
37% fewer tokens. `[train] model` picks the model (`1.7b` or `4b`, under `[train.models]`, with each model's token
budget and gradient checkpointing); `--model` picks another. Everything goes to `<work>/models/<model>/`:

- `subset.parquet`, built once from `train.parquet`, `labels.parquet`, `messages.parquet` and the `[trainset] extra`
  files. The labelled lines are every ban row, every row from the extra files, 110,000 ok rows from inside labelled
  windows and 70,000 from outside them; each text is kept at most 5 times per label, and rows matching a case-file
  text are dropped. Their conversations are cut into the model host's windows, each kept up to its last labelled
  line; every other line (amb, unlabelled, over a cap) is context without a loss. A step holds whole windows with
  about 32 labelled lines, and warmup and cosine follow the labelled lines seen. The same inputs give the same windows
  (249,415 labelled lines in 103,296 windows from the current data, 8,608 of them in the 600 synthetic conversations).
- `ckpt/step-*`, every 500 steps, with the val AUROC on 2,000 val rows (with the history the streaming window would
  hold) in the log. The last two are kept. Ctrl+C finishes the current step and saves a checkpoint, and so does
  `--until N` after step N; `uv run train` continues from the newest one after a stop or a crash.
- `adapter/` and `train_done.json` at the end.

Each of these records what it was made from: `subset.parquet.inputs.json` holds the fingerprints of `train.parquet`,
`labels.parquet` and `messages.parquet`, a hash of the case-file texts it drops, the recipe, the base model and the
prompt format; the checkpoints and `train_done.json` hold the subset's fingerprint and the training settings. When any
of that changes, the subset is rebuilt and the run is trained again; `adapter/` and `train_done.json` are replaced when
it finishes. If checkpoints exist from other settings, `train` stops and names the changed settings: delete `ckpt/` to
train from the start, or restore the settings to continue.

On a 32 GB card the 1.7B model takes about 52 minutes and the 4B about 2 hours (about 7,000 steps at 0.40 s and
0.94 s).

To compare base models and recipe changes, `uv run train --steps 2000 [--seed S] [--tag TAG] [--model NAME]` makes
a bounded run: the same subset, the first steps of the epoch that hold 2,000 x 32 labelled lines, and warmup and
cosine over those steps. The seed (default `[train] seed`) sets the batch order, the LoRA initialization and
dropout. Each run has its own folder,
`<work>/models/bounded/<model>[-<tag>]-s<seed>-n<steps>/`. Use a new `--tag` after changing the recipe to keep the
earlier run: a folder whose recipe or inputs changed is trained again from the start.

## export

Turns `<work>/models/<model>/adapter` into the GGUF that eval and the model host use. Each output records what it was
made from (`<output>.inputs.json`: the fingerprints of its input files and llama.cpp tools, its settings and the
prompt format). A stage whose output was made from the current inputs is skipped, so a rerun continues after the last
finished stage; anything else is made again, along with what was made from it:

1. `merged/`: merges the adapter into the base model. The margins on 64 val prompts must stay within
   `merge_tolerance` of the margins recorded at the end of training.
2. `gguf/`: the llama.cpp converter (`[paths] llama_src` in `data/local.toml`, a llama.cpp source checkout) writes
   bf16.
   `llama-imatrix` computes an importance matrix from 1,000 training-split prompts (`imatrix.txt`, never val); it
   reads the file as one token stream in fixed 512-token chunks that run across prompt boundaries. `llama-quantize`
   makes the `[export] deploy` type (Q5_K_M) with it and the token embedding, which is also the output head, at
   Q8_0.
3. `eval/parity-<deploy>.json`: llama-server's margins for the deployed GGUF, with the deployment's server settings,
   against PyTorch's on 300 val prompts.
4. `data/classifier.toml`: `[models.classifier] gguf` (the deployed GGUF) and the tokenizer of the base model the run
   was trained on. It overrides `ecc.toml`, and `data/local.toml` overrides it. `uv run eval` then scores that GGUF and
   adds the Platt parameters it fits on the GGUF's own val margins. Its P(ban) is calibrated on val's class balance
   (about 5% ban), not on live chat, so it is a fixed scale for thresholds, which come from evaluation.

Development scores on the GPU: `[eval] server` points at a CUDA build of the same llama.cpp release. llama.cpp's
margins depend on the backend (CPU builds for different instruction sets differ by about 0.1-0.2 on average), so a
small sample is checked once on a CPU build that matches the deployment CPU, such as llama.cpp's AVX2-only build for
a Ryzen 5 3600X.

The 4B model's export took 33 minutes on the RTX 5090 machine, most of it the importance matrix (25 minutes with a CPU
build of llama.cpp on 16 threads of a Ryzen 9 9950X3D); quantizing takes seconds.

For a bounded run, `uv run export` with the run's train flags scores the adapter on the base model with PyTorch,
without merging, on val, the case sets and the real log, summarized as `eval` does (val also on its rows whose text
never appears in train), with a Platt fit on val (`platt-stream.json`). It writes the scores, `summary.md` and
`metrics.json` to `eval/torch-stream/` in the run folder and rebuilds
`<work>/models/bounded/table.md`, one row per bounded run. The table's tf column is
recomputed from each run's tf scores with the current `known_offenders`: the players other than those offenders who
trigger at the highest threshold that still triggers all of them. That threshold is picked on tf itself, so the count
is in-sample. The step does not touch `data/classifier.toml`.

## eval

`uv run eval` scores a model on every set: the case-set folders in `[eval.sets]`, `attacks` (what a player controls
in the classifier's input, built by `data/cases/attacks/`), `val`, and `tf` (every message of the current Total
Freedom logs, which are never used for training). The classifier sees each message with the history
its conversation's window would hold in the model host: val and tf messages replay their conversation from
`messages.parquet`, case messages their case-file conversation. The classifier is scored by the llama.cpp library of
the `[eval] server` build (`src/ecc/llama.py`, in its own process), one forward pass per conversation window: while a
conversation's prompts each extend the previous one, every message's margin is read at the end of its own line,
where the logits are those llama-server gives that message's prompt alone, with the same top-20 rule for an answer
token llama-server would not list. Each window gets a 1,024-token sequence, a deployment slot, and `[eval] slots`
windows share a decode. The margin is read from the exact ` ban` and ` ok` token ids. A prompted model is served by
a llama-server that eval starts and stops. Scores go to
`<work>/eval/<model>/<set>.jsonl` one message at a time, so a stopped or killed run continues where it stopped.
`<set>.jsonl.inputs.json` records what they come from (the case files, or the ingest batches of `messages.parquet`
and for val `val.parquet`; the GGUF, the llama.cpp build and its settings, the prompt format and tokenizer); when any
of it changes, the set is scored again from the start. A newly ingested batch only adds messages, so the next run
scores just those. `summary.md` is rebuilt from every set whose messages all have current scores (a partly scored
set is listed with its unscored count): AUROC and PR-AUC, recall at val false-positive rates on
all of val and on val rows whose text never appears in train, players caught by each window rule, and, on the real
log, the players triggered per day, the known offenders caught, and the labelled offenders caught at the operating cut
(the margin at val FPR 1%). The classifier's thresholds apply to its Platt
P(ban), as in the model host.

`failures.md`, rebuilt with the summary, shows where the model fails:
- on each case set, the innocent players that set the zero-innocent bar, every offender under it, and the bans missed
  and other messages flagged at the operating cut;
- on val, the highest-scoring ok rows and the lowest-scoring bans;
- on the real log, the labelled offenders the cut misses and the other players it triggers.

Each message is shown with the last lines the model saw, their labels (with the labelling rule) and scores, and its
own P(ban) in context and alone. A message that is high only in context was flagged for its conversation; one that is
high alone was flagged for its own words.

`--model baseline` scores the prompted Qwen3-4B for comparison. `--set` picks sets;
`--set train` scores the labelled training rows of 2,000 random windows (`[review] sample_windows`) for `review`.
`--set outside` scores every message outside every window, of every source; they count as ok only because no risk
term opened a window, so the summary lists how many reach each threshold, by source. A
full run takes about 10 minutes on the RTX 5090.

## review

Reads a model's scores on labelled training rows (`uv run eval --model <model> --set train` makes them) and finds the
labelled lines the model strongly disagrees with (confident learning). Only training rows are queued, so relabelling
never moves val or tf toward the model. They go to `<work>/review/queue.txt`. To fix a label, write `ban`, `ok` or `amb`
after `fix:` and run `uv run review` again. The fix moves to `data/corrections.jsonl`. `uv run merge` applies it.

## redteam

Runs the slow check as ECC-Server runs it on the `[redteam.sets]` case-set folders: Kimi K3 on Bedrock with
datamarked transcripts and code-verified quotes, roles and clearing, and the fine-tuned classifier
(`[redteam] fast_model`, its case-set scores from `eval`) as the fast layer. The thresholds are the deployed ones in
`data/decision.toml` (`--trigger`, `--clearing` and `--corroboration` try others). A player triggers at their first
message with P(ban) at or above the trigger; a conversation where nobody triggers gets no Kimi call. A triggered
player is cleared only with no verified quote and every earlier message below `clearing`. A player who never
triggered, in a conversation where someone did, is banned only with corroboration: a verified quote tagged starting,
directed, pushing or grooming, and at least one of their quoted messages at or above `corroboration`. Every message
of the sets being run needs a current `eval` score. Writes
`data/redteam/<tag>.report.md`, counting offenders cleared, innocents banned and so on. Every Kimi answer is cached in
`<tag>.calls.jsonl`, so a rerun pays only for missing calls. `--cached` never contacts Bedrock.

## bundle

`uv run bundle` writes the classifier bundle the model host serves to `<work>/bundles/<run>-<GGUF SHA-256
prefix>-<content hash>/`: `model.gguf`, `bundle.json` (Platt parameters, prompt format and window bounds, the trigger,
clearing and corroboration thresholds from `data/decision.toml`, the slow check's system prompt, the pinned llama.cpp
release and llama-server's deployment arguments) and `manifest.json` with each file's SHA-256. It needs the Platt
parameters `eval` fits. It then uploads the folder to the private W&B project in `[wandb]` as a version of the bundle
artifact, aliased with the bundle version (`--local` skips the upload).

## promote

`uv run promote [version]` moves the W&B `production` alias to an uploaded bundle (default: the one `bundle` builds
from the current configuration). That is the whole deployment: every ECC-Server picks the new bundle up at its next
check or at `ecc-server update`, verifies it and swaps it in. Promoting an earlier version rolls back the same way.

## backup

`uv run backup` copies the whole `data/` folder to the private `ecc-data` raw repository on Telesphorepo
(`telesphorepo.toml`): one gzipped tar per folder in it, one for the files directly in it, and `manifest.json` with
each file's SHA-256, size and modification time. The work folder and `.env` are never included. The repository keeps
one file per path and a publish replaces it, so each backup goes to its own `backups/<UTC time>/` folder, uploaded
with `--if-new` so no backup is ever written over; `latest.txt` names the newest. Run it after every change to
`data/`. `--dry-run` builds the archives in `<work>/backup-out/` without publishing.

`uv run restore` brings the newest backup back (`--backup backups/<UTC time>` for another one). It needs a read token
of `ecc-data` as `TELESPHOREPO_USER` and `TELESPHOREPO_READ_TOKEN` in `.env`. It checks every bundle and file
against the manifest, writes only missing files, and stops without writing when a file in `data/` differs. Restored
files get their recorded modification times, so every step that was current at the backup is current again.
`uv run restore --check` downloads and verifies a backup and compares it with `data/` without writing anything.

## Deploying

ECC-Server (github.com/plexusorg/ECC-Server) serves the bundle `promote` made production, beside `llama-server` on the
game host's CPU, and makes every chat-safety decision: the trigger, the slow check on Bedrock and the verdicts.
Module-ECC, the Plex module, sends it each message and carries out its mutes and bans; `docs/design.md` describes the
whole system and its protocol. ECC-Server builds each conversation's growing prompt exactly as `chat.py` does and
sends it to llama-server with `cache_prompt`, so the server's prefix cache needs the bundle's llama-server settings
(`-np 4 -c 4096 --slot-prompt-similarity 0.5 --cache-ram 1024 --no-cache-idle-slots`), the settings `export`'s
parity check runs.
