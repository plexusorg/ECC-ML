# Working in ECC-ML

ECC-ML is the training pipeline for the chat-safety classifier that ECC-Server runs for Module-ECC. The repository is
public (github.com/plexusorg/ECC-ML); its data, policy and model are private.

## Where things go

- `data/` is the private folder: everything that cannot be rebuilt, and everything private. That is the labelled data
  (`label/`, the merged labels, roles, windows and corrections, `label-history/`), the raw chat logs, the message table
  the labels point into, the case and synthetic sets, the policy, the prompts, the term lists, the decision thresholds
  and `data/local.toml`. It is gitignored and backed up whole by `uv run backup`; run the backup after every change to
  it.
- The work folder (`[paths] work` in `data/local.toml`, outside the repository) is the scratch space: everything the
  pipeline rebuilds (models, bundles, eval scores, the train and val tables, the review queue, logs, tools),
  experiments, and notes in `notes/`. Nothing in it is backed up. Delete an experiment when it ends, and a note when
  its work is done.
- A model ships through W&B: `uv run bundle` uploads it, `uv run promote` makes it the one ECC-Server serves. A model on
  disk is a working copy.
- Code goes in `src/`. Configuration names folders, not lists of files: a case set is a folder, and the code reads
  what is in it.
- Never work in a session scratchpad or temp folder; it has been wiped mid-task more than once.
- Anything that would help someone evade or reproduce the filter stays in `data/`: the policy, prompts, thresholds,
  term lists, case, attack and synthetic sets, and measured model gaps. Code and the README are public.
- Specs and working documents are not kept in the repository. Plans, progress and open decisions go in the handoff
  (`notes/HANDOFF.md` in the work folder); what the system does is documented by the README and the code. `docs/` is
  gitignored and holds only the public web pages (the welfare and safety pages).

## The labelled data

The labels are the most valuable thing the project has; everything else can be rebuilt from them and the raw logs.
Write a relabel as new outputs beside the old ones, never over them, and keep earlier merged versions in
`data/label-history/`. Ask the owner before deleting anything in `data/` you did not create.

## Working rules

- `data/policy.md` is the only rulebook; a rule changes only by the owner's ruling.
- Recall comes first: tune thresholds, rules, labels and ship gates so the classifier catches every offender. The slow
  check (Kimi on Bedrock) clears the false positives.
- Fix a problem where it starts, and change whatever the fix needs: data formats, labels, the data layout, prompts,
  bundles, interfaces. The project keeps no backwards compatibility, because earlier versions have known problems and
  keeping their behavior keeps the problems. When a fix breaks something downstream, fix that too.
- One heavy job at a time on this machine. Check free RAM and the GPU (`nvidia-smi`) before training, export, eval or
  a llama-server; development scoring runs on the GPU, the CPU only for a small sanity sample.
- Commit verified code on master as you go.
- Do not read `.env`.
