# Working in Module-ECC

- Never work in a session scratchpad or temp folder. It has been wiped mid-task more than once. Code goes in `src/`.
  `data/` is the private folder, backed up whole: put only what cannot be rebuilt there (checks and their results in
  a subfolder of `data/audit/`). What the pipeline rebuilds goes to the work folder (`[paths] work` in
  `data/local.toml`, outside the repository), and experiments go there too and are deleted when they end.
- Delete only what you created. Never remove a folder you did not make, even under `data/audit/`.
- One heavy job at a time on this machine. Check free RAM and the GPU (`nvidia-smi`) before starting training,
  export, eval or a llama-server; dev scoring runs on the GPU, the CPU only for a small sanity sample.
- Commit verified code as you go. `data/` is gitignored and private: never commit or publish real chat, labels or
  weights. After a labelling, merge or case-file change, run `uv run backup`.
- `data/policy.md` is the only rulebook; a rule changes only by the owner's ruling.
- The repository is public (github.com/plexusorg/ECC-ML). Everything that would help someone evade or reproduce the
  filter, and all data, stays in `data/` (backed up to Telesphorepo): the policy, prompts, thresholds, term lists,
  case files, attack and synthetic sets. Code and the design spec are public.
- Notes (handoffs, progress files, logs of what was done) go in `notes/` in the work folder: never in the repository
  or `data/`, never backed up, and deleted when their work is done.
- Do not read `.env`.
- Fix a problem where it starts, and change whatever the fix needs: data formats, labels, the data layout, prompts,
  bundles, interfaces. This project keeps no backwards compatibility, because earlier versions have known problems
  and keeping their behavior keeps the problems. When a fix breaks something downstream, fix that too.
