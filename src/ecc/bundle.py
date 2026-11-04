"""uv run bundle: the classifier bundle the model host (ECC-Server, docs/spec.md section 3) serves, as a folder.

<work>/bundles/<version>/ holds:
- model.gguf, data/classifier.toml's GGUF;
- bundle.json: the version, the Platt parameters, the prompt header and window bounds (chat.PROMPT_FORMAT; the host
  builds lines exactly as chat.clean and chat.prompt_line do), the decision parameters ([redteam] trigger, clearing and
  corroboration with the single-message rule), the slow check's system prompt with the policy (redteam.evidence_system;
  the plugin is open source, so the policy reaches it only through the bundle), the llama.cpp release the host runs,
  and llama-server's deployment arguments (the host adds -t from its own config);
- manifest.json: each file's SHA-256 and size.

The version is the model run's folder name, the GGUF's SHA-256 prefix and a hash of bundle.json's other content, so a
new export, a refit Platt, new thresholds or a policy change is a new version. It needs Platt parameters, so
`uv run eval` must have scored the GGUF's val first.

The folder is then uploaded to W&B ([wandb] in ecc.toml, WANDB_API_KEY in .env) as a version of the bundle artifact,
aliased with the bundle version; a version W&B already holds is not uploaded again. `uv run promote` moves the
production alias, which the model host serves.
"""
import argparse
import hashlib
import json
import os
import shutil

from .backup import sha256
from .chat import PROMPT_FORMAT
from .core import log, run, write_text
from .redteam import evidence_system

# The llama.cpp release every parity check and eval ran (Windows CUDA and AVX2 builds); the host runs its Linux CPU
# build. GitHub's SHA-256 of the asset.
LLAMA_RELEASE = {"release": "b12059", "asset": "llama-b12059-bin-ubuntu-x64.tar.gz",
                 "url": "https://github.com/ggml-org/llama.cpp/releases/download/b12059/llama-b12059-bin-ubuntu-x64.tar.gz",
                 "sha256": "cfb45331bf4ab9af5834dedfed77ec10d531a83de9a2dc05a03d3e131c7cdb12"}
# docs/spec.md section 3: 4 slots of 1,024 tokens, prompt-prefix caching
SERVER_ARGS = ["-np", "4", "-c", "4096", "--slot-prompt-similarity", "0.5", "--cache-ram", "1024",
               "--no-cache-idle-slots"]


def planned(cfg):
    """The bundle the current configuration makes: (version, bundle.json content without the version, GGUF path, its
    SHA-256)."""
    spec = cfg["models"]["classifier"]
    if "platt" not in spec:
        raise SystemExit("data/classifier.toml has no Platt parameters; run `uv run eval` on this GGUF first")
    gguf = cfg.path(spec["gguf"])
    digest = sha256(gguf)
    run_name = gguf.parents[1].name     # <work>/models/<run>/gguf/model.gguf
    rt = cfg["redteam"]
    content = {
        "gguf": "model.gguf",
        "platt": {"a": spec["platt"][0], "b": spec["platt"][1]},
        "prompt": {"header": PROMPT_FORMAT["header"],
                   "window_min": PROMPT_FORMAT["window"][0], "window_max": PROMPT_FORMAT["window"][1],
                   "window_tokens": PROMPT_FORMAT["window"][2], "answers": [" ban", " ok"], "top_logprobs": 20},
        "decision": {"window_rule": "single", "trigger": rt["trigger"], "clearing": rt["clearing"],
                     "corroboration": rt["corroboration"]},
        "slow_check": {"system": evidence_system(cfg)},
        "llama_cpp": LLAMA_RELEASE,
        "llama_server_args": SERVER_ARGS,
    }
    content_hash = hashlib.sha256(json.dumps(content, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return f"{run_name}-{digest[:12]}-{content_hash[:8]}", content, gguf, digest


def wandb_path(cfg, alias):
    w = cfg["wandb"]
    return f"{w['entity']}/{w['project']}/{w['artifact']}:{alias}"


def wandb_api(cfg):
    cfg.load_env()
    if not os.environ.get("WANDB_API_KEY"):
        raise SystemExit("set WANDB_API_KEY in .env (wandb.ai/authorize)")
    os.environ.setdefault("WANDB_SILENT", "true")
    os.environ.setdefault("WANDB_DIR", str(cfg.work))     # W&B's local run files go to <work>/wandb/
    import wandb
    return wandb


def uploaded(cfg, version):
    """Whether W&B holds the bundle version (its artifact version carries the version as an alias)."""
    wandb = wandb_api(cfg)
    try:
        wandb.Api().artifact(wandb_path(cfg, version))
        return True
    except wandb.errors.CommError:
        return False


def upload(cfg, version, out):
    """Logs the bundle folder as a version of the [wandb] artifact, aliased with the bundle version."""
    if uploaded(cfg, version):
        log.info("W&B already holds bundle %s (%s)", version, wandb_path(cfg, version))
        return
    wandb = wandb_api(cfg)
    w = cfg["wandb"]
    bundle = json.loads((out / "bundle.json").read_text(encoding="utf-8"))
    log.info("uploading bundle %s to W&B (%.1f GB)", version,
             sum(f.stat().st_size for f in out.iterdir()) / 1e9)
    run_ = wandb.init(entity=w["entity"], project=w["project"], job_type="bundle", name=f"bundle-{version}",
                      config={"version": version, "decision": bundle["decision"], "platt": bundle["platt"]})
    art = wandb.Artifact(w["artifact"], type="classifier-bundle",
                         metadata={"version": version, "decision": bundle["decision"]})
    for name in ("model.gguf", "bundle.json", "manifest.json"):
        art.add_file(str(out / name), name=name)
    run_.log_artifact(art, aliases=[version]).wait()
    run_.finish()
    log.info("uploaded %s; `uv run promote %s` makes it the bundle the model host serves", wandb_path(cfg, version),
             version)


def body(cfg, args):
    version, content, gguf, digest = planned(cfg)
    rt = cfg["redteam"]
    out = cfg.work / "bundles" / version
    if (out / "manifest.json").exists():
        log.info("bundle %s already exists: %s", version, out)
        if not args.local:
            upload(cfg, version, out)
        return
    out.mkdir(parents=True, exist_ok=True)
    target = out / "model.gguf"
    if not target.exists():
        try:
            os.link(gguf, target)    # same volume: no 3 GB copy
        except OSError:
            shutil.copy2(gguf, target)
    bundle = {"version": version, **content}
    write_text(out / "bundle.json", json.dumps(bundle, indent=1, ensure_ascii=False) + "\n")
    files = {"model.gguf": {"sha256": digest, "bytes": target.stat().st_size},
             "bundle.json": {"sha256": sha256(out / "bundle.json"), "bytes": (out / "bundle.json").stat().st_size}}
    write_text(out / "manifest.json", json.dumps({"version": version, "files": files}, indent=1) + "\n")
    log.info("bundle %s: %s (trigger %s, clearing %s, corroboration %s)", version, out, rt["trigger"], rt["clearing"],
             rt["corroboration"])
    if not args.local:
        upload(cfg, version, out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--local", action="store_true", help="build the bundle folder without uploading it to W&B")
    run("bundle", body, ap)
