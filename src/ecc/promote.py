"""uv run promote [VERSION]: make a bundle the one the model host serves, by moving the W&B production alias to it.

VERSION is a bundle version (<work>/bundles/<version>); without it, the bundle `uv run bundle` built last. The bundle
must be uploaded already (`uv run bundle`). ECC-Server checks the alias on a schedule and at `ecc-server update`,
downloads the new bundle, verifies it and swaps it in; moving the alias back to an earlier version rolls every host
back the same way.
"""
import argparse

from .bundle import uploaded, wandb_api, wandb_path
from .core import log, run


def last_built(cfg):
    """The version of the newest bundle folder in <work>/bundles."""
    built = sorted((p.stat().st_mtime, p.parent.name) for p in (cfg.work / "bundles").glob("v4-*/manifest.json")
                   if not p.parent.name.endswith(".tmp"))
    if not built:
        raise SystemExit(f"no bundle in {cfg.work / 'bundles'}; run `uv run bundle`")
    return built[-1][1]


def body(cfg, args):
    version = args.version or last_built(cfg)
    if not uploaded(cfg, version):
        raise SystemExit(f"W&B has no bundle {version}; run `uv run bundle` first")
    wandb = wandb_api(cfg)
    api = wandb.Api()
    art = api.artifact(wandb_path(cfg, version))
    if "production" in art.aliases:
        log.info("bundle %s is already production", version)
        return
    try:
        previous = api.artifact(wandb_path(cfg, "production"))
        before = next((a for a in previous.aliases if a not in ("production", "latest") and not a.startswith("v")),
                      previous.version)
    except wandb.errors.CommError:
        before = None
    art.aliases.append("production")
    art.save()
    now = api.artifact(wandb_path(cfg, "production"))
    if version not in now.aliases:
        raise SystemExit(f"W&B production is {now.aliases} after the move, not {version}")
    log.info("production is now bundle %s%s; model hosts pick it up at their next check or `ecc-server update`",
             version, f" (was {before})" if before else "")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("version", nargs="?", help="the bundle version (default: the one `uv run bundle` built last)")
    run("promote", body, ap)
