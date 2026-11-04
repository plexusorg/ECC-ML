"""uv run promote [VERSION]: make a bundle the one the model host serves, by moving the W&B production alias to it.

VERSION is a bundle version (<work>/bundles/<version>); without it, the bundle the current configuration makes (what
`uv run bundle` built last). The bundle must be uploaded already (`uv run bundle`). ECC-Server checks the alias on a
schedule and at `ecc-server update`, downloads the new bundle, verifies it and swaps it in; moving the alias back to
an earlier version rolls every host back the same way.
"""
import argparse

from .bundle import planned, uploaded, wandb_api, wandb_path
from .core import log, run


def body(cfg, args):
    version = args.version or planned(cfg)[0]
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
    ap.add_argument("version", nargs="?", help="the bundle version (default: the one the current configuration makes)")
    run("promote", body, ap)
