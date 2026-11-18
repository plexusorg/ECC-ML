"""Qualify the frozen C release against configured owner-positive targets."""
import json
import re

import numpy as np

from . import inputs as I
from .backup import sha256
from .core import log, write_text
from .export import lines_path, links, load_lines, model_inputs, ov_scores, slot_thresholds


def require_known_offenders(cfg, f):
    """Require every bound owner-positive target to pass the calibrated recall point on CPU.

    Account identity selects existing labels. It never supplies a positive label. Owner sitting rulings do not
    assert that a human selected each positive line or its exact timing.
    """
    import pandas as pd

    from .trainset import hold_path

    report_path = f["out"] / "known-offenders.json"
    report = {"passed": False, "batch": cfg["export"]["batch"], "targets": []}

    def save():
        write_text(report_path, json.dumps(report, indent=1, ensure_ascii=False, allow_nan=False) + "\n")

    def refuse(reason):
        report["failure"] = reason
        save()
        raise SystemExit(f"known-offender release gate failed; see {report_path}")

    required = cfg.raw.get("eval", {}).get("known_offenders")
    if not isinstance(required, list) or not required or any(
            not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_]{1,16}", name) for name in required):
        refuse("[eval] known_offenders must be a nonempty list of Minecraft account names")
    required = [name.lower() for name in required]
    if len(set(required)) != len(required):
        refuse("[eval] known_offenders contains duplicate account identities")
    report["required_accounts"] = required
    calibrated = slot_thresholds(cfg, f)
    threshold = calibrated["hold_recall"]
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not np.isfinite(threshold):
        refuse("the calibrated recall threshold is not finite")
    report["hold_recall"] = threshold
    report["made_from"] = calibrated["made_from"]
    report["release_files"] = {
        str(path): {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in (f["out"] / "thresholds.json", f["models"] / "graphs.json",
                     f["models"] / "c.xml", f["models"] / "c.bin")}

    df, _ = load_lines(cfg)
    keys = ["dossier", "id"]
    table = pd.read_parquet(lines_path(cfg), columns=keys + ["account", "channel", "judged", "split"])
    labels = pd.read_parquet(hold_path(cfg, cfg["export"]["batch"]),
                             columns=keys + ["hold", "sitting_source"])
    if labels.duplicated(keys).any() or table.duplicated(keys).any():
        refuse("the selected table or hold labels contain duplicate target keys")
    targets = table.merge(labels, on=keys, how="left", validate="one_to_one")
    targets["account"] = targets.account.fillna("").str.lower()
    targets = targets[targets.account.isin(required) & targets.judged & (targets.channel != "discord") &
                      (targets.sitting_source == "owner") & (targets.hold == 1)].reset_index(drop=True)
    targets = targets.rename(columns={"split": "period"})
    report["targets"] = json.loads(targets[keys + ["account", "period"]].to_json(orient="records"))
    report["missing_accounts"] = sorted(set(required) - set(targets.account))
    if report["missing_accounts"]:
        refuse("a required account has no judged owner-positive target in the selected export batch")

    # load_lines retains whole validation/test runs. A required training target can be in another run.
    selected = set(map(tuple, targets[keys].to_numpy()))
    if not selected <= set(map(tuple, df[keys].to_numpy())):
        import pyarrow.parquet as pq
        from .trainset import with_raw
        path = lines_path(cfg)
        cols = ["dossier", "run", "id", "time", "channel", "partner", "speaker", "account", "text", "judged", "split"]
        names = pq.ParquetFile(path).schema_arrow.names
        df = with_raw(cfg, pd.read_parquet(path, columns=cols + (["raw_text"] if "raw_text" in names else ["row"])))
        allow, canonical = links(cfg)
        df["text"] = [I.model_text(t, allow, canonical) if isinstance(t, str) else t for t in df.text]
    positions = df[keys].copy()
    positions["row"] = np.arange(len(df))
    targets = targets.merge(positions, on=keys, validate="one_to_one")
    report["targets"] = json.loads(targets[keys + ["account", "period", "row"]].to_json(orient="records"))

    rows = targets.row.to_numpy()
    _, _, routes = model_inputs(cfg, df, rows)
    rejected = routes[rows] != "c"
    for target, route in zip(report["targets"], routes[rows]):
        target["coverage"] = str(route)
    report["rejected"] = int(rejected.sum())
    if rejected.any():
        refuse("a required positive target was rejected before model scoring")

    scores = ov_scores(cfg, f, df, targets, np.arange(len(targets)))
    finite = np.isfinite(scores).all(axis=1)
    held = finite & (scores[:, I.HOLD] >= threshold)
    for target, z, valid, detected in zip(report["targets"], scores, finite, held):
        target["finite"] = bool(valid)
        target["hold_logit"] = float(z[I.HOLD]) if np.isfinite(z[I.HOLD]) else None
        target["held"] = bool(detected)
    report["nonfinite"] = int((~finite).sum())
    report["missed"] = int((finite & ~held).sum())
    if not held.all():
        refuse("a required positive has nonfinite logits or misses the fixed calibrated recall point")
    report["passed"] = True
    save()
    log.info("known-offender release gate: %d owner-positive targets, %d accounts passed on CPU; %s",
             len(targets), len(required), report_path)
    return report
