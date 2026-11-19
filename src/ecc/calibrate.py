"""The hold thresholds of a C export slot, set on the owner's validation weeks only.

Named hold cuts on C's hold logit:
- `1pct`, `2pct`: the cuts that hold that share of the covered validation lines.
- `offenders`: the highest cut at which every validation sitting with a hold-positive line has a line held in time:
  a positive line, or its ban line, at or before the sitting's first ban line (`ban_here`); a sitting with no ban
  line, any of its positive lines. A line rejected before scoring is blocked, so a sitting with such a line in time
  needs no cut. Lowered by the allowed CPU logit error.
- `recall`, the served default: the lower of `2pct` and `offenders`. It holds every validation offender in time and,
  while `offenders` is not under `2pct`, at most 2% of validation lines. `ceiling_ok` tells whether it is; bundle
  refuses a slot where it is not, because a C that must hold more chat than that to catch every offender is not
  shipped.

Kind cuts are hints and never clear a hold.
"""
import numpy as np

from . import inputs as I

C_TOLERANCE = 1e-3      # the allowed |CPU - torch| logit error
SHARES = {"1pct": 0.01, "2pct": 0.02}
CEILING = "2pct"        # the most validation traffic the served default may hold


def kth(scores, k):
    """The k-th highest score: holding score >= it takes k lines (more on a tie)."""
    if not 1 <= k <= len(scores):
        raise ValueError(f"the {k}-th highest of {len(scores)} scores")
    return float(np.sort(np.asarray(scores, np.float64))[::-1][k - 1])


def offender_cut(z, covered, hold, sitting, ban_here):
    """(the offenders cut, the sittings it was set on), over validation rows in arrival order."""
    sittings = 0
    cut = np.inf
    for s in np.unique(sitting[hold == 1]):
        rows = np.flatnonzero(sitting == s)
        bans = rows[ban_here[rows]]
        due = rows[rows <= bans[0]] if len(bans) else rows
        eligible = due[(hold[due] == 1) | ban_here[due]]
        sittings += 1
        if (~covered[eligible]).any():
            continue
        if not len(eligible):
            raise SystemExit(f"validation sitting {s} has no positive or ban line to hold before its ban line")
        cut = min(cut, float(z[eligible].max()))
    if not np.isfinite(cut):
        raise SystemExit("the offenders cut needs a validation sitting whose positive lines C scores")
    return cut - C_TOLERANCE, sittings


def calibrate(t, meta, judged, labels):
    """Calibrate actual scored keys against their complete source table and hold labels.

    t and meta are the validation and test keys in arrival order; labels holds dossier, id, hold, sitting and
    ban_here. Return numerical evidence only. Release code adds and verifies model and immutable-file provenance.
    """
    key = ["dossier", "id"]
    if t.duplicated(key).any() or not t[key + ["period"]].equals(meta[key + ["period"]]):
        raise SystemExit("calibration scores do not contain the exact validation/test keys and periods")
    if not t.coverage.isin(["c", "reject"]).all():
        raise SystemExit("calibration requires explicit C-only scores")
    expected = judged[judged.split.isin(["val", "test"])][key + ["split"]].rename(columns={"split": "period"})
    if not meta[key + ["period"]].reset_index(drop=True).equals(expected.reset_index(drop=True)):
        raise SystemExit("calibration requires every judged validation/test source key and its exact period")
    if judged.duplicated(key).any() or labels.duplicated(key).any():
        raise SystemExit("calibration source has duplicate label keys")
    if set(map(tuple, judged[key].to_numpy())) != set(map(tuple, labels[key].to_numpy())):
        raise SystemExit("hold labels must match every judged table key exactly")
    if not labels.hold.isin([-1, 0, 1]).all():
        raise SystemExit("calibration needs explicit known or masked hold labels")
    lab = t[key].merge(judged[key + I.KINDS], on=key, how="left", validate="1:1")
    lab = lab.merge(labels, on=key, how="left", validate="1:1")
    if lab[I.KINDS + ["hold", "sitting", "ban_here"]].isna().any().any():
        raise SystemExit("calibration scores have missing labels")
    val = t.period.to_numpy() == "val"
    covered = t.coverage.to_numpy() == "c"
    calibration = val & covered
    positive = calibration & (lab.hold.to_numpy() == 1)
    z = t.c_hold.to_numpy(np.float64)
    if not calibration.any() or not positive.any():
        raise SystemExit("recall calibration needs covered validation targets and known hold-positive evidence")
    if not np.isfinite(t.loc[covered, ["c_" + o for o in I.OUTS]].to_numpy()).all():
        raise SystemExit("covered C logits are not finite")
    hold = {k: kth(z[calibration], max(1, int(np.ceil(share * calibration.sum())))) for k, share in SHARES.items()}
    hold["offenders"], sittings = offender_cut(z[val], covered[val], lab.hold.to_numpy()[val],
                                               lab.sitting.to_numpy()[val], lab.ban_here.to_numpy(bool)[val])
    hold["recall"] = min(hold[CEILING], hold["offenders"])
    marks = {k: int((lab[k].to_numpy()[calibration] == 1).sum()) for k in I.KINDS}
    kinds = {k: min(kth(t["c_" + k].to_numpy()[calibration], count), 0.0) if count else 0.0
             for k, count in marks.items()}
    traffic = {}
    for period in ("val", "test"):
        rows = t.period.to_numpy() == period
        total = int(rows.sum())
        traffic[period] = {name: {"total": total, "covered": int((rows & covered).sum()),
            "rejected": int((rows & ~covered).sum()),
            "model_held": int((rows & covered & (z >= threshold)).sum()),
            "blocked": int((rows & ((~covered) | (z >= threshold))).sum())}
            for name, threshold in hold.items()}
    return {"scoring": {"mode": "c_only"}, "default": "recall",
            **{"hold_" + k: v for k, v in hold.items()}, "val_lines": int(calibration.sum()),
            "guard_band": C_TOLERANCE, "ceiling": CEILING, "ceiling_ok": hold["offenders"] >= hold[CEILING],
            "validation_hold_positive": int(positive.sum()), "validation_offender_sittings": sittings,
            "validation_hold_labels": {str(k): int((calibration & (lab.hold.to_numpy() == k)).sum())
                                       for k in (-1, 0, 1)}, "traffic": traffic,
            "set_on": "Validation only. Recall is the lower of the 2pct cut and the highest cut that holds a line of "
                      "every validation hold sitting at or before its ban line, less the allowed CPU parity error. "
                      "A slot whose offenders cut is under the 2pct cut does not ship.",
            "kinds": kinds, "kind_marks": marks,
            "kinds_set_on": "Validation kind quantiles capped at zero; diagnostic hints only, never a clearance gate."}
