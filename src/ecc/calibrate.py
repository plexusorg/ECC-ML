"""The hold thresholds of an export slot, set on the owner's validation weeks only.

Named hold cuts on the classifier's hold logit:
- `1pct`, `2pct`: the cuts that hold that share of the covered validation lines.
- `offenders`: the highest cut at which every validation session with a hold-positive line has a line held in time:
  a positive line, or its ban line, at or before the session's first ban line (`ban_here`); a session with no ban
  line, any of its positive lines. A line rejected before scoring is blocked, so a session with such a line in time
  needs no cut. Lowered by the allowed CPU logit error.
- `recall`, the served default: the lower of `2pct` and `offenders`. It holds every validation offender in time and,
  while `offenders` is not under `2pct`, at most 2% of validation lines. `ceiling_ok` tells whether it is; bundle
  refuses a slot where it is not, because a classifier that must hold more chat than that to catch every offender is not
  shipped.

Category cuts are hints and never clear a hold.
"""
import numpy as np

from . import inputs as I

CPU_TOLERANCE = 1e-3      # the allowed |CPU - torch| logit error
SHARES = {"1pct": 0.01, "2pct": 0.02}
CEILING = "2pct"        # the most validation traffic the served default may hold


def kth(scores, k):
    """The k-th highest score: holding score >= it takes k lines (more on a tie)."""
    if not 1 <= k <= len(scores):
        raise ValueError(f"the {k}-th highest of {len(scores)} scores")
    return float(np.sort(np.asarray(scores, np.float64))[::-1][k - 1])


def offender_cut(z, covered, hold, session, ban_here):
    """(the offenders cut, the sessions it was set on), over validation rows in arrival order."""
    sessions = 0
    cut = np.inf
    for s in np.unique(session[hold == 1]):
        rows = np.flatnonzero(session == s)
        bans = rows[ban_here[rows]]
        due = rows[rows <= bans[0]] if len(bans) else rows
        eligible = due[(hold[due] == 1) | ban_here[due]]
        sessions += 1
        if (~covered[eligible]).any():
            continue
        if not len(eligible):
            raise SystemExit(f"validation session {s} has no positive or ban line to hold before its ban line")
        cut = min(cut, float(z[eligible].max()))
    if not np.isfinite(cut):
        raise SystemExit("the offenders cut needs a validation session whose positive lines the classifier scores")
    return cut - CPU_TOLERANCE, sessions


def calibrate(t, meta, judged, labels):
    """Calibrate actual scored keys against their complete source table and hold labels.

    t and meta are the validation and test keys in arrival order; labels holds transcript, id, hold, session and
    ban_here. Return numerical evidence only. Release code adds and verifies model and immutable-file provenance.
    """
    key = ["transcript", "id"]
    if t.duplicated(key).any() or not t[key + ["period"]].equals(meta[key + ["period"]]):
        raise SystemExit("calibration scores do not contain the exact validation/test keys and periods")
    if not t.coverage.isin(["scored", "rejected"]).all():
        raise SystemExit("calibration requires every key's coverage, scored or rejected")
    expected = judged[judged.split.isin(["val", "test"])][key + ["split"]].rename(columns={"split": "period"})
    if not meta[key + ["period"]].reset_index(drop=True).equals(expected.reset_index(drop=True)):
        raise SystemExit("calibration requires every judged validation/test source key and its exact period")
    if judged.duplicated(key).any() or labels.duplicated(key).any():
        raise SystemExit("calibration source has duplicate label keys")
    if set(map(tuple, judged[key].to_numpy())) != set(map(tuple, labels[key].to_numpy())):
        raise SystemExit("hold labels must match every judged table key exactly")
    if not labels.hold.isin([-1, 0, 1]).all():
        raise SystemExit("calibration needs explicit known or masked hold labels")
    lab = t[key].merge(judged[key + I.CATEGORIES], on=key, how="left", validate="1:1")
    lab = lab.merge(labels, on=key, how="left", validate="1:1")
    if lab[I.CATEGORIES + ["hold", "session", "ban_here"]].isna().any().any():
        raise SystemExit("calibration scores have missing labels")
    val = t.period.to_numpy() == "val"
    covered = t.coverage.to_numpy() == "scored"
    calibration = val & covered
    positive = calibration & (lab.hold.to_numpy() == 1)
    z = t.classifier_hold.to_numpy(np.float64)
    if not calibration.any() or not positive.any():
        raise SystemExit("recall calibration needs covered validation targets and known hold-positive evidence")
    if not np.isfinite(t.loc[covered, ["classifier_" + o for o in I.OUTS]].to_numpy()).all():
        raise SystemExit("covered classifier logits are not finite")
    hold = {k: kth(z[calibration], max(1, int(np.ceil(share * calibration.sum())))) for k, share in SHARES.items()}
    hold["offenders"], sessions = offender_cut(z[val], covered[val], lab.hold.to_numpy()[val],
                                               lab.session.to_numpy()[val], lab.ban_here.to_numpy(bool)[val])
    hold["recall"] = min(hold[CEILING], hold["offenders"])
    marks = {k: int((lab[k].to_numpy()[calibration] == 1).sum()) for k in I.CATEGORIES}
    categories = {k: min(kth(t["classifier_" + k].to_numpy()[calibration], count), 0.0) if count else 0.0
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
    return {"default": "recall",
            **{"hold_" + k: v for k, v in hold.items()}, "val_lines": int(calibration.sum()),
            "guard_band": CPU_TOLERANCE, "ceiling": CEILING, "ceiling_ok": hold["offenders"] >= hold[CEILING],
            "validation_hold_positive": int(positive.sum()), "validation_offender_sessions": sessions,
            "validation_hold_labels": {str(k): int((calibration & (lab.hold.to_numpy() == k)).sum())
                                       for k in (-1, 0, 1)}, "traffic": traffic,
            "set_on": "Validation only. Recall is the lower of the 2pct cut and the highest cut that holds a line of "
                      "every validation hold session at or before its ban line, less the allowed CPU parity error. "
                      "A slot whose offenders cut is under the 2pct cut does not ship.",
            "categories": categories, "category_marks": marks,
            "categories_set_on": "Validation category quantiles capped at zero; diagnostic hints only, never a "
                                 "clearance gate."}
