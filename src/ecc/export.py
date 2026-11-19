"""Export the completed classifier as one OpenVINO graph and check it against Torch.

The slot is <[export] out>/<the classifier's folder name>. It holds models/classifier.xml and classifier.bin,
torch.parquet, the validation-only thresholds.json and parity.json. Reject incomplete targets before context. Score
every covered target with the classifier. Preserve raw preparation, model meaning, canonical links and the trained
token budget. The named thresholds and the served default are set as src/ecc/calibrate.py states. Keep model-held,
input-rejected and total traffic counts separate. Never choose thresholds from test outcomes. Refuse smoke, research
or changed weights.
"""
import os

os.environ.setdefault("ONEDNN_MAX_CPU_ISA", "AVX2")    # the host's instruction set; set before OpenVINO loads
os.environ.setdefault("RAYON_RS_NUM_CPUS", "6")        # the tokenizer's threads
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
import argparse  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
from importlib.metadata import version as package_version  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from . import inputs as I  # noqa: E402
from .calibrate import CPU_TOLERANCE, SHARES, calibrate  # noqa: E402
from .core import fingerprint, is_current, log, progress, record_inputs, run, write_text  # noqa: E402

GRAPHS = ("classifier",)
# How ECC-Server compiles each graph (bundle.json `openvino.compile`).
COMPILE = {"INFERENCE_PRECISION_HINT": "f32", "INFERENCE_NUM_THREADS": 6, "PERFORMANCE_HINT": "LATENCY",
           "NUM_STREAMS": 1}
NEAR = 0.01             # sampled lines: every line whose torch score is this close to a threshold
SEED = 20261114


def folders(cfg):
    """The classifier's folder and its export slot."""
    e = cfg["export"]
    model = cfg.path(e["classifier"])
    out = cfg.path(e["out"]) / model.name
    return {"classifier": model, "out": out, "models": out / "models"}


def classifier_run(folder):
    """A training run's input.json: `run` (what the run is) and `max_tokens` (the length the classifier serves: its
    input is cut from the left to that many tokens). Every input export and the bundle build is cut at it."""
    p = Path(folder) / "input.json"
    if not p.exists():
        raise SystemExit(f"{folder}: no input.json, which records the run's max_tokens")
    run_ = json.loads(p.read_text(encoding="utf-8"))
    n = run_.get("max_tokens")
    if not isinstance(n, int) or isinstance(n, bool) or n < 4:
        raise SystemExit(f"{p}: max_tokens {n!r} is not a length of at least 4 tokens")
    if not isinstance(run_.get("run"), str) or not run_["run"]:
        raise SystemExit(f"{p}: `run`, what the run is, is missing")
    if run_.get("smoke") or run_.get("limit"):
        raise SystemExit(f"{folder}: refuse to export a smoke run")
    if run_.get("coverage_tokens") != n:
        raise SystemExit(f"{folder}: coverage_tokens must equal the served max_tokens {n}; rebuild the inputs and retrain")
    return run_


def pair(f):
    """Verify the completed classifier, training source and owned weight bytes before release."""
    from .base import sha256
    from .core import source_hash
    c = classifier_run(f["classifier"])
    mark = f["classifier"] / "trained.json"
    if not mark.is_file() or not (f["classifier"] / "train.json").is_file():
        raise SystemExit("a release needs completed training metadata, not a research fixture")
    trained = json.loads(mark.read_text(encoding="utf-8"))
    req = trained.get("request", {})
    if req.get("smoke") or req.get("limit"):
        raise SystemExit("refuse a smoke classifier")
    if req.get("code") != source_hash("train", "models", "inputs", "core"):
        raise SystemExit("the classifier's training source differs from the recorded source; preserve the trained source")
    if req.get("made_from") != c.get("made_from") or req.get("tokens") != c["max_tokens"]:
        raise SystemExit("the classifier's input metadata does not match its verified training request")
    if c.get("outputs") != I.OUTS or c.get("counts", {}).get("train_hold1", 0) <= 0:
        raise SystemExit("a release needs the eight ordered outputs and real hold-positive training evidence")
    for name in ("config.json", "model.safetensors", "tokenizer.json"):
        path = f["classifier"] / name
        if not path.is_file() or trained.get("files", {}).get(name) != {"size": path.stat().st_size,
                                                                       "sha256": sha256(path)}:
            raise SystemExit(f"training file is missing or changed: {path}")


def model_marks(f):
    """The classifier's folder and its fingerprint, recorded with calibrated thresholds."""
    return {"classifier": {"folder": f["classifier"].as_posix(), "fingerprint": fingerprint(f["classifier"])}}


def slot_thresholds(cfg, f):
    """Refuse thresholds unless their model, score file, table and hold labels are current."""
    path = f["out"] / "thresholds.json"
    if not path.exists():
        raise SystemExit(f"no {path}; run `uv run export`")
    t = json.loads(path.read_text(encoding="utf-8"))
    if "hold_recall" not in t:
        raise SystemExit(f"{path}: thresholds require a calibrated recall threshold; run `uv run export`")
    now = model_marks(f)
    was = t.get("models", {})
    for k in ("classifier",):
        if was.get(k, {}).get("fingerprint") != now[k]["fingerprint"]:
            set_from = was.get(k, {}).get("folder", "models it does not name")
            raise SystemExit(f"{path} was set from {set_from} {was.get(k, {}).get('fingerprint')}, not "
                             f"from [export] {k} {now[k]['folder']} {now[k]['fingerprint']}; run `uv run export`")
    provenance = t.get("made_from", {})
    score = f["out"] / "torch.parquet"
    if provenance.get("torch") != fingerprint(score):
        raise SystemExit(f"{path}: calibrated score file changed; run `uv run export`")
    if not is_current(score, torch_made_from(cfg, f)):
        raise SystemExit(f"{path}: score inputs changed; run `uv run export`")
    current = calibration_sources(cfg, f)
    for key, name in (("table", "selected table"), ("hold", "selected hold labels")):
        if provenance.get(key) != current[key]:
            raise SystemExit(f"{path}: {name} changed since calibration; run `uv run export`")
    if provenance != thresholds_made_from(cfg, f):
        raise SystemExit(f"{path}: calibration provenance changed; run `uv run export`")
    return t


def thresholds(cfg, f):
    t = slot_thresholds(cfg, f)
    return {k: t["hold_" + k] for k in ("1pct", "2pct", "recall")}


def versions():
    return {k: package_version(k) for k in ("openvino", "torch", "transformers", "tokenizers")}


# ---------------------------------------------------------------- 1. graphs

def name_io(g, ins, outs):
    for p, n in zip(g.inputs, ins):
        p.get_tensor().set_names({n})
    for p, n in zip(g.outputs, outs):
        p.get_tensor().set_names({n})


def describe(path):
    import openvino as ov
    g = ov.Core().read_model(path)
    xml = Path(path).read_text(encoding="utf-8")
    ops = sorted(set(re.findall(r'<layer [^>]*version="(opset\d+)"', xml)), key=lambda s: int(s[5:]))

    def io(ps):
        return [{"name": p.get_any_name(), "shape": str(p.get_partial_shape()),
                 "dtype": str(p.get_element_type().get_type_name())} for p in ps]

    return {"ir_version": int(re.search(r'<net name="[^"]*" version="(\d+)"', xml).group(1)), "opsets": ops,
            "inputs": io(g.inputs), "outputs": io(g.outputs)}


def check_graph(tag, compiled, feeds, ref):
    d = max(float(np.abs(compiled(f)[0] - r).max()) for f, r in zip(feeds, ref))
    log.info("%s: OpenVINO against torch on %d made-up inputs, max |diff| %.2e", tag, len(feeds), d)
    if not np.isfinite(d) or d > CPU_TOLERANCE:
        raise SystemExit(f"{tag}: the graph differs from the torch model by {d:.3g} on made-up inputs")


def graphs_made_from(f):
    from .core import source_hash
    return {"classifier": fingerprint(f["classifier"]), **versions(),
            "code": source_hash("core", "inputs", "models", "export")}


def convert_graph(model_folder, out, max_tokens):
    """Convert the classifier and compare numerical samples. This function does not certify release provenance."""
    import openvino as ov
    import torch
    from .models import classifier_logits, load_classifier
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(6)
    core = ov.Core()
    rng = np.random.default_rng(0)
    dyn2 = ov.PartialShape([-1, -1])
    w = classifier_logits(load_classifier(model_folder))
    ex = torch.from_numpy(np.concatenate([[[0]] * 2, rng.integers(5, 200000, (2, 62)), [[2]] * 2], 1).astype(np.int64))
    with torch.no_grad():
        g = ov.convert_model(w, example_input=(ex, torch.ones_like(ex)), input=[dyn2, dyn2])
    name_io(g, ["input_ids", "attention_mask"], ["logits"])
    ov.save_model(g, out / "classifier.xml", compress_to_fp16=False)
    del g
    cm = core.compile_model(out / "classifier.xml", "CPU", COMPILE)
    feeds = []
    for n in (5, 100, max_tokens):
        ids = np.concatenate([[0], rng.integers(5, 200000, n - 2), [2]]).astype(np.int64)[None]
        feeds.append({"input_ids": ids, "attention_mask": np.ones_like(ids)})
    with torch.no_grad():
        ref = [w(torch.from_numpy(x["input_ids"]), torch.from_numpy(x["attention_mask"])).numpy() for x in feeds]
    check_graph("classifier", cm, feeds, ref)
    return {"classifier": describe(out / "classifier.xml"), "openvino": ov.__version__, "torch": torch.__version__}


def export_graphs(cfg, f):
    """Export only the verified production classifier. Numerical fixtures cannot pass this guard."""
    pair(f)
    out = f["models"]
    made_from = graphs_made_from(f)
    if all((out / f"{k}.{ext}").is_file() for k in GRAPHS for ext in ("xml", "bin")) and \
            is_current(out / "graphs.json", made_from):
        log.info("graphs: %s is current", out)
        return
    info = convert_graph(f["classifier"], out, classifier_run(f["classifier"])["max_tokens"])
    write_text(out / "graphs.json", json.dumps(info, indent=1))
    record_inputs(out / "graphs.json", made_from)


# ---------------------------------------------------------------- 2. the validation and test lines

def links(cfg):
    """(allowlist, canonical link): the module's default allowed_links (the plugin sends data/v4/allowlist.txt's
    entries at start) and bundle.toml's links.canonical, which ECC-Server puts in place of each allowed link."""
    import tomllib

    from .line_labels import allowlist
    canon = tomllib.loads((cfg.path(cfg["bundle"]["private"]) / "bundle.toml").read_text(encoding="utf-8"))["links"]
    return allowlist(cfg), canon["canonical"]


def lines_path(cfg):
    """The selected table version of the export batch."""
    from .line_labels import table_path
    return table_path(cfg, cfg["export"]["batch"])


def load_lines(cfg):
    """Every row of the runs that hold a val or test file of [export] batch (the line-label table's `split`), in run order, and each
    judged row of those files with its split as its period: (rows: DataFrame with the columns of ecc.inputs plus
    transcript and judged, meta: DataFrame transcript, id, period, row). The text is what the classifier reads live: each link
    the module's default allowlist allows replaced by the canonical one (links)."""
    import pandas as pd
    allow, canon = links(cfg)
    import pyarrow.parquet as pq
    from .trainset import with_raw
    path = lines_path(cfg)
    cols = ["transcript", "run", "id", "time", "channel", "partner", "speaker", "account", "text", "judged", "split"]
    names = pq.ParquetFile(path).schema_arrow.names
    df = with_raw(cfg, pd.read_parquet(path, columns=cols + (["raw_text"] if "raw_text" in names else ["row"])))
    held_out = df.split.isin(["val", "test"])
    runs = set(df.run[held_out])
    keep = df.run.isin(runs).to_numpy()
    df, held_out = df[keep].reset_index(drop=True), held_out[keep].to_numpy()   # whole runs: contexts as in training
    df["text"] = [I.model_text(t, allow, canon) if isinstance(t, str) else t for t in df.text]
    j = np.flatnonzero(df.judged.to_numpy() & held_out)
    meta = df.iloc[j][["transcript", "id", "split"]].rename(columns={"split": "period"})
    meta["row"] = j
    return df, meta[["transcript", "id", "period", "row"]].reset_index(drop=True)


def model_inputs(cfg, df, rows):
    """Causal context, exact classifier ids and full-table complete-target routes."""
    from .coverage import contexts
    f = folders(cfg)
    ctok = I.ClassifierInputs(f["classifier"] / "tokenizer.json")
    n = classifier_run(f["classifier"])["max_tokens"]
    ctx, routes = contexts(df, ctok.tok, n)
    return ctx, ctok.build(I.render(df, ctx, rows), n), routes


def torch_made_from(cfg, f):
    """The classifier, immutable table, live input rules and library versions used for scores."""
    from .core import source_hash
    return {"classifier": fingerprint(f["classifier"]), "lines": fingerprint(lines_path(cfg)),
            "allowlist": fingerprint(cfg.path(cfg["line_labels"]["allowlist"])), "canonical": links(cfg)[1], **versions(),
            "code": source_hash("core", "inputs", "model_text", "models", "export", "line_labels", "trainset", "coverage")}


def torch_scores(cfg, f, df, meta):
    """Score the classifier on every covered judged key. Rejected inputs have absent logits."""
    import pandas as pd
    path = f["out"] / "torch.parquet"
    made_from = torch_made_from(cfg, f)
    if is_current(path, made_from):
        return pd.read_parquet(path)
    import torch
    from .models import fp32, limit_gpu_memory, load_classifier, score_classifier
    _, cids, all_routes = model_inputs(cfg, df, meta.row.to_numpy())
    routes = all_routes[meta.row.to_numpy()]
    torch.set_num_threads(6)
    fp32()
    limit_gpu_memory(log)
    dev = torch.device("cuda")
    zc = np.full((len(meta), len(I.OUTS)), np.nan, np.float32)
    covered = np.flatnonzero(routes == "scored")
    if len(covered):
        zc[covered] = score_classifier(load_classifier(f["classifier"], dev), [cids[k] for k in covered], dev)
        torch.cuda.empty_cache()
    if not np.isfinite(zc[covered]).all():
        raise SystemExit("the classifier returned a non-finite logit")
    out = meta[["transcript", "id", "period"]].copy()
    out["coverage"] = routes
    for k, o in enumerate(I.OUTS):
        out[f"classifier_{o}"] = zc[:, k]
    out.to_parquet(path, index=False)
    record_inputs(path, made_from)
    return out


# ---------------------------------------------------------------- 2b. thresholds

def calibration_sources(cfg, f):
    """The exact score, selected table and selected hold-label files used for calibration."""
    from .trainset import hold_path
    from .base import sha256
    from .core import source_hash
    table = lines_path(cfg)
    hold = hold_path(cfg, cfg["export"]["batch"])

    def mark(path):
        return {"path": str(path), "fingerprint": fingerprint(path), "sha256": sha256(path)}

    return {"torch": fingerprint(f["out"] / "torch.parquet"), "period": "val",
            "table": mark(table), "hold": mark(hold), "guard_band": CPU_TOLERANCE,
            "hold_shares": SHARES, "code": source_hash("calibrate", "inputs")}


def thresholds_made_from(cfg, f):
    """Validation-only calibration sources, verified against the model's training provenance."""
    current = calibration_sources(cfg, f)
    batch = cfg["export"]["batch"]
    model = classifier_run(f["classifier"])
    recorded = model.get("made_from", {}).get("hold", {}).get(batch)
    inputs = model.get("made_from", {}).get("inputs", {})
    if inputs.get("tables", {}).get(batch) != current["table"]["fingerprint"]:
        raise SystemExit("selected immutable table differs from the classifier's training provenance")
    if inputs.get("allowlist") != fingerprint(cfg.path(cfg["line_labels"]["allowlist"])) or inputs.get("canonical") != links(cfg)[1]:
        raise SystemExit("selected link rules differ from the classifier's training provenance")
    if recorded != current["hold"]["fingerprint"] or not recorded:
        raise SystemExit("selected hold labels differ from the classifier's training provenance")
    return {**current, "model_inputs": model["made_from"]}


def slot_stale(cfg, f):
    """Why `uv run export` would make part of the slot again, or None when it would make nothing again but the parity
    run: the graphs, torch.parquet and thresholds.json each against what it records it was made from."""
    if not (all((f["models"] / f"{k}.bin").exists() for k in GRAPHS) and is_current(
            f["models"] / "graphs.json", graphs_made_from(f))):
        return "the models, code or libraries changed since the graphs were made"
    if not is_current(f["out"] / "torch.parquet", torch_made_from(cfg, f)):
        return ("no torch.parquet" if not (f["out"] / "torch.parquet").exists() else
                "the models, code, [export] batch, the allowlist, the canonical link or the libraries changed since "
                "torch.parquet was made")
    path = f["out"] / "thresholds.json"
    if not path.exists():
        return "no thresholds.json"
    try:
        slot_thresholds(cfg, f)
    except SystemExit as error:
        return str(error)
    return None


def set_thresholds(cfg, f, meta, t):
    """Write validation calibration only after checking exact model and source provenance."""
    import pandas as pd
    from .trainset import hold_path
    provenance = thresholds_made_from(cfg, f)
    table = pd.read_parquet(lines_path(cfg))
    labels = pd.read_parquet(hold_path(cfg, cfg["export"]["batch"]), columns=["transcript", "id", "hold", "session", "ban_here"])
    out = calibrate(t, meta, table[table.judged].copy(), labels)
    out.update(coverage_tokens=classifier_run(f["classifier"])["max_tokens"], models=model_marks(f), made_from=provenance)
    path = f["out"] / "thresholds.json"
    if not path.exists() or json.loads(path.read_text(encoding="utf-8")) != out:
        write_text(path, json.dumps(out, indent=1) + "\n")


# ---------------------------------------------------------------- 3. parity

def parity_rows(t, thr, n, full, files):
    """Include every covered target near any named threshold, plus a reproducible sample."""
    rows = np.arange(len(t))
    if files:
        rows = rows[t.transcript.isin(sorted(t.transcript.unique())[:files]).to_numpy()]
    if full:
        return rows
    near = np.zeros(len(t), bool)
    for threshold in thr.values():
        near |= (t.coverage.to_numpy() == "scored") & (np.abs(t.classifier_hold.to_numpy() - threshold) <= NEAR)
    must = rows[near[rows]]
    rest = np.setdiff1d(rows, must)
    extra = np.random.default_rng(SEED).choice(rest, size=max(0, min(len(rest), n - len(must))), replace=False)
    return np.sort(np.concatenate([must, extra]))


def ov_scores(cfg, f, df, meta, picked):
    """The graph's actual CPU logits. Rejected targets never invoke it."""
    import openvino as ov
    req = ov.Core().compile_model(f["models"] / "classifier.xml", "CPU", COMPILE).create_infer_request()
    rows = meta.row.to_numpy()[picked]
    _, cids, all_routes = model_inputs(cfg, df, rows)
    zc = np.full((len(rows), len(I.OUTS)), np.nan, np.float32)
    with progress("parity (classifier, CPU)", len(rows), "lines") as advance:
        for r, row in enumerate(rows):
            if all_routes[row] == "scored":
                ids = cids[r].astype(np.int64)[None]
                req.infer({"input_ids": ids, "attention_mask": np.ones_like(ids)})
                zc[r] = req.get_output_tensor(0).data[0]
            advance()
    return zc


def parity(cfg, f, df, meta, t, args):
    """Compare all eight outputs and decisions on identical real inputs."""
    thr = thresholds(cfg, f)
    picked = parity_rows(t, thr, args.sample, args.full, args.files)
    zc = ov_scores(cfg, f, df, meta, picked)
    tp = t.iloc[picked].reset_index(drop=True)
    covered = tp.coverage.to_numpy() == "scored"
    tc = tp[["classifier_" + o for o in I.OUTS]].to_numpy()
    if not covered.any() or not np.isfinite(zc[covered]).all():
        raise SystemExit("parity needs finite real covered outputs")
    delta = np.abs(zc[covered] - tc[covered])
    maximum = float(delta.max())
    bad = []
    for name, threshold in thr.items():
        a, b = tc[covered, I.HOLD], zc[covered, I.HOLD]
        differing = ((a >= threshold) != (b >= threshold)) & (a != threshold)
        if differing.any():
            bad.append({"threshold": name, "count": int(differing.sum())})
    result = {"sampled": len(tp), "covered": int(covered.sum()),
              "rejected": int((~covered).sum()), "max_abs_diff": maximum,
              "per_output_max": dict(zip(I.OUTS, map(float, delta.max(axis=0)))), "decision_errors": bad}
    write_text(f["out"] / "parity.json", json.dumps(result, indent=1) + "\n")
    if maximum > CPU_TOLERANCE or bad:
        raise SystemExit(f"parity failed: error {maximum}, tolerance {CPU_TOLERANCE}, decisions {bad}")


def body(cfg, args):
    f = folders(cfg)
    for k in ("classifier",):
        if not f[k].is_dir():
            raise SystemExit(f"[export] {k}: {f[k]} is not a model folder")
    export_graphs(cfg, f)
    df, meta = load_lines(cfg)
    t = torch_scores(cfg, f, df, meta)
    if not (t[["transcript", "id"]].to_numpy() == meta[["transcript", "id"]].to_numpy()).all():
        raise SystemExit("torch.parquet's lines are not the current validation and test lines")
    set_thresholds(cfg, f, meta, t)
    if not args.no_parity:
        parity(cfg, f, df, meta, t, args)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", action="store_true", help="parity on every validation and test line (about 100 min)")
    ap.add_argument("--sample", type=int, default=None, help="lines of the sampled parity run ([export] sample)")
    ap.add_argument("--files", type=int, default=0, help="parity only on the first N validation and test files")
    ap.add_argument("--no-parity", action="store_true", help="export the graphs and score torch, without the parity run")

    def body_(cfg, args):
        if args.sample is None:
            args.sample = cfg["export"]["sample"]
        body(cfg, args)

    run("export", body_, ap)
