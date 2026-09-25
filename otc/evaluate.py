from __future__ import annotations
import csv
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
from .io import read_jsonl, read_json, write_json, write_jsonl, stable_seed, softmax, digest, file_hash
from .selection import ResidualScorer, eligible, select_groups, CONFOUNDS
from .readout import planes, read, public_eval_key

METHODS = ("outlier", "matched", "random", "nearest")


def csv_write(path, rows):
    if not rows: return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def two_way_ci(records, metric, method, comparator, repeats, seed):
    """Resample original source groups and readout keys (crossed bootstrap).

    Average within a source group first; no treating paraphrases as independent.
    Returns a conditional-on-this-selected-protocol exploratory interval, no p-value.
    """
    table = defaultdict(dict)
    for r in records:
        if r["method"] in (method, comparator):
            table[(r["group_id"], r["key_seed"], r["method"])].setdefault(r["parent_id"], r[metric])
    groups = sorted({r["group_id"] for r in records})
    keys = sorted({r["key_seed"] for r in records})
    if not groups or not keys:
        return {"delta": None, "ci_low": None, "ci_high": None, "source_groups": 0, "keys": 0}
    values = np.zeros((len(groups), len(keys)))
    for i, group in enumerate(groups):
        for j, key in enumerate(keys):
            a = table[(group, key, method)]
            b = table[(group, key, comparator)]
            common = sorted(a.keys() & b.keys())
            if not common:
                raise ValueError("paired bootstrap received an incomplete design")
            values[i, j] = np.mean([a[p] - b[p] for p in common])
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(repeats):
        ii = rng.integers(len(groups), size=len(groups)); jj = rng.integers(len(keys), size=len(keys))
        means.append(float(values[np.ix_(ii, jj)].mean()))
    return {"delta": float(values.mean()), "ci_low": float(np.quantile(means, .025)),
            "ci_high": float(np.quantile(means, .975)), "source_groups": len(groups), "keys": len(keys)}


def pool_distribution(labels, margins, rows, width, margin):
    """Finite-pool proxy: p proportional to exp(-plain-text total NLL).

    NOT the true language distribution, proposal distribution, or a security bound.
    Uniform target symbols; unavailable symbols abort. TV is conditional on success.
    """
    weights = softmax(-np.asarray([r["nll_sum"] for r in rows]))
    valid = margins >= margin
    q = np.bincount(labels[valid], weights=weights[valid], minlength=2**width)
    present = q > 0
    stego = np.zeros(len(rows))
    if np.any(present):
        for i in np.flatnonzero(valid):
            if q[labels[i]] > 0:
                stego[i] = weights[i] / q[labels[i]] / present.sum()
        tv = float(np.abs(stego - weights).sum() / 2)
    else:
        tv = None
    return weights, q, float(1 - present.mean()), tv


def evaluate(cfg, run):
    run = Path(run)
    rows = read_jsonl(run / "features.jsonl")
    parents = read_jsonl(run / "parents.jsonl")
    z = np.load(run / "vectors.npz", allow_pickle=False)
    features, calibration = z["vectors"], z["calibration"]
    if len(features) != len(rows): raise ValueError("feature-row mismatch")
    if any(r["vector_index"] != i for i, r in enumerate(rows)): raise ValueError("bad vector_index")
    selected_rows = [r for r in rows if eligible(r, cfg)]
    dev = [r for r in selected_rows if r["split"] == "dev"]
    if len(dev) < 24:
        raise ValueError(f"only {len(dev)} eligible dev candidates; inspect generation/filters before fitting")
    scorer = ResidualScorer().fit(dev, cfg["selection"]["ridge_alpha"])
    residuals = scorer.score(selected_rows)
    write_json(run / "residual_fit.json", scorer.describe())
    by_parent = defaultdict(list)
    for r, residual in zip(selected_rows, residuals):
        r = dict(r, residual=float(residual))
        by_parent[r["parent_id"]].append(r)
    selections, diagnostics = {}, []
    for p in parents:
        candidates = by_parent[p["id"]]
        groups, diag = select_groups(candidates, [r["residual"] for r in candidates], cfg, p["id"])
        selections[p["id"]] = None if groups is None else {m: [candidates[i]["vector_index"] for i in inds] for m, inds in groups.items()}
        diagnostics.append({"parent_id": p["id"], "split": p["split"], **diag})
    write_json(run / "selections.json", selections)
    write_jsonl(run / "selection_diagnostics.jsonl", diagnostics)
    # Export only public decoder state. No candidate vectors, ids, target bits, or keys.
    profile = run / "profile"; profile.mkdir(exist_ok=True)
    np.save(profile / "calibration.npy", calibration)
    signature = read_json(run / "feature_signature.json")
    profile_meta = {"version": 1, "feature_signature": signature,
                    "calibration_sha256": file_hash(profile / "calibration.npy"),
                    "calibration_split": "dev", "calibration_count": len(calibration),
                    "feature_dimension": features.shape[1]}
    write_json(profile / "profile.json", profile_meta)
    results, detected = [], []
    for p in parents:
        groups = selections[p["id"]]
        for key_seed in cfg["evaluation"]["key_seeds"]:
            key = public_eval_key(key_seed)
            for width in cfg["evaluation"]["bits"]:
                r, thresholds = planes(key, "PUBLIC-pilot-v1", stable_seed(p["id"]), width, calibration)
                for method in METHODS:
                    inds = [] if groups is None else groups[method]
                    source = [rows[i] for i in inds]
                    if inds:
                        t0 = time.perf_counter()
                        labels, margins = read(features[inds], r, thresholds)
                        readout_time = time.perf_counter() - t0
                    for margin in cfg["evaluation"]["margins"]:
                        base = {"parent_id": p["id"], "group_id": p["group_id"], "split": p["split"],
                                "key_seed": key_seed, "bits": width, "margin": margin, "method": method,
                                "matched_parent": bool(inds), "candidate_count": len(inds)}
                        if not inds:
                            results.append({**base, "coverage": 0.0, "full_coverage": 0.0,
                                            "mean_margin": None, "pool_proxy_tv": None,
                                            "target_abort": 1.0, "readout_seconds": 0.0})
                            continue
                        ok = margins >= margin
                        covered = len(set(labels[ok].tolist()))
                        weights, q, abort, tv = pool_distribution(labels, margins, source, width, margin)
                        results.append({**base, "coverage": covered / 2**width,
                                        "full_coverage": float(covered == 2**width),
                                        "mean_margin": float(margins.mean()), "pool_proxy_tv": tv,
                                        "target_abort": abort, "readout_seconds": readout_time})
                        # One paired sample per parent/key for an inexpensive detector.
                        if width == cfg["evaluation"]["primary_bits"] and margin == cfg["evaluation"]["primary_margin"]:
                            rng = np.random.default_rng(stable_seed(cfg["seed"], p["id"], key_seed, "detector"))
                            target = int(rng.integers(2**width))
                            can = np.flatnonzero(ok & (labels == target))
                            if len(can):
                                chosen = int(rng.choice(can, p=weights[can] / weights[can].sum()))
                                cover = int(rng.choice(len(source), p=weights))
                                for label, index in ((1, chosen), (0, cover)):
                                    detected.append({"parent_id": p["id"], "split": p["split"], "method": method,
                                                     "key_seed": key_seed, "label": label, "text": source[index]["text"]})
    write_jsonl(run / "metrics.jsonl", results)
    write_jsonl(run / "detector_samples.jsonl", detected)
    summaries, deltas = [], []
    for split in ("test", "ood"):
        for width in cfg["evaluation"]["bits"]:
            for margin in cfg["evaluation"]["margins"]:
                cells = [r for r in results if r["split"] == split and r["bits"] == width and r["margin"] == margin]
                for method in METHODS:
                    rr = [r for r in cells if r["method"] == method]
                    ok = [r for r in rr if r["matched_parent"]]
                    tvs = [r["pool_proxy_tv"] for r in ok if r["pool_proxy_tv"] is not None]
                    summaries.append({"split": split, "bits": width, "margin": margin, "method": method,
                                      "parents": len({r["parent_id"] for r in rr}),
                                      "matched_parents": len({r["parent_id"] for r in ok}),
                                      "coverage_all": float(np.mean([r["coverage"] for r in rr])),
                                      "coverage_matched": float(np.mean([r["coverage"] for r in ok])) if ok else None,
                                      "full_coverage_all": float(np.mean([r["full_coverage"] for r in rr])),
                                      "pool_proxy_tv_success": float(np.mean(tvs)) if tvs else None})
                paired_cells = [r for r in cells if r["matched_parent"]]
                ci = two_way_ci(paired_cells, "coverage", "outlier", "matched", cfg["evaluation"]["bootstrap"],
                                stable_seed(cfg["seed"], split, width, margin))
                deltas.append({"split": split, "bits": width, "margin": margin, "comparison": "outlier-minus-matched", **ci})
    csv_write(run / "summary.csv", summaries); csv_write(run / "paired_deltas.csv", deltas)
    detector = detector_report(detected, cfg)
    write_json(run / "detector.json", detector)
    audit = []
    for split in ("test", "ood"):
        samples = [r for r in selected_rows if r["split"] == split]
        rng = np.random.default_rng(stable_seed(cfg["seed"], split, "audit"))
        for i in rng.choice(len(samples), size=min(50, len(samples)), replace=False) if samples else []:
            row = samples[i]
            audit.append({"parent_id": row["parent_id"], "candidate_id": row["candidate_id"],
                          "reference": row["reference"], "candidate": row["text"], "semantically_equivalent_0_1": "",
                          "natural_0_1": "", "notes": ""})
    csv_write(run / "human_audit.csv", audit)
    write_report(run, signature, summaries, deltas, diagnostics, detector, cfg)
    print(f"report: {run / 'report.md'}")


def detector_report(samples, cfg):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    output = {"scope": "accepted transmissions; same-pool NLL-proxy covers; char n-gram detector only", "results": []}
    key_seeds = cfg["evaluation"]["key_seeds"]
    cut = max(1, len(key_seeds) // 2)
    train_keys, eval_keys = set(key_seeds[:cut]), set(key_seeds[cut:])
    for method in METHODS:
        train = [r for r in samples if r["method"] == method and r["split"] == "dev" and r["key_seed"] in train_keys]
        if len(train) < 20 or len({r["text"] for r in train}) < 5:
            output["results"].append({"method": method, "status": "insufficient_dev_samples"}); continue
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, max_features=20000)
        x = vec.fit_transform([r["text"] for r in train])
        model = LogisticRegression(C=1.0, max_iter=1000, random_state=cfg["seed"]).fit(x, [r["label"] for r in train])
        for split in ("test", "ood"):
            test = [r for r in samples if r["method"] == method and r["split"] == split and r["key_seed"] in eval_keys]
            if len(test) < 10:
                output["results"].append({"method": method, "split": split, "status": "insufficient_test_samples"}); continue
            score = model.predict_proba(vec.transform([r["text"] for r in test]))[:, 1]
            output["results"].append({"method": method, "split": split, "status": "ok", "train_examples": len(train),
                                       "test_examples": len(test), "test_parents": len({r["parent_id"] for r in test}),
                                       "auroc": float(roc_auc_score([r["label"] for r in test], score))})
    return output


def write_report(run, signature, summaries, deltas, diagnostics, detector, cfg):
    synthetic = signature["backend"] == "synthetic"
    primary = [r for r in summaries if r["bits"] == cfg["evaluation"]["primary_bits"] and r["margin"] == cfg["evaluation"]["primary_margin"]]
    lines = ["# Outlier text pilot", "", "**SYNTHETIC SMOKE TEST — NOT LLM EXPERIMENTAL EVIDENCE.**" if synthetic else "Real-model pilot; human equivalence audit is still required.", "",
             f"Primary setting: {cfg['evaluation']['primary_bits']} bit(s), margin {cfg['evaluation']['primary_margin']}.", "",
             "| Split | Method | Matched / total parents | Coverage (all) | Coverage (matched) | Full coverage (all) |",
             "|---|---|---:|---:|---:|---:|"]
    for r in primary:
        v = r["coverage_matched"]
        lines.append(f"| {r['split']} | {r['method']} | {r['matched_parents']} / {r['parents']} | {r['coverage_all']:.4f} | {v if v is not None else 'NA'} | {r['full_coverage_all']:.4f} |")
    lines += ["", "Paired outlier minus matched-control differences (source-group × key bootstrap; exploratory 95% intervals):", ""]
    for r in deltas:
        if r["bits"] == cfg["evaluation"]["primary_bits"] and r["margin"] == cfg["evaluation"]["primary_margin"]:
            if r["delta"] is None:
                lines.append(f"- {r['split']}: effect NA, CI NA (no comparable matched parents); delivery remains reported in coverage_all.")
            else:
                lines.append(f"- {r['split']}: {r['delta']:.4f}, [{r['ci_low']:.4f}, {r['ci_high']:.4f}], {r['source_groups']} groups, {r['keys']} keys (common matched subset).")
    lines += ["", "## Checks and interpretation", "",
              "- Unmatched parents remain in coverage_all with zero delivery. coverage_matched is explicitly conditional.",
              "- Main proxy is last-layer representation distance, not a demonstrated change of downstream behavior.",
              "- Quality and NLI are independent models, but neither replaces human review. Audit human_audit.csv.",
              "- NLL and mean word frequency are confound proxies. Matching does not establish causality or exact distribution preservation.",
              "- Median thresholds balance dev-reference marginal bits approximately; joint codeword mass need not be balanced.",
              "- pool_proxy_tv_success is conditional finite-pool TV under exp(-total NLL), not global text TV or a security bound.",
              "- Detector AUROC uses held-out parents and held-out evaluation keys. One weak detector cannot certify security.",
              "- These are coding-layer coverage metrics. Full-message, authentication and wire overhead are in codec diagnostics.",
              "- See selection_diagnostics.jsonl for pair counts, exclusion reasons and confound balance; costs.jsonl includes generation and evaluator work.",
              "- Readout times omit plane construction and full feature extraction; do not present them as end-to-end latency.",
              "- Development residuals are in-sample and not evidence. Do not tune on test/ood.",
              "", "## Selection accounting", ""]
    for split in ("dev", "test", "ood"):
        ds = [d for d in diagnostics if d["split"] == split]
        counts = {reason: sum(d["reason"] == reason for d in ds) for reason in sorted({d["reason"] for d in ds})}
        lines.append(f"- {split}: {counts}")
    costs = read_jsonl(run / "costs.jsonl")
    lines += ["", "## Shared preprocessing cost", "", f"Generation attempts: {sum(c['generation_attempts'] for c in costs)}.",
              f"Generated tokens (including rejected candidates): {sum(c['generated_tokens'] for c in costs)}.",
              f"Generation + feature + quality seconds: {sum(c['generation_seconds'] + c['feature_seconds'] + c['quality_seconds'] for c in costs):.2f}.",
              "These costs are shared across methods on a common proposal pool; label cold and amortized costs separately.",
              "", "## Next decision", "",
              "Inspect exclusions and manually review semantics first. A positive paired difference is a pilot signal only.",
              "Check the OOD split and a second public model with the same frozen protocol before investing in a larger codec."]
    (run / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
