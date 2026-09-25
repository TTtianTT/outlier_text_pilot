"""Dev-only cached 1-bit diagnostics; no generation, test tuning, or codec trials."""
from __future__ import annotations
import argparse
import copy
from collections import Counter
from pathlib import Path
import numpy as np
from .evaluate import csv_write, two_way_ci
from .io import read_json, read_jsonl, write_json, write_jsonl, stable_seed, file_hash
from .readout import planes, read, public_eval_key
from .selection import eligible, ResidualScorer, select_groups


def independent_groups(rows, cfg, parent_id):
    keep = cfg["selection"]["keep"]
    groups = {"full_pool": list(range(len(rows))), "random": [], "nearest": []}
    if len(rows) >= keep:
        rng = np.random.default_rng(stable_seed(cfg["seed"], parent_id, "random-control"))
        groups["random"] = rng.choice(len(rows), keep, replace=False).tolist()
        groups["nearest"] = sorted(range(len(rows)), key=lambda i: (rows[i]["sem_dist"], rows[i]["candidate_id"]))[:keep]
    return groups


def symbol_metrics(labels, margins, guard):
    accepted = labels[margins >= guard]
    covered = len(set(accepted.tolist()))
    return {"coverage": covered / 2, "full_coverage": float(covered == 2),
            "accepted_zero": int(np.sum(accepted == 0)), "accepted_one": int(np.sum(accepted == 1)),
            "single_region": int(covered == 1), "no_accepted_symbols": int(covered == 0)}


def summarize(records):
    summaries = []
    for scope, method, guard in sorted({(r["scope"], r["method"], r["margin"]) for r in records}):
        rr = [r for r in records if (r["scope"], r["method"], r["margin"]) == (scope, method, guard)]
        available = [r for r in rr if r["candidate_count"]]
        summaries.append({"scope": scope, "method": method, "margin": guard,
                          "parents": len({r["parent_id"] for r in rr}),
                          "available_parents": len({r["parent_id"] for r in available}),
                          "parent_key_cells": len(rr),
                          "coverage_all": float(np.mean([r["coverage"] for r in rr])),
                          "coverage_available": float(np.mean([r["coverage"] for r in available])) if available else None,
                          "full_coverage_all": float(np.mean([r["full_coverage"] for r in rr])),
                          "single_region_all": float(np.mean([r["single_region"] for r in rr])),
                          "no_accepted_symbols_all": float(np.mean([r["no_accepted_symbols"] for r in rr]))})
    return summaries


def audit_samples(rows, cfg, output):
    """One passing + one rejected candidate per parent, blind to method and readout."""
    selected = []
    for pid in sorted({r["parent_id"] for r in rows}):
        for passed in (True, False):
            choices = [r for r in rows if r["parent_id"] == pid and eligible(r, cfg) == passed]
            if choices:
                selected.append(min(choices, key=lambda r: stable_seed(cfg["seed"], r["candidate_id"], "dev-audit-v1")))
    selected.sort(key=lambda r: stable_seed(cfg["seed"], r["candidate_id"], "audit-order-v1"))
    blind, key = [], []
    for i, r in enumerate(selected):
        aid = f"audit-{i+1:03d}"
        blind.append({"audit_id": aid, "reference": r["reference"], "candidate": r["text"],
                      "facts_preserved_yes_no_uncertain": "", "natural_yes_no_uncertain": "",
                      "notes": "", "reviewer": ""})
        key.append({"audit_id": aid, "parent_id": r["parent_id"], "candidate_id": r["candidate_id"],
                    "filter_passed": eligible(r, cfg), "cosine": 1-r["sem_dist"], "nli_min": r["nli_min"],
                    "edit_ratio": r["edit_ratio"], "n_tokens": r["n_tokens"], "ascii_printable": r["ascii_printable"]})
    csv_write(output / "human_audit_blind.csv", blind)
    csv_write(output / "audit_filter_key.csv", key)
    return len(blind)


def diagnose(run, output):
    run, output = Path(run), Path(output)
    if output.exists(): raise ValueError("choose a NEW output directory")
    cfg = copy.deepcopy(read_json(run / "manifest.json")["config"])
    cfg["selection"]["keep"] = 2
    cfg["evaluation"]["bits"] = [1]
    guards = sorted({0.0, cfg["evaluation"]["primary_margin"]})
    parents = [p for p in read_jsonl(run / "parents.jsonl") if p["split"] == "dev"]
    rows = [r for r in read_jsonl(run / "features.jsonl") if r["split"] == "dev"]
    with np.load(run / "vectors.npz", allow_pickle=False) as z:
        vectors, calibration = z["vectors"], z["calibration"]
    good = [r for r in rows if eligible(r, cfg)]
    if len(good) < 24: raise ValueError("insufficient eligible dev candidates")
    scorer = ResidualScorer().fit(good, cfg["selection"]["ridge_alpha"])
    residuals = dict(zip([r["candidate_id"] for r in good], scorer.score(good)))
    output.mkdir(parents=True)
    write_json(output / "protocol.json", {"source_run": str(run), "split": "dev", "bits": 1,
               "keep": 2, "margins": guards, "config": cfg,
               "source_hashes": {name: file_hash(run / name) for name in
                                 ("manifest.json", "features.jsonl", "vectors.npz", "feature_signature.json")},
               "warning": "Exploratory in-sample dev diagnosis; no confirmatory inference. Full pool has variable budget."})
    records, diagnostics, geometry = [], [], []
    for p in parents:
        candidates = [r for r in good if r["parent_id"] == p["id"]]
        independent = independent_groups(candidates, cfg, p["id"])
        paired, diag = select_groups(candidates, [residuals[r["candidate_id"]] for r in candidates], cfg, p["id"])
        diagnostics.append({"parent_id": p["id"], "split": "dev", **diag})
        x = vectors[[r["vector_index"] for r in candidates]]
        for seed in cfg["evaluation"]["key_seeds"]:
            r, tau = planes(public_eval_key(seed), "PUBLIC-pilot-v1", stable_seed(p["id"]), 1, calibration)
            labels, margins = read(x, r, tau)
            signed = (x @ r.T - tau).ravel()
            geometry.append({"parent_id": p["id"], "key_seed": seed, "eligible": len(candidates),
                             "signed_projection_min": float(signed.min()) if len(signed) else None,
                             "signed_projection_max": float(signed.max()) if len(signed) else None,
                             "projection_span": float(np.ptp(signed)) if len(signed) else None})
            scopes = {"independent": independent,
                      "paired_protocol": paired or {m: [] for m in ("outlier", "matched", "random", "nearest")}}
            if paired: scopes["common_matched"] = paired
            for scope, groups in scopes.items():
                for method, inds in groups.items():
                    for guard in guards:
                        records.append({"parent_id": p["id"], "group_id": p["group_id"], "split": "dev",
                                        "key_seed": seed, "scope": scope, "method": method, "bits": 1,
                                        "margin": guard, "candidate_count": len(inds),
                                        **symbol_metrics(labels[inds], margins[inds], guard)})
    summaries = summarize(records)
    effects = []
    for guard in guards:
        rr = [r for r in records if r["scope"] == "common_matched" and r["margin"] == guard]
        effects.append({"margin": guard, "comparison": "outlier-minus-matched", "scope": "common_matched",
                        **two_way_ci(rr, "coverage", "outlier", "matched", cfg["evaluation"]["bootstrap"],
                                     stable_seed(cfg["seed"], "dev-keep2", guard))})
    csv_write(output / "summary.csv", summaries)
    csv_write(output / "paired_deltas.csv", effects)
    csv_write(output / "pool_geometry.csv", geometry)
    write_jsonl(output / "metrics.jsonl", records)
    write_jsonl(output / "selection_diagnostics.jsonl", diagnostics)
    audit_count = audit_samples(rows, cfg, output)
    result = {"parents": len(parents), "feature_candidates": len(rows), "eligible_candidates": len(good),
              "matched_parents_keep2": sum(d["reason"] == "ok" for d in diagnostics),
              "selection_reasons": dict(Counter(d["reason"] for d in diagnostics)), "audit_samples": audit_count,
              "summaries": summaries, "effects": effects}
    write_json(output / "summary.json", result)
    print(f"Dev diagnostics saved to {output}; matched {result['matched_parents_keep2']}/{len(parents)}", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True); parser.add_argument("--out", required=True)
    args = parser.parse_args()
    diagnose(args.run, args.out)
