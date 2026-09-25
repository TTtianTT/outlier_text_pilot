"""Offline plumbing checks only. Features and scores here are ARTIFICIAL."""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
from .io import unit, stable_seed, write_json, write_jsonl


class SyntheticBackend:
    def __init__(self, dimension=32):
        self.dimension = dimension
        self.signature = {"backend": "synthetic", "feature_version": "hash-noise-v1", "dimension": dimension}

    def feature(self, text):
        # Deliberately random-looking features, not a model of linguistic behavior.
        rng = np.random.default_rng(stable_seed("synthetic-feature-v1", text))
        v = unit(rng.normal(size=self.dimension))
        n = max(1, len(text.split()))
        return v, {"n_tokens": n, "nll_sum": n * 2.0, "nll_mean": 2.0}

    def close(self): pass


def make_demo(cfg, run):
    run = Path(run)
    if (run / "manifest.json").exists():
        raise ValueError("demo needs a new run directory")
    run.mkdir(parents=True, exist_ok=True)
    model = SyntheticBackend()
    rng = np.random.default_rng(cfg["seed"])
    parents, rows, vectors, refs, costs = [], [], [], [], []
    for split in ("dev", "test", "ood"):
        for pi in range(12):
            pid = f"{split}-{pi:03d}"
            text = f"Synthetic carrier {pid}."
            p = {"id": pid, "group_id": pid, "text": text, "split": split, "domain": "SYNTHETIC"}
            parents.append(p)
            reference, _ = model.feature(text)
            if split == "dev": refs.append(reference)
            for j in range(32):
                text = f"Synthetic carrier {pid} variant {j:03d}."
                v, stats = model.feature(text)
                row = {"candidate_id": f"{pid}:{j}", "parent_id": pid, "group_id": pid, "text": text,
                       "reference": p["text"], "split": split, "domain": "SYNTHETIC", **stats,
                       "sem_dist": float(rng.uniform(.02, .04)), "nli_min": .98,
                       "nli_forward": .98, "nli_backward": .98, "zipf_mean": 4.5,
                       "edit_ratio": .15, "ascii_printable": True,
                       "behavior_dist": float(1 - v @ reference), "vector_index": len(rows)}
                rows.append(row); vectors.append(v)
            costs.append({"parent_id": pid, "split": split, "generation_attempts": 32, "generated_tokens": 0,
                          "generation_seconds": 0., "feature_seconds": 0., "quality_seconds": 0.})
    write_json(run / "manifest.json", {"config": cfg, "synthetic": True, "warning": "NOT LLM EVIDENCE"})
    write_jsonl(run / "parents.jsonl", parents)
    write_jsonl(run / "features.jsonl", rows)
    write_jsonl(run / "costs.jsonl", costs)
    np.savez_compressed(run / "vectors.npz", vectors=np.asarray(vectors), calibration=np.asarray(refs))
    write_json(run / "feature_signature.json", model.signature)
