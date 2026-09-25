from __future__ import annotations
import difflib
import re
import time
from pathlib import Path
import numpy as np
from .io import corpus, digest, read_json, write_json, read_jsonl, write_jsonl, stable_seed


def run_manifest(cfg, run):
    run = Path(run); run.mkdir(parents=True, exist_ok=True)
    parents = corpus(cfg)
    signature = {"config": cfg, "corpus_hash": digest(parents), "version": 1}
    p = run / "manifest.json"
    if p.exists() and read_json(p) != signature:
        raise ValueError("run config/corpus changed; choose a NEW --run directory")
    if not p.exists():
        write_json(p, signature)
        write_jsonl(run / "parents.jsonl", parents)
    return parents


def safe_name(parent):
    return digest(parent["id"])[:24]


def generate(cfg, run):
    from .hf import CausalBackend, PROMPT_VERSION
    run = Path(run)
    parents = run_manifest(cfg, run)
    model = CausalBackend(cfg["model"])
    provenance = {"model": model.signature, "prompt_version": PROMPT_VERSION}
    pp = run / "generation_provenance.json"
    if pp.exists() and read_json(pp) != provenance:
        raise ValueError("generation model/tokenizer/runtime changed during resume")
    write_json(pp, provenance)
    for pi, parent in enumerate(parents):
        target = run / "candidate_shards" / (safe_name(parent) + ".json")
        if target.exists():
            continue
        t0 = time.perf_counter(); records = []; unique = set([parent["text"].casefold()]); tokens = 0
        for attempt in range(cfg["generation"]["attempts"]):
            seed = stable_seed(cfg["seed"], parent["id"], attempt)
            raw, n, ended = model.generate(parent["text"], attempt, seed, cfg["generation"])
            tokens += n
            text = raw.strip().strip('"').strip()
            reason = "ok"
            if not ended: reason = "generation_cutoff"
            elif not text or "\n" in text: reason = "empty_or_multiline"
            elif text.casefold() in unique: reason = "duplicate_or_identity"
            unique.add(text.casefold())
            records.append({"candidate_id": f"{parent['id']}:{attempt:03d}", "parent_id": parent["id"],
                            "attempt": attempt, "seed": seed, "text": text, "raw": raw,
                            "generated_tokens": n, "generation_status": reason})
        write_json(target, {"parent": parent, "attempts": records, "generated_tokens": tokens,
                            "generation_seconds": time.perf_counter() - t0})
        print(f"generate {pi+1}/{len(parents)} {parent['id']}: {sum(r['generation_status']=='ok' for r in records)} unique", flush=True)
    model.close()


def features(cfg, run):
    from .hf import CausalBackend, SemanticBackend
    from wordfreq import zipf_frequency
    run = Path(run); parents = run_manifest(cfg, run)
    shards = [run / "candidate_shards" / (safe_name(p) + ".json") for p in parents]
    if not all(p.exists() for p in shards):
        raise ValueError("run generate first; candidate shards are incomplete")
    # Cross-split exact text reuse is rejected, including candidate-parent collisions.
    seen = {p["text"].casefold(): p["split"] for p in parents}
    for p in shards:
        s = read_json(p)
        for r in s["attempts"]:
            if r["generation_status"] != "ok": continue
            k = r["text"].casefold(); split = s["parent"]["split"]
            if k in seen and seen[k] != split:
                raise ValueError("identical candidate text occurs across splits; deduplicate corpus and regenerate")
            seen[k] = split
    model = CausalBackend(cfg["model"])
    sig_path = run / "feature_signature.json"
    if sig_path.exists() and read_json(sig_path) != model.signature:
        raise ValueError("feature model/tokenizer/runtime changed during resume")
    write_json(sig_path, model.signature)
    # Heavy representation forwards cached once per parent, independent of evaluation keys.
    for pi, sp in enumerate(shards):
        source = read_json(sp); parent = source["parent"]
        target = run / "causal_shards" / (safe_name(parent) + ".npz")
        meta_path = target.with_suffix(".json")
        if target.exists() and meta_path.exists(): continue
        t0 = time.perf_counter(); vectors = []; rows = []
        ref, _ = model.feature(parent["text"])
        rejected = []
        for c in source["attempts"]:
            if c["generation_status"] != "ok": continue
            try:
                vector, stats = model.feature(c["text"])
            except ValueError as exc:
                rejected.append({"candidate_id": c["candidate_id"], "reason": str(exc)})
                continue
            tokens = re.findall(r"[A-Za-z]+", c["text"].lower())
            row = {**c, **stats, "split": parent["split"], "domain": parent["domain"],
                   "group_id": parent["group_id"], "reference": parent["text"],
                   "behavior_dist": float(np.clip(1 - vector @ ref, 0, 2)),
                   "edit_ratio": 1 - difflib.SequenceMatcher(None, parent["text"].lower().split(), c["text"].lower().split(), autojunk=False).ratio(),
                   "zipf_mean": float(np.mean([zipf_frequency(t, "en") for t in tokens])) if tokens else 0.0,
                   "ascii_printable": all(32 <= ord(ch) < 127 for ch in c["text"])}
            rows.append(row); vectors.append(vector)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, reference=ref, vectors=np.asarray(vectors).reshape(-1, len(ref)))
        write_json(meta_path, {"rows": rows, "rejected": rejected,
                               "feature_seconds": time.perf_counter() - t0})
        print(f"features {pi+1}/{len(shards)} {parent['id']}", flush=True)
    model.close()
    semantic = SemanticBackend(cfg["quality"])
    qp = run / "quality_provenance.json"
    if qp.exists() and read_json(qp) != semantic.provenance:
        raise ValueError("semantic/NLI evaluator changed during resume")
    write_json(qp, semantic.provenance)
    all_rows, all_vectors, refs, costs = [], [], [], []
    for parent, source_path in zip(parents, shards):
        base = run / "causal_shards" / (safe_name(parent) + ".npz")
        rowsmeta = read_json(base.with_suffix(".json")); src = read_json(source_path)
        z = np.load(base, allow_pickle=False)
        if parent["split"] == "dev": refs.append(z["reference"])
        qpath = run / "quality_shards" / (safe_name(parent) + ".json")
        if qpath.exists():
            q = read_json(qpath)
        else:
            t0 = time.perf_counter()
            measured = []
            for row in rowsmeta["rows"]:
                try:
                    measured.append({**row, **semantic.compare(parent["text"], row["text"]), "quality_status": "ok"})
                except ValueError as exc:
                    measured.append({**row, "quality_status": str(exc)})
            q = {"rows": measured, "quality_seconds": time.perf_counter() - t0}
            write_json(qpath, q)
        for row, vector in zip(q["rows"], z["vectors"]):
            if row["quality_status"] != "ok": continue
            row["vector_index"] = len(all_rows)
            all_rows.append(row); all_vectors.append(vector)
        costs.append({"parent_id": parent["id"], "split": parent["split"],
                      "generation_attempts": len(src["attempts"]), "generated_tokens": src["generated_tokens"],
                      "generation_seconds": src["generation_seconds"],
                      "feature_seconds": rowsmeta["feature_seconds"], "quality_seconds": q["quality_seconds"],
                      "feature_rejected": len(rowsmeta["rejected"]),
                      "quality_rejected": sum(r["quality_status"] != "ok" for r in q["rows"])})
    semantic.close()
    if not all_rows or len(refs) < 2: raise ValueError("no usable candidates or insufficient dev references")
    write_jsonl(run / "features.jsonl", all_rows)
    np.savez_compressed(run / "vectors.npz", vectors=np.asarray(all_vectors), calibration=np.asarray(refs))
    write_jsonl(run / "costs.jsonl", costs)
    print(f"saved {len(all_rows)} candidate features; calibration uses {len(refs)} dev parents only")
