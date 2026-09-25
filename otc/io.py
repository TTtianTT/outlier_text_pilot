from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import yaml


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, p)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(s) for s in f if s.strip()]


def write_jsonl(path, rows):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(canonical(r) + "\n" for r in rows), encoding="utf-8")


def stable_seed(*parts):
    return int(digest(parts)[:8], 16)


def config(path):
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if cfg["evaluation"]["bits"] != sorted(set(cfg["evaluation"]["bits"])):
        raise ValueError("bits must be sorted and unique")
    if any(b < 1 or b > 8 for b in cfg["evaluation"]["bits"]):
        raise ValueError("use 1..8 bits per sentence")
    ev = cfg["evaluation"]
    if ev["primary_bits"] not in ev["bits"] or ev["primary_margin"] not in ev["margins"]:
        raise ValueError("primary setting must be in the evaluation grid")
    if len(set(ev["key_seeds"])) != len(ev["key_seeds"]) or len(ev["key_seeds"]) < 2:
        raise ValueError("use at least two distinct evaluation key seeds")
    if any(m < 0 or not np.isfinite(m) for m in ev["margins"]):
        raise ValueError("margins must be finite and nonnegative")
    return cfg


def corpus(cfg):
    rows = read_jsonl(cfg["data"])
    ids, texts, group_splits = set(), set(), {}
    for r in rows:
        for key in ("id", "group_id", "split", "domain", "text"):
            if not isinstance(r.get(key), str) or not r[key].strip():
                raise ValueError(f"missing/empty {key}: {r}")
        if r["id"] in ids or r["text"].strip().casefold() in texts:
            raise ValueError("duplicate parent id/text")
        ids.add(r["id"]); texts.add(r["text"].strip().casefold())
        if r["split"] not in {"dev", "test", "ood"}:
            raise ValueError("split must be dev/test/ood")
        group_splits.setdefault(r["group_id"], r["split"])
        if group_splits[r["group_id"]] != r["split"]:
            raise ValueError("one group spans multiple splits")
    limit = cfg.get("parents_per_split")
    if limit:
        # Stable subset, not the first N rows which may be one topic only.
        selected = []
        for split in ("dev", "test", "ood"):
            sub = [r for r in rows if r["split"] == split]
            sub.sort(key=lambda r: stable_seed(cfg["seed"], r["id"]))
            selected.extend(sub[:limit])
        rows = selected
    if not all(any(r["split"] == s for r in rows) for s in ("dev", "test", "ood")):
        raise ValueError("all three splits must be present")
    return rows


def unit(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def softmax(x):
    x = np.asarray(x, dtype=float)
    z = np.exp(x - x.max())
    return z / z.sum()
