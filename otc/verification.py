from __future__ import annotations
from pathlib import Path
import numpy as np
from .io import read_json, read_jsonl, write_json, stable_seed
from .codec import load_profile, backend_for
from .readout import read, planes, public_eval_key


def verify_readout(run, limit=100, *, scope="matched", split=None, bits=None, output=None):
    """Fresh receiver forwards; cached sender features are used only as a comparator."""
    if limit <= 0: raise ValueError("limit must be positive")
    run = Path(run)
    cfg = read_json(run / "manifest.json")["config"]
    meta, calibration = load_profile(run / "profile")
    rows = read_jsonl(run / "features.jsonl")
    vectors = np.load(run / "vectors.npz", allow_pickle=False)["vectors"]
    if scope not in {"matched", "eligible"} or split not in {None, "dev", "test", "ood"}:
        raise ValueError("invalid verification scope/split")
    splits = {split} if split else {"test", "ood"}
    if scope == "eligible":
        from .selection import eligible
        candidates = {i for i, row in enumerate(rows) if row["split"] in splits and eligible(row, cfg)}
    else:
        selections = read_json(run / "selections.json")
        candidates = {i for groups in selections.values() if groups for inds in groups.values() for i in inds
                      if rows[i]["split"] in splits}
    ids = sorted(candidates, key=lambda i: stable_seed(cfg["seed"], rows[i]["candidate_id"], "verify"))[:limit]
    width = max(cfg["evaluation"]["bits"]) if bits is None else bits
    if not 1 <= width <= 8: raise ValueError("bits must be 1..8")
    backend = backend_for(meta)
    total_bits = mismatched_bits = accepted_bits = accepted_mismatch = 0
    max_drift = 0.; ambiguous = 0
    guard = cfg["evaluation"]["primary_margin"]
    try:
        for i in ids:
            fresh, _ = backend.feature(rows[i]["text"])
            max_drift = max(max_drift, float(np.linalg.norm(fresh - vectors[i])))
            for key_seed in cfg["evaluation"]["key_seeds"]:
                r, tau = planes(public_eval_key(key_seed), "PUBLIC-pilot-v1", stable_seed(rows[i]["parent_id"]), width, calibration)
                old, old_m = read(vectors[i], r, tau); new, new_m = read(fresh, r, tau)
                errors = bin(int(old[0]) ^ int(new[0])).count("1")
                total_bits += width; mismatched_bits += errors
                if old_m[0] >= guard:
                    accepted_bits += width; accepted_mismatch += errors
                    ambiguous += int(new_m[0] < guard)
    finally:
        backend.close()
    status = ("no_candidates" if not ids else "no_guarded_bits" if not accepted_bits else
              "failed" if accepted_mismatch or ambiguous else "passed")
    result = {"status": status, "backend": meta["feature_signature"]["backend"], "scope": scope,
              "splits": sorted(splits), "bits_per_readout": width, "candidate_population": len(candidates),
              "parents_checked": len({rows[i]["parent_id"] for i in ids}), "fresh_text_forwards": len(ids),
              "bits_checked": total_bits, "bit_errors": mismatched_bits,
              "ber": mismatched_bits / total_bits if total_bits else None,
              "sender_guarded_bits": accepted_bits, "sender_guarded_bit_errors": accepted_mismatch,
              "receiver_guard_rejections": ambiguous, "max_feature_l2_drift": max_drift,
              "warning": "Same runtime only; this does not establish cross-device or edited-text reliability."}
    write_json(output or run / "readout_verification.json", result)
    return result


def check_payload(original, recovered, packet=None):
    a, b = Path(original).read_bytes(), Path(recovered).read_bytes()
    aa, bb = np.unpackbits(np.frombuffer(a, dtype=np.uint8)), np.unpackbits(np.frombuffer(b, dtype=np.uint8))
    length = max(len(aa), len(bb)); overlap = min(len(aa), len(bb))
    errors = int(np.sum(aa[:overlap] != bb[:overlap]) + abs(len(aa) - len(bb)))
    result = {"byte_exact": a == b, "original_bytes": len(a), "recovered_bytes": len(b),
              "bit_errors_including_length": errors, "ber": errors / length if length else 0.}
    if packet:
        diagnostic = read_json(str(packet) + ".diagnostics.json")
        wire_tokens = diagnostic.get("wire_tokens")
        result["verified_net_bits_per_wire_token"] = len(a) * 8 / wire_tokens if a == b and wire_tokens else (0. if wire_tokens else None)
        result["wire_bytes"] = Path(packet).stat().st_size
        result["verified_net_bits_per_wire_bit"] = (len(a) * 8 / (Path(packet).stat().st_size * 8)) if a == b and Path(packet).stat().st_size else 0.
    write_json(str(recovered) + ".check.json", result)
    return result
