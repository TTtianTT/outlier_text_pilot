"""Framed experimental channel; optional standard AES-GCM wraps the payload.

No retries after a failed target, no target-dependent public search counter,
no original message, candidate index, codeword labels or cached vectors in packet.
"""
from __future__ import annotations
import os
import secrets
import time
from pathlib import Path
import numpy as np
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from .io import read_json, read_jsonl, write_json, canonical, digest, file_hash, softmax
from .readout import planes, read, bits_to_symbols, symbols_to_bytes


def keygen(path):
    # Never overwrite an existing key and never print it.
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as f: f.write(secrets.token_bytes(32))


def load_key(path):
    key = Path(path).read_bytes()
    if len(key) != 32: raise ValueError("key file must contain exactly 32 raw bytes")
    return key


def derive(master, nonce, purpose):
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=bytes.fromhex(nonce),
                info=("otc-v1/" + purpose).encode()).derive(master)


def load_profile(path):
    path = Path(path)
    meta = read_json(path / "profile.json")
    if meta["version"] != 1 or meta["calibration_split"] != "dev":
        raise ValueError("unsupported or invalid profile")
    if file_hash(path / "calibration.npy") != meta["calibration_sha256"]:
        raise ValueError("profile calibration checksum mismatch")
    cal = np.load(path / "calibration.npy", allow_pickle=False)
    if cal.ndim != 2 or cal.shape != (meta["calibration_count"], meta["feature_dimension"]) or not np.isfinite(cal).all():
        raise ValueError("invalid calibration matrix")
    return meta, cal


def backend_for(meta):
    signature = meta["feature_signature"]
    if signature["backend"] == "synthetic":
        from .synthetic import SyntheticBackend
        backend = SyntheticBackend(signature["dimension"])
    elif signature["backend"] == "hf":
        from .hf import CausalBackend
        backend = CausalBackend(signature["spec"])
    else:
        raise ValueError("unsupported feature backend")
    if backend.signature != signature:
        backend.close()
        raise ValueError("decoder model/tokenizer/dtype/runtime differs from feature profile")
    return backend


def encode(run, key_path, input_path, output_path, method="outlier", bits=1,
           margin=0.0001, mode="raw", split="test", nonce=None):
    if method not in {"outlier", "matched", "random", "nearest"} or not 1 <= bits <= 8:
        raise ValueError("invalid method/bits")
    if margin < 0 or not np.isfinite(margin): raise ValueError("invalid margin")
    if mode not in {"raw", "aead"}: raise ValueError("mode must be raw/aead")
    if split not in {"test", "ood"}: raise ValueError("encode uses test or ood carriers")
    output_path = Path(output_path)
    if output_path.exists() or output_path.with_suffix(output_path.suffix + ".diagnostics.json").exists():
        raise ValueError("choose a new output path; never leave a stale successful packet after failure")
    run = Path(run); meta, calibration = load_profile(run / "profile")
    master = load_key(key_path)
    # nonce argument is for deterministic unit tests, not exposed by the CLI.
    nonce = nonce or secrets.token_hex(16)
    if len(bytes.fromhex(nonce)) != 16: raise ValueError("nonce must be 16 bytes")
    payload = Path(input_path).read_bytes()
    coding_key = derive(master, nonce, "coding")
    header = {"version": 1, "profile_id": digest(meta), "nonce": nonce,
              "bits": bits, "margin": margin, "mode": mode,
              "payload_bytes": len(payload) + (16 if mode == "aead" else 0)}
    if mode == "aead":
        # Independent random 96-bit AEAD nonce, included in authenticated header.
        header["aead_nonce"] = secrets.token_hex(12)
        payload = AESGCM(derive(master, nonce, "aead")).encrypt(
            bytes.fromhex(header["aead_nonce"]), payload, canonical(header).encode())
    targets = bits_to_symbols(payload, bits)
    selections = read_json(run / "selections.json")
    rows = read_jsonl(run / "features.jsonl")
    z = np.load(run / "vectors.npz", allow_pickle=False)["vectors"]
    parents = sorted([p for p in read_jsonl(run / "parents.jsonl") if p["split"] == split],
                     key=lambda p: digest(["public-parent-order-v1", nonce, p["id"]]))
    if not parents: raise ValueError("no parents for requested split")
    blocks, token_count = [], 0
    rng = np.random.default_rng(secrets.randbits(128))
    t0 = time.perf_counter()
    failed = None
    for block, target in enumerate(targets):
        # Public nonce chooses an order independent of bits; cycle without skipping.
        p = parents[block % len(parents)]
        groups = selections[p["id"]]
        if groups is None:
            failed = {"block": block, "reason": "candidate_selection_failed"}; break
        inds = groups[method]
        r, tau = planes(coding_key, nonce, block, bits, calibration)
        labels, margins = read(z[inds], r, tau)
        choices = np.flatnonzero((labels == int(target)) & (margins >= margin))
        if not len(choices):
            failed = {"block": block, "reason": "target_symbol_unavailable"}; break
        log_weights = -np.array([rows[inds[i]]["nll_sum"] for i in choices])
        chosen = inds[int(rng.choice(choices, p=softmax(log_weights)))]
        blocks.append(rows[chosen]["text"])
        token_count += rows[chosen]["n_tokens"]
    diagnostics = {"status": "failed" if failed else "encoded_not_yet_receiver_verified",
                   "backend": meta["feature_signature"]["backend"], "method": method,
                   "message_bits": Path(input_path).stat().st_size * 8,
                   "channel_bits_with_aead": len(payload) * 8,
                   "required_blocks": len(targets), "constructed_blocks": len(blocks),
                   "failure": failed, "offline_selection_seconds": time.perf_counter() - t0,
                   "candidate_feature_cache_reused": True,
                   "note": "Selection timing excludes cached generation/features. No ECC. No edits/resynchronization guarantee."}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if failed:
        diagnostics.update({"transmitted_tokens": 0, "wire_bytes": 0,
                            "aborted_before_transmission": True, "recovered_message_bits": 0})
    else:
        packet = {"header": header, "blocks": blocks}
        wire = canonical(packet).encode()
        # Count complete serialized framing with exactly the same tokenizer.
        if meta["feature_signature"]["backend"] == "hf":
            from transformers import AutoTokenizer
            import hashlib
            signature = meta["feature_signature"]
            tokenizer = AutoTokenizer.from_pretrained(signature["spec"]["name"], revision=signature["spec"]["revision"], trust_remote_code=False)
            tok_string = tokenizer.backend_tokenizer.to_str() if tokenizer.is_fast else repr(tokenizer.get_vocab())
            if hashlib.sha256(tok_string.encode()).hexdigest() != signature["provenance"]["tokenizer_sha256"]:
                raise ValueError("wire tokenizer does not match profile")
            wire_tokens = len(tokenizer.encode(wire.decode(), add_special_tokens=False))
        else:
            wire_tokens = None  # Synthetic whitespace counts are NOT tokenizer metrics.
        output_path.write_bytes(wire)
        diagnostics.update({"carrier_tokens": token_count, "wire_bytes": len(wire),
                            "wire_tokens": wire_tokens, "aborted_before_transmission": False,
                            "wire_bits_per_message_bit": len(wire) * 8 / max(1, diagnostics["message_bits"]),
                            "candidate_message_bits_per_wire_token": diagnostics["message_bits"] / wire_tokens if wire_tokens else None})
    write_json(output_path.with_suffix(output_path.suffix + ".diagnostics.json"), diagnostics)
    return diagnostics


def decode(profile, key_path, packet_path, output_path, backend=None):
    meta, calibration = load_profile(profile)
    packet = read_json(packet_path)
    if set(packet) != {"header", "blocks"}: raise ValueError("unexpected packet fields")
    h = packet["header"]
    expected = {"version", "profile_id", "nonce", "bits", "margin", "mode", "payload_bytes"}
    if h.get("mode") == "aead": expected.add("aead_nonce")
    if set(h) != expected or h.get("version") != 1: raise ValueError("unsupported frame")
    if h["profile_id"] != digest(meta): raise ValueError("wrong public profile")
    if h["mode"] not in {"raw", "aead"}: raise ValueError("invalid mode")
    if not isinstance(h["bits"], int) or not 1 <= h["bits"] <= 8: raise ValueError("invalid bits")
    if not isinstance(h["payload_bytes"], int) or not 0 <= h["payload_bytes"] <= 1000000: raise ValueError("invalid payload length")
    if not isinstance(h["margin"], (int, float)) or not 0 <= h["margin"] <= 10: raise ValueError("invalid margin")
    if len(bytes.fromhex(h["nonce"])) != 16: raise ValueError("invalid nonce")
    if not isinstance(packet["blocks"], list) or len(packet["blocks"]) != (8 * h["payload_bytes"] + h["bits"] - 1) // h["bits"]:
        raise ValueError("missing/extra blocks; edits and resegmentation are unsupported")
    if any(not isinstance(s, str) or len(s) > 10000 for s in packet["blocks"]): raise ValueError("invalid carrier")
    master = load_key(key_path); coding_key = derive(master, h["nonce"], "coding")
    owns_backend = backend is None
    backend = backend_for(meta) if owns_backend else backend
    if backend.signature != meta["feature_signature"]:
        raise ValueError("receiver backend does not match public profile")
    symbols, minimum_margin = [], float("inf")
    t0 = time.perf_counter()
    try:
        for block, text in enumerate(packet["blocks"]):
            v, _ = backend.feature(text)  # Recompute from RECEIVED TEXT, never load sender cache.
            r, tau = planes(coding_key, h["nonce"], block, h["bits"], calibration)
            label, margin = read(v, r, tau)
            minimum_margin = min(minimum_margin, float(margin[0]))
            if margin[0] < h["margin"]:
                raise ValueError("received carrier lies within margin guard; refusing an uncertain decode")
            symbols.append(int(label[0]))
        recovered = symbols_to_bytes(symbols, h["bits"], h["payload_bytes"])
        if h["mode"] == "aead":
            recovered = AESGCM(derive(master, h["nonce"], "aead")).decrypt(
                bytes.fromhex(h["aead_nonce"]), recovered, canonical(h).encode())
        output_path = Path(output_path)
        if output_path.exists(): raise ValueError("refusing to overwrite an existing recovered file")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(recovered)
        diagnostics = {"status": "authenticated" if h["mode"] == "aead" else "decoded_raw_unverified",
                       "bytes": len(recovered), "decoder_seconds": time.perf_counter() - t0,
                       "minimum_received_margin": minimum_margin if symbols else None,
                       "sender_cache_used": False}
        write_json(output_path.with_suffix(output_path.suffix + ".diagnostics.json"), diagnostics)
        return diagnostics
    finally:
        if owns_backend: backend.close()
