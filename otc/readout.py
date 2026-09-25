"""Stateless keyed readout. Keys in research evaluation are PUBLIC seeds.

Rademacher directions come from HMAC-SHA256 + SHAKE256; no Python hash(),
no key-dependent model loading, no access to the sender's reference sentence.
"""
from __future__ import annotations
import hashlib
import hmac
import numpy as np
from .io import canonical


def planes(key: bytes, nonce: str, block: int, bits: int, calibration: np.ndarray):
    if len(key) != 32:
        raise ValueError("coding key must be 32 bytes")
    if block < 0 or not 1 <= bits <= 8 or len(calibration) < 2:
        raise ValueError("invalid block/bits/calibration")
    d = calibration.shape[1]
    directions = []
    for j in range(bits):
        message = canonical(["otc-readout-v1", nonce, block, j]).encode()
        seed = hmac.new(key, message, hashlib.sha256).digest()
        raw = hashlib.shake_256(seed).digest((d + 7) // 8)
        signs = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))[:d].astype(float)
        directions.append((2 * signs - 1) / np.sqrt(d))
    r = np.asarray(directions)
    thresholds = np.median(np.asarray(calibration, dtype=float) @ r.T, axis=0)
    return r, thresholds


def read(features, r, thresholds):
    features = np.atleast_2d(features).astype(np.float64)
    logits = features @ r.T - thresholds
    bits = logits >= 0
    powers = 1 << np.arange(r.shape[0] - 1, -1, -1)
    symbols = bits.astype(np.int64) @ powers
    # Euclidean feature-space margin; each direction has norm one.
    margins = np.min(np.abs(logits), axis=1)
    return symbols, margins


def bits_to_symbols(data: bytes, width: int):
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    padding = (-len(bits)) % width
    bits = np.pad(bits, (0, padding)).reshape(-1, width)
    return bits @ (1 << np.arange(width - 1, -1, -1))


def symbols_to_bytes(symbols, width: int, n_bytes: int):
    s = np.asarray(symbols, dtype=np.int64)
    if np.any(s < 0) or np.any(s >= 2**width):
        raise ValueError("invalid symbol")
    bits = ((s[:, None] >> np.arange(width - 1, -1, -1)) & 1).ravel()
    if n_bytes < 0 or n_bytes * 8 > len(bits):
        raise ValueError("incomplete frame")
    if np.any(bits[n_bytes * 8:]):
        raise ValueError("nonzero frame padding")
    return np.packbits(bits[:n_bytes * 8]).tobytes()


def public_eval_key(seed):
    return hashlib.sha256(f"PUBLIC-evaluation-key:{seed}".encode()).digest()
