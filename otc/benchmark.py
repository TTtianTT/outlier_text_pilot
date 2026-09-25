"""Raw uniformly random payload benchmark with fresh receiver forwards.

Same payload/key/nonce/schedule across methods in each trial. No retries.
This benchmarks the coding layer, not confidentiality. Trial keys stay local.
"""
from pathlib import Path
import secrets
import time
import numpy as np
from .codec import encode, decode, keygen, load_profile, backend_for
from .verification import check_payload
from .io import write_json
from .evaluate import csv_write, METHODS


def benchmark(run, output, trials=5, payload_bytes=1, bits=1, margin=.0001, split="test"):
    if not 1 <= trials <= 10000 or not 1 <= payload_bytes <= 10000:
        raise ValueError("invalid trial count or payload size")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    key = output / "private_trials.key"; keygen(key)
    profile = Path(run) / "profile"; meta, _ = load_profile(profile)
    backend = backend_for(meta)
    records = []
    try:
        for trial in range(trials):
            payload = output / f"payload-{trial:04d}.bin"
            payload.write_bytes(secrets.token_bytes(payload_bytes))
            nonce = secrets.token_hex(16)
            for method in METHODS:
                packet = output / f"{method}-{trial:04d}.packet.json"
                recovered = output / f"{method}-{trial:04d}.recovered.bin"
                t0 = time.perf_counter()
                sender = encode(run, key, payload, packet, method=method, bits=bits,
                                margin=margin, mode="raw", split=split, nonce=nonce)
                row = {"trial": trial, "method": method, "split": split, "payload_bytes": payload_bytes,
                       "bits_per_sentence": bits, "margin": margin,
                       "encoding_success": sender["status"] != "failed", "byte_exact": False,
                       "ber_accepted": None, "wire_tokens": sender.get("wire_tokens"),
                       "wire_bytes": sender.get("wire_bytes", 0), "recovered_message_bits": 0,
                       "seconds": 0.0, "failure": sender.get("failure")}
                if row["encoding_success"]:
                    try:
                        decode(profile, key, packet, recovered, backend=backend)
                        check = check_payload(payload, recovered, packet)
                        row["byte_exact"] = check["byte_exact"]
                        row["ber_accepted"] = check["ber"]
                        row["recovered_message_bits"] = payload_bytes * 8 if check["byte_exact"] else 0
                    except ValueError as exc:
                        row["failure"] = str(exc)
                row["seconds"] = time.perf_counter() - t0
                records.append(row)
            print(f"codec trial {trial+1}/{trials}", flush=True)
    finally:
        backend.close()
    csv_write(output / "trials.csv", records)
    summary = []
    for method in METHODS:
        rr = [r for r in records if r["method"] == method]
        wire_tokens = sum(r["wire_tokens"] or 0 for r in rr)
        wire_bytes = sum(r["wire_bytes"] for r in rr)
        recovered_bits = sum(r["recovered_message_bits"] for r in rr)
        summary.append({"method": method, "messages_attempted": len(rr),
                        "messages_encoded": sum(r["encoding_success"] for r in rr),
                        "messages_recovered": sum(r["byte_exact"] for r in rr),
                        "message_recovery_rate": sum(r["byte_exact"] for r in rr) / len(rr),
                        "total_transmitted_wire_tokens": wire_tokens if wire_tokens else None,
                        "net_recovered_bits_per_wire_token": recovered_bits / wire_tokens if wire_tokens else None,
                        "net_recovered_bits_per_wire_bit": recovered_bits / (8 * wire_bytes) if wire_bytes else None,
                        "runtime_seconds": sum(r["seconds"] for r in rr)})
    write_json(output / "summary.json", {"backend": meta["feature_signature"]["backend"],
               "scope": "raw random payloads, fresh receiver features, no ECC, no retries; shared preprocessing excluded",
               "note": "Always report recovery rate alongside net rate; pre-transmission aborts send zero tokens but consume compute.",
               "results": summary})
    return summary
