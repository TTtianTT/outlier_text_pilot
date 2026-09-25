from __future__ import annotations
import argparse
import json
from pathlib import Path
from .io import config, read_json, write_json, file_hash


def run_config(run):
    return read_json(Path(run) / "manifest.json")["config"]


def main():
    p = argparse.ArgumentParser(description="Outlier-assisted text coding pilot; use README.md first.")
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("generate", "features", "all"):
        s = sub.add_parser(name); s.add_argument("--config", required=True); s.add_argument("--run", required=True)
    s = sub.add_parser("evaluate"); s.add_argument("--run", required=True)
    s = sub.add_parser("demo"); s.add_argument("--run", default="runs/demo"); s.add_argument("--config", default="configs/demo.yaml")
    s = sub.add_parser("keygen"); s.add_argument("--out", required=True)
    s = sub.add_parser("random-payload"); s.add_argument("--out", required=True); s.add_argument("--bytes", type=int, default=1)
    s = sub.add_parser("encode")
    for name in ("run", "key", "input", "out"): s.add_argument("--" + name, required=True)
    s.add_argument("--method", choices=["outlier", "matched", "random", "nearest"], default="outlier")
    s.add_argument("--bits", type=int, default=1); s.add_argument("--margin", type=float, default=.0001)
    s.add_argument("--mode", choices=["raw", "aead"], default="raw"); s.add_argument("--split", choices=["test", "ood"], default="test")
    s = sub.add_parser("decode")
    for name in ("profile", "key", "packet", "out"): s.add_argument("--" + name, required=True)
    s = sub.add_parser("check"); s.add_argument("--original", required=True); s.add_argument("--recovered", required=True)
    s.add_argument("--packet", help="optional: calculate verified net wire rate from sender diagnostics")
    s = sub.add_parser("verify-readout"); s.add_argument("--run", required=True); s.add_argument("--limit", type=int, default=100)
    s = sub.add_parser("plot"); s.add_argument("--run", required=True)
    s = sub.add_parser("codec-bench")
    s.add_argument("--run", required=True); s.add_argument("--out", required=True)
    s.add_argument("--trials", type=int, default=5); s.add_argument("--bytes", type=int, default=1)
    s.add_argument("--bits", type=int, default=1); s.add_argument("--margin", type=float, default=.0001)
    s.add_argument("--split", choices=["test", "ood"], default="test")
    a = p.parse_args()
    try:
        if a.command in {"generate", "features", "all"}:
            from .pipeline import generate, features
            cfg = config(a.config)
            if a.command in {"generate", "all"}: generate(cfg, a.run)
            if a.command in {"features", "all"}: features(cfg, a.run)
            if a.command == "all":
                from .evaluate import evaluate
                evaluate(cfg, a.run)
        elif a.command == "evaluate":
            from .evaluate import evaluate
            evaluate(run_config(a.run), a.run)
        elif a.command == "demo":
            from .synthetic import make_demo
            from .evaluate import evaluate
            cfg = config(a.config); make_demo(cfg, a.run); evaluate(cfg, a.run)
        elif a.command == "keygen":
            from .codec import keygen
            keygen(a.out); print("Key saved; keep it separate from packets and experiment uploads.")
        elif a.command == "random-payload":
            import secrets
            if not 0 < a.bytes <= 1000000: raise ValueError("bytes must be 1..1000000")
            with open(a.out, "xb") as f: f.write(secrets.token_bytes(a.bytes))
            print(f"created {a.bytes} random bytes")
        elif a.command == "encode":
            from .codec import encode
            result = encode(a.run, a.key, a.input, a.out, a.method, a.bits, a.margin, a.mode, a.split)
            print(json.dumps(result, indent=2))
            if result["status"] == "failed": raise SystemExit(2)
        elif a.command == "decode":
            from .codec import decode
            print(json.dumps(decode(a.profile, a.key, a.packet, a.out), indent=2))
        elif a.command == "check":
            from .verification import check_payload
            result = check_payload(a.original, a.recovered, a.packet)
            print(json.dumps(result, indent=2))
            if not result["byte_exact"]: raise SystemExit(3)
        elif a.command == "verify-readout":
            from .verification import verify_readout
            result = verify_readout(a.run, a.limit)
            print(json.dumps(result, indent=2))
            if result["status"] != "passed": raise SystemExit(3)
        elif a.command == "plot":
            from .plotting import plot
            plot(a.run)
        elif a.command == "codec-bench":
            from .benchmark import benchmark
            print(json.dumps(benchmark(a.run, a.out, a.trials, a.bytes, a.bits, a.margin, a.split), indent=2))
    except (ValueError, FileNotFoundError, ImportError) as exc:
        p.exit(1, f"{type(exc).__name__}: {exc}\n")


if __name__ == "__main__": main()
