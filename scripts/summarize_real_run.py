"""Summarize exclusions and primary results without changing the protocol."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run
    cfg = json.loads((run / "manifest.json").read_text())["config"]
    rows = read_lines(run / "features.jsonl")
    diagnostics = read_lines(run / "selection_diagnostics.jsonl")
    generation = Counter()
    for shard in sorted((run / "candidate_shards").glob("*.json")):
        generation.update(r["generation_status"] for r in json.loads(shard.read_text())["attempts"])
    f = cfg["filter"]
    splits = {}
    for split in ("dev", "test", "ood"):
        rr = [r for r in rows if r["split"] == split]
        dd = [d for d in diagnostics if d["split"] == split]
        failures = Counter()
        for r in rr:
            checks = {
                "cosine": r["sem_dist"] > 1 - f["min_cosine"],
                "bidirectional_nli": r["nli_min"] < f["min_entailment"],
                "token_length": not f["min_tokens"] <= r["n_tokens"] <= f["max_tokens"],
                "edit_ratio": r["edit_ratio"] > f["max_edit_ratio"],
                "ascii": not r["ascii_printable"],
            }
            failures.update(k for k, failed in checks.items() if failed)
        splits[split] = {
            "parents": len(dd), "feature_candidates": len(rr),
            "eligible_candidates": sum(d["eligible"] for d in dd),
            "matched_parents": sum(d["reason"] == "ok" for d in dd),
            "selection_reasons": dict(Counter(d["reason"] for d in dd)),
            "filter_failure_counts_nonexclusive": dict(failures),
        }
    primary = {}
    for name in ("summary", "paired_deltas"):
        with (run / f"{name}.csv").open() as handle:
            primary[name] = [r for r in csv.DictReader(handle)
                             if int(r["bits"]) == cfg["evaluation"]["primary_bits"]
                             and float(r["margin"]) == cfg["evaluation"]["primary_margin"]]
    result = {"generation_status": dict(generation), "splits": splits, "primary": primary,
              "readout": json.loads((run / "readout_verification.json").read_text())}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    (run / "execution_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
