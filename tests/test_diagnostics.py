import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
from otc.diagnostics import independent_groups, symbol_metrics
from otc.evaluate import two_way_ci
from otc.io import write_json, write_jsonl
from otc.verification import verify_readout


class DiagnosticTests(unittest.TestCase):
    def test_baselines_available_without_enough_candidates_for_pairs(self):
        rows = [{"candidate_id": str(i), "sem_dist": d} for i, d in enumerate([.1, .02, .04])]
        groups = independent_groups(rows, {"seed": 1, "selection": {"keep": 2}}, "p")
        self.assertEqual(groups["full_pool"], [0, 1, 2])
        self.assertEqual(groups["nearest"], [1, 2])
        self.assertEqual(len(set(groups["random"])), 2)
        small = independent_groups(rows[:1], {"seed": 1, "selection": {"keep": 2}}, "p")
        self.assertEqual(small, {"full_pool": [0], "random": [], "nearest": []})

    def test_full_pool_collapse_and_guard_rejection_are_distinct(self):
        labels, margins = np.array([0, 1]), np.array([.2, .00001])
        self.assertEqual(symbol_metrics(labels, margins, 0)["full_coverage"], 1)
        guarded = symbol_metrics(labels, margins, .0001)
        self.assertEqual(guarded["coverage"], .5)
        self.assertEqual(guarded["single_region"], 1)
        self.assertEqual(symbol_metrics(labels, margins, 1)["no_accepted_symbols"], 1)

    def test_no_comparable_samples_yields_na_effect_not_zero(self):
        result = two_way_ci([], "coverage", "outlier", "matched", 100, 0)
        self.assertIsNone(result["delta"])
        self.assertIsNone(result["ci_low"])
        self.assertIsNone(result["ci_high"])

    def test_eligible_readout_ignores_matching_and_other_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            cfg = {"seed": 1, "evaluation": {"bits": [1], "primary_margin": 0., "key_seeds": [0, 1]},
                   "filter": {"min_cosine": .85, "min_entailment": .8, "min_tokens": 5,
                              "max_tokens": 80, "max_edit_ratio": .7}}
            base = {"sem_dist": .1, "nli_min": .9, "n_tokens": 10, "edit_ratio": .2,
                    "ascii_printable": True, "split": "dev", "parent_id": "p", "text": "accepted"}
            rows = [{**base, "candidate_id": "a"}, {**base, "candidate_id": "b", "nli_min": .1},
                    {**base, "candidate_id": "c", "split": "test"}]
            write_json(run / "manifest.json", {"config": cfg})
            write_jsonl(run / "features.jsonl", rows)
            vec = np.array([1., 0.])
            cal = np.array([[0., 1.], [0., -1.]])
            np.savez(run / "vectors.npz", vectors=np.array([vec, vec, vec]))
            backend = Mock(); backend.feature.return_value = (vec, {})
            output = run / "independent.json"
            with patch("otc.verification.load_profile", return_value=({"feature_signature": {"backend": "synthetic"}}, cal)), \
                 patch("otc.verification.backend_for", return_value=backend):
                result = verify_readout(run, 100, scope="eligible", split="dev", bits=1, output=output)
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["fresh_text_forwards"], 1)
            self.assertEqual(result["bits_checked"], 2)
            backend.feature.assert_called_once_with("accepted")
            backend.close.assert_called_once()
            self.assertFalse((run / "readout_verification.json").exists())
            self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()
