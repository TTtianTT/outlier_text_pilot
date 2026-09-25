import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np
from cryptography.exceptions import InvalidTag
from otc.io import write_json, write_jsonl, file_hash, unit, digest
from otc.readout import planes, read, bits_to_symbols, symbols_to_bytes
from otc.selection import ResidualScorer, select_groups, CONFOUNDS
from otc.evaluate import pool_distribution, two_way_ci
from otc.synthetic import SyntheticBackend
from otc.codec import keygen, encode, decode


class ReadoutTests(unittest.TestCase):
    def test_bit_packing_all_widths(self):
        for payload in (b"", b"\x00\xff\xa7", bytes(range(256))):
            for width in range(1, 9):
                self.assertEqual(payload, symbols_to_bytes(bits_to_symbols(payload, width), width, len(payload)))

    def test_nonzero_padding_is_rejected(self):
        symbols = bits_to_symbols(b"a", 3); symbols[-1] |= 1
        with self.assertRaises(ValueError): symbols_to_bytes(symbols, 3, 1)

    def test_key_nonce_block_separation_and_margin(self):
        rng = np.random.default_rng(7); cal = unit(rng.normal(size=(20, 128)))
        a, t = planes(b"a" * 32, "n1", 0, 3, cal)
        b, _ = planes(b"a" * 32, "n1", 0, 3, cal)
        np.testing.assert_array_equal(a, b)
        for key, nonce, block in ((b"b"*32, "n1", 0), (b"a"*32, "n2", 0), (b"a"*32, "n1", 1)):
            other, _ = planes(key, nonce, block, 3, cal)
            self.assertFalse(np.array_equal(a, other))
        x = cal[0]; label, margin = read(x, a, t)
        perturb = unit(rng.normal(size=128)) * margin[0] * .99
        np.testing.assert_array_equal(label, read(x + perturb, a, t)[0])

    def test_finite_pool_distribution_and_aborts(self):
        rows = [{"nll_sum": 0.}, {"nll_sum": 0.}]
        _, _, abort, tv = pool_distribution(np.array([0, 1]), np.ones(2), rows, 1, 0.)
        self.assertEqual(abort, 0); self.assertAlmostEqual(tv, 0)
        _, _, abort, tv = pool_distribution(np.array([0, 0]), np.ones(2), rows, 1, 0.)
        self.assertEqual(abort, .5); self.assertAlmostEqual(tv, 0)


class SelectionTests(unittest.TestCase):
    def rows(self, n=24):
        return [{"candidate_id": str(i), "split": "dev", "behavior_dist": .1 + i / 100,
                 **{c: 1.0 for c in CONFOUNDS}} for i in range(n)]

    def cfg(self):
        return {"seed": 1, "selection": {"keep": 4, "outlier_fraction": .25,
                                         "calipers": {c: .1 for c in CONFOUNDS}}}

    def test_test_data_cannot_fit_nuisance_model(self):
        rows = self.rows(); rows[0]["split"] = "test"
        with self.assertRaises(ValueError): ResidualScorer().fit(rows)

    def test_matching_is_equal_budget_and_can_fail(self):
        rows = self.rows(); scores = np.arange(len(rows))
        groups, d = select_groups(rows, scores, self.cfg(), "p")
        self.assertEqual(d["reason"], "ok")
        self.assertTrue(all(len(v) == 4 for v in groups.values()))
        self.assertFalse(set(groups["outlier"]) & set(groups["matched"]))
        for r in rows[-6:]: r["sem_dist"] = 4.
        groups, d = select_groups(rows, scores, self.cfg(), "p")
        self.assertIsNone(groups); self.assertEqual(d["reason"], "caliper_match_failed")

    def test_bootstrap_preserves_parent_pairing(self):
        records = []
        for group in ("a", "b", "c"):
            for key in (0, 1, 2):
                for method, value in (("outlier", .8), ("matched", .5)):
                    records.append(dict(group_id=group, parent_id=group, key_seed=key, method=method, coverage=value))
        ci = two_way_ci(records, "coverage", "outlier", "matched", 100, 0)
        for field in ("delta", "ci_low", "ci_high"): self.assertAlmostEqual(ci[field], .3)


class CodecTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.run = self.root / "sender"; self.run.mkdir(); self.profile = self.run / "profile"; self.profile.mkdir()
        self.backend = SyntheticBackend()
        cal = np.array([self.backend.feature(f"dev reference {i}")[0] for i in range(64)])
        np.save(self.profile / "calibration.npy", cal)
        self.meta = {"version": 1, "feature_signature": self.backend.signature, "calibration_split": "dev",
                     "calibration_count": len(cal), "feature_dimension": cal.shape[1],
                     "calibration_sha256": file_hash(self.profile / "calibration.npy")}
        write_json(self.profile / "profile.json", self.meta)
        rows, vectors = [], []
        # Wide test-only pool makes the independent decoder contract test reliable.
        # This fixture is not a research comparison and never reports efficacy.
        for i in range(256):
            text = f"Fixture carrier variation {i}"
            v, stats = self.backend.feature(text)
            rows.append({"text": text, "vector_index": i, **stats}); vectors.append(v)
        write_jsonl(self.run / "features.jsonl", rows)
        write_jsonl(self.run / "parents.jsonl", [{"id": "p", "split": "test"}])
        write_json(self.run / "selections.json", {"p": {m: list(range(256)) for m in ("outlier", "matched", "random", "nearest")}})
        np.savez_compressed(self.run / "vectors.npz", vectors=np.array(vectors), calibration=cal)
        self.key = self.root / "key"; keygen(self.key)
        self.payload = self.root / "payload"; self.payload.write_bytes(b"\x00\xffHello\x00")

    def tearDown(self): self.temp.cleanup()

    def test_raw_decode_in_isolated_receiver_without_sender_cache(self):
        packet = self.root / "packet.json"
        result = encode(self.run, self.key, self.payload, packet, bits=3, margin=0., nonce="01"*16)
        self.assertNotEqual(result["status"], "failed")
        self.assertEqual(set(json.loads(packet.read_text())), {"header", "blocks"})
        receiver_profile = self.root / "receiver-profile"; shutil.copytree(self.profile, receiver_profile)
        shutil.rmtree(self.run)
        out = self.root / "recovered"
        decode(receiver_profile, self.key, packet, out)
        self.assertEqual(out.read_bytes(), self.payload.read_bytes())

    def test_aead_and_wrong_key(self):
        packet = self.root / "aead.json"
        result = encode(self.run, self.key, self.payload, packet, bits=2, margin=0., mode="aead", nonce="02"*16)
        self.assertNotEqual(result["status"], "failed")
        out = self.root / "recovered"
        decode(self.profile, self.key, packet, out)
        self.assertEqual(out.read_bytes(), self.payload.read_bytes())
        wrong = self.root / "wrong"; keygen(wrong)
        with self.assertRaises((InvalidTag, ValueError)):
            decode(self.profile, wrong, packet, self.root / "wrong-result")
        self.assertFalse((self.root / "wrong-result").exists())

    def test_injected_target_field_and_missing_block_rejected(self):
        packet = self.root / "p.json"
        encode(self.run, self.key, self.payload, packet, bits=1, margin=0., nonce="03"*16)
        body = json.loads(packet.read_text()); body["target"] = "1111"; write_json(packet, body)
        with self.assertRaises(ValueError): decode(self.profile, self.key, packet, self.root / "out")
        del body["target"]; body["blocks"].pop(); write_json(packet, body)
        with self.assertRaises(ValueError): decode(self.profile, self.key, packet, self.root / "out")

    def test_selection_failure_writes_no_successful_packet(self):
        write_json(self.run / "selections.json", {"p": None})
        packet = self.root / "failed.json"
        result = encode(self.run, self.key, self.payload, packet)
        self.assertEqual(result["status"], "failed")
        self.assertFalse(packet.exists())
        self.assertEqual(result["transmitted_tokens"], 0)

    def test_profile_tampering_rejected(self):
        (self.profile / "calibration.npy").write_bytes(b"broken")
        with self.assertRaises(ValueError): encode(self.run, self.key, self.payload, self.root / "p.json")

    def test_keygen_refuses_overwrite(self):
        before = self.key.read_bytes()
        with self.assertRaises(FileExistsError): keygen(self.key)
        self.assertEqual(before, self.key.read_bytes())


if __name__ == "__main__": unittest.main()
