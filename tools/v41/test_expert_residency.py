#!/usr/bin/env python3
import contextlib
from decimal import Decimal
import io
import json
import os
from pathlib import Path
import unittest

import expert_residency as utility


class ResidencyTests(unittest.TestCase):
    def setUp(self):
        self.paths = []
        self.metadata = (
            "model abc 4096\ngeometry 2 3 1 256 256\n" +
            "".join(f"expert {l} {e} {l} {257 + e * 256} {512 + e * 256} 10 10 11\n"
                    for l in range(2) for e in range(3)))

    def tearDown(self):
        for path in self.paths:
            path.unlink(missing_ok=True)

    def path(self, label):
        path = Path(f".residency-test-{os.getpid()}-{label}")
        self.paths.append(path)
        return path

    def profile(self, label="train", decode=(98, 1, 1, 98, 1, 1), prefill=None):
        path = self.path(label)
        prefill = prefill or [0] * 6
        path.write_text("DS41_ROUTES_V1\n" + self.metadata + "".join(
            f"count {i // 3} {i % 3} {prefill[i]} {decode[i]} 1\n" for i in range(6)))
        return path

    def test_exact_target_ties_phases_and_physical_bytes(self):
        p = utility.load_profile(self.profile(prefill=(0, 0, 100, 0, 0, 100)))
        result, selected = utility.recommend(p, "decode", Decimal(".99"))
        self.assertEqual(selected, [(0, 0), (1, 0), (0, 1), (0, 2)])
        self.assertEqual(result["calibration"]["hits"], 198)
        self.assertEqual(result["cards"]["0"]["selected_physical_bytes"], 2304)
        self.assertEqual(result["cards"]["1"]["pinned_tail_blob_bytes"], 1282)
        _, prompt_set = utility.recommend(p, "prefill")
        self.assertEqual(prompt_set, [(0, 2), (1, 2)])
        with self.assertRaisesRegex(ValueError, "cannot fit"):
            utility.recommend(p, "decode", capacities={0: 512, 1: 512})

    def test_merge_heldout_separate_and_cli_failure(self):
        train = self.profile()
        held = self.profile("held", (0, 0, 100, 0, 0, 100))
        p = utility.load_profile(train)
        merged = utility.merge_profiles([p, p])
        self.assertEqual(utility.phase_counts(merged, "all")[0, 0], 196)
        result, selected = utility.recommend(p, "decode", ".98", held_out=utility.load_profile(held))
        self.assertEqual(selected, [(0, 0), (1, 0)])
        self.assertEqual(result["held_out"]["hit_rate"], 0)
        self.assertFalse(result["held_out"]["meets_target"])
        report, seed = self.path("report"), self.path("seed")
        with contextlib.redirect_stderr(io.StringIO()) as errors:
            status = utility.main([str(train), str(train), "--phase", "decode",
                                   "--target", ".98", "--held-out", str(held),
                                   "--output", str(report), "--seed-output", str(seed)])
        self.assertEqual(status, 3)
        self.assertIn("held-out coverage is below", errors.getvalue())
        self.assertEqual(json.loads(report.read_text())["calibration"]["routes"], 400)
        self.assertEqual(seed.read_text(), "DS41_SEED_V1\n" + self.metadata +
                         "select 0 0\nselect 1 0\n")

    def test_target_boundary_does_not_round_down(self):
        p = utility.load_profile(self.profile())
        target = Decimal(".980000000000000000000000000001")
        result, selected = utility.recommend(p, "decode", target, held_out=p)
        self.assertEqual(len(selected), 3)
        self.assertEqual(result["calibration"]["hits"], 197)
        self.assertTrue(result["held_out"]["meets_target"])
        different_phase = utility.load_profile(self.profile(
            "phase-boundary", prefill=(98, 2, 0, 98, 1, 1), decode=(98, 0, 2, 98, 0, 2)))
        result, _ = utility.recommend(different_phase, "decode", target)
        self.assertEqual(result["calibration_by_phase"]["prefill"]["hits"], 196)
        self.assertFalse(result["calibration_by_phase"]["prefill"]["meets_target"])

    def test_invalid_profiles_zero_overflow_and_incompatibility(self):
        valid = self.profile()
        original = valid.read_text()
        for malformed in (original.replace("count 0 0 0 98", "count 0 0 0 -1"),
                          original.replace("count 0 0 0 98", f"count 0 0 0 {1 << 64}"),
                          original.replace("257 512", "257 256"),
                          original.replace("geometry 2 3 1", "geometry 2 3 3"),
                          original.replace("expert 1 2 1 769 1024 10 10 11\n", ""),
                          original + "count 0 0 0 1 1\n",
                          original.replace("count 1 2 0 1 1\n", "")):
            valid.write_text(malformed)
            with self.assertRaises(ValueError):
                utility.load_profile(valid)
        valid.write_text(original)
        p = utility.load_profile(valid)
        for target in ("0", "-1", "1.001", "NaN", "Infinity"):
            with self.assertRaises(ValueError):
                utility.recommend(p, target=target)
        with self.assertRaisesRegex(ValueError, "zero"):
            utility.recommend(p, "prefill")
        for before, after in (("model abc", "model def"),
                              ("geometry 2 3 1", "geometry 2 3 2"),
                              ("expert 0 0 0", "expert 0 0 1"),
                              ("10 10 11", "11 11 11")):
            other = dict(p, metadata=p["metadata"].replace(before, after))
            with self.assertRaisesRegex(ValueError, "incompatible"):
                utility.merge_profiles([p, other])
            with self.assertRaisesRegex(ValueError, "incompatible"):
                utility.recommend(p, held_out=other)
        overflow = dict(p, counts={key: (utility.MAX_U64, 0) for key in p["counts"]})
        with self.assertRaisesRegex(ValueError, "overflow"):
            utility.merge_profiles([overflow, p])
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(utility.main([str(valid), "--output", str(valid)]), 2)
        with contextlib.redirect_stderr(io.StringIO()) as errors:
            self.assertEqual(utility.main([str(valid), "--held-out", str(valid.resolve())]), 2)
        self.assertIn("separate from calibration", errors.getvalue())
        self.assertEqual(valid.read_text(), original)


if __name__ == "__main__":
    unittest.main()
