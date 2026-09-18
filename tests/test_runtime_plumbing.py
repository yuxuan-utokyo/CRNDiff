"""Regression tests for runtime plumbing; no scientific algorithms are replaced."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from scripts.reproduce._array_compare import compare_npz

ROOT = Path(__file__).resolve().parents[1]


class ExtractComparisonTests(unittest.TestCase):
    def test_rounding_allowed_but_changed_counts_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            a, b = Path(td) / "a.npz", Path(td) / "b.npz"
            np.savez(a, counts=np.array([1, 2]), weights=np.array([0.25, 0.75]))
            np.savez(b, counts=np.array([1, 2]), weights=np.nextafter([0.25, 0.75], 1.0))
            self.assertTrue(compare_npz(a, b)[0])
            np.savez(b, counts=np.array([1, 3]), weights=np.array([0.25, 0.75]))
            self.assertFalse(compare_npz(a, b)[0])

    def test_changed_weights_keys_dtype_and_missing_file_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            a, b = Path(td) / "a.npz", Path(td) / "b.npz"
            np.savez(a, weights=np.array([0.25, 0.75]))
            self.assertFalse(compare_npz(a, b)[0])
            for data in ({"weights": np.array([0.250001, 0.749999])},
                         {"other": np.array([0.25, 0.75])},
                         {"weights": np.array([0.25, 0.75], dtype=np.float32)},
                         {"weights": np.array([np.nan, 0.75])}):
                np.savez(b, **data)
                self.assertFalse(compare_npz(a, b)[0])


class CLITests(unittest.TestCase):
    def invoke(self, script, *args):
        return subprocess.run([sys.executable, str(ROOT / script), *args],
                              cwd=ROOT, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=60,
                              env=dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8"))

    def test_training_help_needs_no_data_or_torch(self):
        result = self.invoke("scripts/train_generator.py", "--help")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_invalid_training_budget_is_rejected_before_loading(self):
        result = self.invoke("scripts/train_generator.py", "--name", "test", "--steps", "0")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--steps must be positive", result.stderr)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "optional PyTorch not installed")
    def test_training_and_sampling_help_without_atlas(self):
        for script in ("experiments/run_one.py", "scripts/train_conditional_generator.py",
                       "scripts/fit_noised_value_classifier.py", "scripts/sample_value_guided.py",
                       "scripts/run_mdlm_baseline.py"):
            with self.subTest(script=script):
                result = self.invoke(script, "--help")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)


if __name__ == "__main__":
    unittest.main()
