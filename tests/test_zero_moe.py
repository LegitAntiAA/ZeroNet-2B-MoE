import json
import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from zero_moe.checkpoint import FORMAT_VERSION, load_expert, pack_checkpoint, read_manifest
from zero_moe.experiment import evaluate, predict, train
from zero_moe.model import EXPERT_COUNT, PARAMETERS_PER_EXPERT, Expert, encode, examples


class ZeroNetTests(unittest.TestCase):
    def test_exact_parameter_count_and_all_layers_receive_updates(self):
        rng = np.random.default_rng(42)
        expert = Expert.initialize(rng)
        self.assertEqual(expert.weights.size, 2_000_000)
        self.assertEqual(PARAMETERS_PER_EXPERT * EXPERT_COUNT, 2_000_000_000)
        original = [matrix.copy() for matrix, _ in expert.layers]
        numbers, labels = examples(rng, 64)
        expert.fit_step(encode(numbers), labels, np.float32(0.018))
        for before, (after, _) in zip(original, expert.layers):
            self.assertGreater(np.count_nonzero(before != after), before.size * 0.9)

    def test_sensor_merges_small_values_and_handles_large_finite_numbers(self):
        features = encode([0, 0.12, -0.12, 0.13, -1e308, 1e308])
        np.testing.assert_array_equal(features[:3], np.zeros((3, 3)))
        self.assertGreater(features[3, 1], 0.0)
        np.testing.assert_array_equal(features[-2:], [[-1, 1, 1], [1, 1, 1]])
        for number in (np.nan, np.inf, -np.inf):
            with self.subTest(number=number), self.assertRaises(ValueError):
                encode([number])

    def test_examples_use_true_labels(self):
        numbers, labels = examples(np.random.default_rng(123), 1000)
        np.testing.assert_array_equal(labels, numbers != 0.0)
        self.assertTrue(0.47 < labels.mean() < 0.53)

    def test_train_save_reload_predict_and_evaluate(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "weights"
            manifest = train(directory, expert_count=2, steps=64, seed=2026, minutes=1)
            self.assertEqual(read_manifest(directory), manifest)
            self.assertEqual(manifest["total_parameters"], 4_000_000)
            self.assertFalse(
                np.array_equal(load_expert(directory, 0).weights, load_expert(directory, 1).weights)
            )
            numbers = [0, 12, -12, 0.05, -0.05]
            expected = [False, True, True, False, False]
            np.testing.assert_array_equal(predict(directory, numbers, seed=7), expected)
            results = evaluate(directory, cases_per_expert=1024, seed=2027)
            self.assertEqual(results["examples"], 2048)
            self.assertTrue(0.65 < results["accuracy"] < 0.75)
            self.assertEqual(results["confusion_matrix"][0][1], 0)

            artifact = Path(temporary) / "release.zip"
            pack_checkpoint(directory, artifact)
            with self.assertRaises(FileExistsError):
                pack_checkpoint(directory, artifact)
            self.assertLess(artifact.stat().st_size, 6_000_000)
            np.testing.assert_array_equal(predict(artifact, numbers, seed=7), expected)
            released = evaluate(artifact, cases_per_expert=1024, seed=2027)
            self.assertEqual(released["examples"], results["examples"])
            self.assertAlmostEqual(released["accuracy"], results["accuracy"], delta=0.005)

    def test_incomplete_checkpoint_does_not_publish_release(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest = {
                "format": FORMAT_VERSION,
                "trained_experts": 1,
                "active_parameters": PARAMETERS_PER_EXPERT,
                "total_parameters": PARAMETERS_PER_EXPERT,
            }
            (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            artifact = directory / "release.zip"
            with self.assertRaises(FileNotFoundError):
                pack_checkpoint(directory, artifact)
            self.assertFalse(artifact.exists())
            self.assertFalse(artifact.with_name("release.zip.tmp").exists())

    def test_timeout_does_not_publish_a_partial_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "weights"
            with self.assertRaisesRegex(TimeoutError, "0/2 completed experts"):
                train(directory, expert_count=2, steps=64, seed=2026, minutes=1e-9)
            self.assertFalse((directory / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
