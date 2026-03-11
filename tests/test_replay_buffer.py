"""Tests for hexzero/replay_buffer.py — ReplayBuffer."""
import unittest

import numpy as np
import torch

from hexzero.replay_buffer import ReplayBuffer


def _make_sample(board_size: int = 5, value: float = 1.0) -> dict:
    n = board_size
    return {
        "features":   np.zeros((8, n, n), dtype=np.float32),
        "policy":     np.ones(n * n + 1, dtype=np.float32) / (n * n + 1),
        "value":      value,
        "board_size": board_size,
        "size_norm":  0.0,
    }


class AddAndLenTests(unittest.TestCase):
    def test_len_grows_with_adds(self):
        buf = ReplayBuffer(100)
        self.assertEqual(len(buf), 0)
        buf.add(_make_sample())
        self.assertEqual(len(buf), 1)
        buf.add(_make_sample())
        self.assertEqual(len(buf), 2)

    def test_len_does_not_exceed_capacity(self):
        buf = ReplayBuffer(3)
        for _ in range(10):
            buf.add(_make_sample())
        self.assertEqual(len(buf), 3)

    def test_ring_overwrite(self):
        """Oldest sample is replaced when buffer is full."""
        buf = ReplayBuffer(2)
        buf.add(_make_sample(value=1.0))
        buf.add(_make_sample(value=2.0))
        buf.add(_make_sample(value=3.0))  # overwrites first
        values = {s["value"] for s in buf._buffer}
        self.assertNotIn(1.0, values)
        self.assertIn(2.0, values)
        self.assertIn(3.0, values)

    def test_add_game_adds_all_samples(self):
        buf = ReplayBuffer(100)
        samples = [_make_sample(value=float(i)) for i in range(5)]
        buf.add_game(samples)
        self.assertEqual(len(buf), 5)


class SampleBatchTests(unittest.TestCase):
    def test_returns_none_when_insufficient_samples(self):
        buf = ReplayBuffer(100)
        for _ in range(3):
            buf.add(_make_sample())
        result = buf.sample_batch(5)
        self.assertIsNone(result)

    def test_returns_dict_with_correct_keys(self):
        buf = ReplayBuffer(100)
        for _ in range(10):
            buf.add(_make_sample())
        batch = buf.sample_batch(4)
        self.assertIsNotNone(batch)
        self.assertIn("features", batch)
        self.assertIn("policy", batch)
        self.assertIn("value", batch)
        self.assertIn("size_scalar", batch)

    def test_batch_tensor_shapes(self):
        n = 5
        buf = ReplayBuffer(100)
        for _ in range(10):
            buf.add(_make_sample(board_size=n))
        batch = buf.sample_batch(4)
        self.assertEqual(batch["features"].shape, (4, 8, n, n))
        self.assertEqual(batch["policy"].shape, (4, n * n + 1))
        self.assertEqual(batch["value"].shape, (4,))
        self.assertEqual(batch["size_scalar"].shape, (4, 1))

    def test_batch_tensor_dtypes(self):
        buf = ReplayBuffer(100)
        for _ in range(10):
            buf.add(_make_sample())
        batch = buf.sample_batch(4)
        self.assertEqual(batch["features"].dtype, torch.float32)
        self.assertEqual(batch["policy"].dtype, torch.float32)
        self.assertEqual(batch["value"].dtype, torch.float32)
        self.assertEqual(batch["size_scalar"].dtype, torch.float32)

    def test_board_size_filter_returns_matching_samples(self):
        buf = ReplayBuffer(100)
        for _ in range(6):
            buf.add(_make_sample(board_size=5))
        for _ in range(6):
            buf.add(_make_sample(board_size=9))
        batch = buf.sample_batch(4, board_size=5)
        self.assertIsNotNone(batch)
        self.assertEqual(batch["features"].shape, (4, 8, 5, 5))

    def test_board_size_filter_none_if_insufficient(self):
        buf = ReplayBuffer(100)
        for _ in range(10):
            buf.add(_make_sample(board_size=5))
        # Only 5×5 samples, request 9×9
        result = buf.sample_batch(4, board_size=9)
        self.assertIsNone(result)

    def test_sample_returns_exact_n_items(self):
        buf = ReplayBuffer(100)
        for _ in range(20):
            buf.add(_make_sample())
        for n in [1, 5, 20]:
            with self.subTest(n=n):
                batch = buf.sample_batch(n)
                self.assertEqual(batch["value"].shape[0], n)


class SizesAvailableTests(unittest.TestCase):
    def test_empty_buffer(self):
        buf = ReplayBuffer(100)
        self.assertEqual(buf.sizes_available(), [])

    def test_single_size(self):
        buf = ReplayBuffer(100)
        buf.add(_make_sample(board_size=7))
        buf.add(_make_sample(board_size=7))
        self.assertEqual(buf.sizes_available(), [7])

    def test_multiple_sizes_sorted(self):
        buf = ReplayBuffer(100)
        buf.add(_make_sample(board_size=9))
        buf.add(_make_sample(board_size=5))
        buf.add(_make_sample(board_size=11))
        self.assertEqual(buf.sizes_available(), [5, 9, 11])


if __name__ == "__main__":
    unittest.main()
