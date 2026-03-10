"""
Replay buffer storing self-play game records.

Each sample is a dict:
    features    : np.ndarray (C, H, W) float32
    policy      : np.ndarray (H*W,)    float32  — MCTS visit distribution
    value       : float                         — game outcome from this player's POV
    board_size  : int
"""

import threading
import random
import torch
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self._buffer: list[dict] = []
        self._idx: int = 0
        self._lock = threading.Lock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)

    def add(self, sample: dict) -> None:
        with self._lock:
            if len(self._buffer) < self.capacity:
                self._buffer.append(sample)
            else:
                self._buffer[self._idx % self.capacity] = sample
            self._idx += 1

    def add_game(self, samples: list[dict]) -> None:
        for s in samples:
            self.add(s)

    def sample_batch(
        self,
        n: int,
        board_size: int | None = None,
    ) -> dict[str, torch.Tensor] | None:
        """
        Sample a batch of n records.  If board_size is given, only samples
        matching that size are returned.  Returns None if not enough samples.
        """
        with self._lock:
            if board_size is not None:
                pool = [s for s in self._buffer if s["board_size"] == board_size]
            else:
                pool = self._buffer

            if len(pool) < n:
                return None

            chosen = random.sample(pool, n)

        features    = torch.tensor(np.stack([s["features"]   for s in chosen]), dtype=torch.float32)
        policy      = torch.tensor(np.stack([s["policy"]     for s in chosen]), dtype=torch.float32)
        value       = torch.tensor([s["value"]                for s in chosen], dtype=torch.float32)
        size_scalar = torch.tensor([[s["size_norm"]]           for s in chosen], dtype=torch.float32)

        return {
            "features":    features,      # (B, C, H, W)
            "policy":      policy,        # (B, H*W)
            "value":       value,         # (B,)
            "size_scalar": size_scalar,   # (B, 1)
        }

    def sizes_available(self) -> list[int]:
        """Return sorted list of board sizes present in the buffer."""
        with self._lock:
            return sorted(set(s["board_size"] for s in self._buffer))

    def save(self, path: str) -> None:
        with self._lock:
            torch.save({"buffer": self._buffer, "idx": self._idx}, path)

    def load(self, path: str) -> None:
        data = torch.load(path, weights_only=False)
        with self._lock:
            self._buffer = data["buffer"]
            self._idx    = data["idx"]
