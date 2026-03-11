"""
GPU inference server — batches state evaluations from worker threads
into a single forward pass, keeping the GPU saturated.

Workers call server.infer(features, size_norm) from any thread.
The call blocks until the batch it belongs to has been evaluated,
then returns (policy np.ndarray, value float).
"""

import queue
import threading
import time

import numpy as np
import torch

from hexzero.net import HexNet


class InferenceServer:
    """
    Runs a single background thread that:
      1. Waits up to `max_wait_ms` for requests to accumulate.
      2. Evaluates a batch of up to `max_batch` states on `device`.
      3. Distributes results back to waiting worker threads.

    NOTE: all states in one batch must share the same board size, which
    is guaranteed when callers play a fixed board size per iteration.
    """

    def __init__(
        self,
        net: HexNet,
        device: torch.device,
        max_batch: int = 64,
        max_wait_ms: float = 5.0,
    ):
        self._net       = net
        self._device    = device
        self._max_batch = max_batch
        self._wait_s    = max_wait_ms / 1000.0
        self._q: queue.Queue = queue.Queue()
        self._stop      = threading.Event()
        self._thread    = threading.Thread(
            target=self._loop, daemon=True, name="InferenceServer"
        )
        self._thread.start()

    # ------------------------------------------------------------------
    # Public API (safe to call from any thread)
    # ------------------------------------------------------------------

    def infer(self, features: np.ndarray, size_norm: float):
        """Submit one state; block until evaluated. Returns (policy, value)."""
        event  = threading.Event()
        result: list = [None]
        self._q.put((features, size_norm, event, result))
        event.wait()
        return result[0]   # (np.ndarray policy, float value)

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    # ------------------------------------------------------------------
    # Background loop (runs in self._thread)
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        self._net.eval()
        while not self._stop.is_set():
            batch = self._collect()
            if batch:
                self._evaluate(batch)

    def _collect(self) -> list:
        """Collect up to max_batch requests, waiting at most _wait_s."""
        batch: list = []
        deadline = time.monotonic() + self._wait_s
        try:
            batch.append(self._q.get(timeout=self._wait_s))
        except queue.Empty:
            return batch
        while len(batch) < self._max_batch:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                batch.append(self._q.get(timeout=remaining))
            except queue.Empty:
                break
        return batch

    def _evaluate(self, batch: list) -> None:
        feats = np.stack([item[0] for item in batch])
        norms = [[item[1]] for item in batch]
        x = torch.tensor(feats, dtype=torch.float32).to(self._device)
        s = torch.tensor(norms, dtype=torch.float32).to(self._device)
        with torch.no_grad():
            log_pi, v = self._net(x, s)
        policies = torch.exp(log_pi).cpu().numpy()
        values   = v.cpu().numpy().flatten()
        for (_, _, event, result), pi, val in zip(batch, policies, values, strict=True):
            result[0] = (pi, float(val))
            event.set()
