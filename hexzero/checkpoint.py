"""
Checkpoint I/O utilities.

Saves/loads model + optimizer state atomically.  Manages a rolling
window of the last N checkpoints and a `best.pt` symlink.
"""

import os
import shutil
from pathlib import Path

import torch

from hexzero.net import HexNet


def save(
    net: HexNet,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    metrics: dict,
    checkpoint_dir: str,
    keep_last_n: int = 5,
) -> str:
    """
    Write checkpoint to `checkpoint_dir/iter_{iteration:06d}.pt`.
    Also updates `best.pt` to point at the latest checkpoint.
    Returns the saved path.
    """
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    filename = f"iter_{iteration:06d}.pt"
    tmp_path  = ckpt_dir / (filename + ".tmp")
    final_path = ckpt_dir / filename

    payload = {
        "iteration": iteration,
        "model_state": net.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "metrics": metrics,
    }
    torch.save(payload, tmp_path)
    os.replace(tmp_path, final_path)  # atomic on POSIX

    # Update best.pt (copy — symlinks can be tricky across filesystems)
    best_path = ckpt_dir / "best.pt"
    shutil.copy2(final_path, best_path)

    # Prune old checkpoints
    checkpoints = sorted(ckpt_dir.glob("iter_*.pt"), key=lambda p: p.name)
    while len(checkpoints) > keep_last_n:
        checkpoints.pop(0).unlink(missing_ok=True)

    return str(final_path)


def load(path: str, device: torch.device | None = None) -> dict:
    """Load a checkpoint dict.  Caller is responsible for applying state dicts."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.load(path, map_location=device, weights_only=False)


def best_checkpoint_path(checkpoint_dir: str) -> str | None:
    p = Path(checkpoint_dir) / "best.pt"
    return str(p) if p.exists() else None


def latest_iteration(checkpoint_dir: str) -> int:
    ckpt_dir = Path(checkpoint_dir)
    checkpoints = sorted(ckpt_dir.glob("iter_*.pt"), key=lambda p: p.name)
    if not checkpoints:
        return 0
    name = checkpoints[-1].stem  # "iter_000042"
    return int(name.split("_")[1])
