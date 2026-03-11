"""
Checkpoint I/O utilities.

Saves/loads model + optimizer state atomically.  Manages a rolling
window of the last N checkpoints and a `best.pt` symlink.
"""

import json
import os
import shutil
from pathlib import Path

import torch

from hexzero.net import HexNet


def _raw(net) -> HexNet:
    """Strip a torch.compile OptimizedModule wrapper, if present."""
    return getattr(net, "_orig_mod", net)


def load_weights(net, state_dict: dict, strict: bool = True) -> None:
    """Load a state dict into net, unwrapping torch.compile if needed."""
    _raw(net).load_state_dict(state_dict, strict=strict)


def save(
    net: HexNet,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    metrics: dict,
    checkpoint_dir: str,
    keep_last_n: int = 5,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
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
        "model_state": _raw(net).state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "metrics": metrics,
    }
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    torch.save(payload, tmp_path)
    os.replace(tmp_path, final_path)  # atomic on POSIX

    # Prune old checkpoints
    checkpoints = sorted(ckpt_dir.glob("iter_*.pt"), key=lambda p: p.name)
    while len(checkpoints) > keep_last_n:
        checkpoints.pop(0).unlink(missing_ok=True)

    return str(final_path)


def promote_to_best(path: str, checkpoint_dir: str) -> None:
    """Copy `path` to `checkpoint_dir/best.pt`.

    Must be called explicitly by the caller after confirming the checkpoint
    is a new champion.  checkpoint.save() intentionally does NOT do this.
    """
    shutil.copy2(path, Path(checkpoint_dir) / "best.pt")


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


def save_training_state(checkpoint_dir: str, state: dict) -> None:
    """Atomically persist a small training-state dict as JSON."""
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tmp  = ckpt_dir / "training_state.json.tmp"
    dest = ckpt_dir / "training_state.json"
    tmp.write_text(json.dumps(state))
    os.replace(tmp, dest)


def load_training_state(checkpoint_dir: str) -> dict:
    """Return the dict saved by save_training_state, or {} if absent/corrupt."""
    path = Path(checkpoint_dir) / "training_state.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}
