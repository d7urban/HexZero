"""
Training loop: pulls batches from the replay buffer, computes losses,
updates the network, and emits metrics.

Can run in a QThread (GUI mode) or as a plain blocking call (headless).
Emits metrics via an optional signals object (gui/signals.py TrainingSignals).
"""

from __future__ import annotations

import random
from collections import deque
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

import hexzero.checkpoint as ckpt_io
from config import HexZeroConfig
from hexzero.net import HexNet, build_net
from hexzero.replay_buffer import ReplayBuffer

if TYPE_CHECKING:
    from hexzero.gui.signals import TrainingSignals


class Trainer:
    def __init__(
        self,
        cfg: HexZeroConfig,
        net: HexNet | None = None,
        device: torch.device | None = None,
        signals: TrainingSignals | None = None,
    ):
        self.cfg = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = net if net is not None else build_net(cfg, self.device)
        self.signals = signals

        self.optimizer = torch.optim.AdamW(
            self.net.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=cfg.lr_cosine_steps,
            eta_min=cfg.lr_min,
        )

        self.replay_buffer = ReplayBuffer(cfg.replay_buffer_capacity)
        self.global_step = 0
        self.metrics_history: deque = deque(maxlen=10_000)

    def load_checkpoint(self, path: str) -> None:
        data = ckpt_io.load(path, self.device)
        ckpt_io.load_weights(self.net, data["model_state"])
        self.optimizer.load_state_dict(data["optimizer_state"])
        if "scheduler_state" in data:
            self.scheduler.load_state_dict(data["scheduler_state"])
        self.global_step = data.get("metrics", {}).get("global_step", 0)

    def save_checkpoint(self, iteration: int, board_size: int | None = None) -> str:
        metrics = {"global_step": self.global_step}
        if board_size is not None:
            metrics["board_size"] = board_size
        return ckpt_io.save(
            self.net, self.optimizer, iteration, metrics,
            self.cfg.checkpoint_dir, self.cfg.keep_last_n_checkpoints,
            scheduler=self.scheduler,
        )

    def reset_lr(self) -> None:
        """Reset LR to initial value and restart the cosine schedule.
        Called at curriculum advances so the network can adapt quickly to
        the new board size without starting from a decayed LR floor."""
        for pg in self.optimizer.param_groups:
            pg["lr"] = self.cfg.learning_rate
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.cfg.lr_cosine_steps,
            eta_min=self.cfg.lr_min,
        )

    def train_step(self, board_size: int | None = None) -> dict | None:
        """
        One gradient update.  Returns metrics dict or None if buffer too small.
        """
        cfg = self.cfg
        # Dynamic batching: randomly pick a size if not specified
        if board_size is None:
            sizes = self.replay_buffer.sizes_available()
            if not sizes:
                return None
            board_size = random.choice(sizes)

        batch = self.replay_buffer.sample_batch(cfg.batch_size, board_size)
        if batch is None:
            return None

        features    = batch["features"].to(self.device)
        policy_tgt  = batch["policy"].to(self.device)
        value_tgt   = batch["value"].to(self.device)
        size_scalar = batch["size_scalar"].to(self.device)

        self.net.train()
        log_pi, value_pred = self.net(features, size_scalar)

        # Policy loss: cross-entropy = -sum(target * log_pred)
        policy_loss = -(policy_tgt * log_pi).sum(dim=1).mean()

        # Value loss: MSE
        value_loss = F.mse_loss(value_pred, value_tgt)

        loss = policy_loss + cfg.value_loss_weight * value_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()

        self.global_step += 1

        # Policy accuracy: fraction where argmax(pred) == argmax(target)
        pred_move = log_pi.argmax(dim=1)
        tgt_move  = policy_tgt.argmax(dim=1)
        accuracy  = (pred_move == tgt_move).float().mean().item()

        metrics = {
            "step":         self.global_step,
            "board_size":   board_size,
            "loss":         loss.item(),
            "policy_loss":  policy_loss.item(),
            "value_loss":   value_loss.item(),
            "policy_acc":   accuracy,
            "value_mae":    (value_pred - value_tgt).abs().mean().item(),
            "lr":           self.scheduler.get_last_lr()[0],
        }
        self.metrics_history.append(metrics)

        if self.signals is not None:
            self.signals.metrics_updated.emit(metrics)

        return metrics

    def train_iteration(self, _iteration: int, board_size: int | None = None) -> list:
        """
        Run cfg.train_steps_per_iteration gradient updates.
        Returns list of per-step metrics.
        """
        all_metrics = []
        for _ in range(self.cfg.train_steps_per_iteration):
            m = self.train_step(board_size)
            if m is not None:
                all_metrics.append(m)
        return all_metrics
