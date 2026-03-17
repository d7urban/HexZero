"""
Training loop: pulls batches from the replay buffer, computes losses,
updates the network, and emits metrics.

Can run in a QThread (GUI mode) or as a plain blocking call (headless).
Emits metrics via an optional signals object (gui/signals.py TrainingSignals).
"""

from __future__ import annotations

import os
import random
import threading
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


# ---------------------------------------------------------------------------
# Callbacks for run_loop — override any method you care about
# ---------------------------------------------------------------------------

class LoopCallbacks:
    """All methods are no-ops by default. Subclass to handle events."""
    def on_status(self, msg: str) -> None: pass
    def on_iteration_start(self, iteration: int, board_size: int) -> None: pass
    def on_self_play_progress(self, done: int, total: int) -> None: pass
    def on_self_play_done(self, n_samples: int, swap_games: int, games_played: int) -> None: pass
    def on_buffer_updated(self, n: int) -> None: pass
    def on_swap_rate(self, swap_games: int, games_played: int) -> None: pass
    def on_train_done(self, metrics_list: list) -> None: pass
    def on_arena_progress(self, done: int, cw_so_far: int, total: int) -> None: pass
    def on_arena_done(self, cw: int, chw: int, draws: int) -> None: pass
    def on_promoted(self, cand_path: str) -> None: pass
    def on_champion_retained(self) -> None: pass
    def on_promotion_freq(self, recent_promotions: list) -> None: pass
    def on_curriculum_progress(self, iters_on_size: int, min_iters: int) -> None: pass
    def on_board_size_advanced(self, board_size: int, reason: str, promoted: bool, iteration: int) -> None: pass
    def on_iteration_done(self, iteration: int) -> None: pass


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

        # Mixed precision: autocast + GradScaler on CUDA; no-ops on CPU.
        # init_scale=2**10 avoids fp16 overflow on early steps with a freshly
        # initialised network (default 65536 is too aggressive for small models).
        self._use_amp = self.device.type == "cuda"
        self._scaler  = torch.amp.GradScaler("cuda", enabled=self._use_amp, init_scale=2**10)

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
        with torch.amp.autocast("cuda", enabled=self._use_amp):
            log_pi, value_pred = self.net(features, size_scalar)

            # Policy loss: cross-entropy = -sum(target * log_pred)
            policy_loss = -(policy_tgt * log_pi).sum(dim=1).mean()

            # Value loss: MSE
            value_loss = F.mse_loss(value_pred, value_tgt)

            loss = policy_loss + cfg.value_loss_weight * value_loss

        self.optimizer.zero_grad()
        self._scaler.scale(loss).backward()
        self._scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
        self._scaler.step(self.optimizer)
        self._scaler.update()
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

    def run_loop(
        self,
        stop_event: threading.Event | None = None,
        callbacks: LoopCallbacks | None = None,
        initial_path: str | None = None,
    ) -> None:
        """Self-play → train → arena → promote → curriculum, forever.

        Args:
            stop_event:   when set, exits cleanly at the next iteration boundary.
            callbacks:    progress hooks (subclass LoopCallbacks for GUI or print output).
            initial_path: override best.pt as starting champion (for --checkpoint resume).
        """
        from hexzero.arena import candidate_is_better, run_arena
        from hexzero.self_play import run_self_play_parallel

        cfg  = self.cfg
        cb   = callbacks or LoopCallbacks()
        stop = stop_event or threading.Event()

        buf_path = os.path.join(cfg.checkpoint_dir, "replay_buffer.pt.gz")

        # ---- Bootstrap or locate champion ----------------------------------------
        best_path = initial_path or ckpt_io.best_checkpoint_path(cfg.checkpoint_dir)
        if best_path is None:
            cb.on_status("Saving initial checkpoint…")
            init_path = self.save_checkpoint(0, cfg.initial_board_size)
            ckpt_io.promote_to_best(init_path, cfg.checkpoint_dir)
            best_path = ckpt_io.best_checkpoint_path(cfg.checkpoint_dir)
            cb.on_promoted(init_path)

        # ---- Replay buffer -------------------------------------------------------
        if os.path.exists(buf_path):
            cb.on_status("Loading replay buffer…")
            try:
                self.replay_buffer.load(buf_path)
                cb.on_buffer_updated(len(self.replay_buffer))
            except Exception as e:
                cb.on_status(f"Could not load replay buffer: {e}")

        # ---- Restore training state ----------------------------------------------
        ts = ckpt_io.load_training_state(cfg.checkpoint_dir)
        board_size = cfg.initial_board_size
        if ts.get("board_size") and ts["board_size"] in cfg.board_sizes:
            board_size = ts["board_size"]
        size_idx          = cfg.board_sizes.index(board_size) if board_size in cfg.board_sizes else 0
        iteration         = ts.get("iteration", ckpt_io.latest_iteration(cfg.checkpoint_dir))
        iters_on_size     = ts.get("iters_on_size", 0)
        recent_promotions: list[bool] = ts.get("recent_promotions", [])
        cb.on_promotion_freq(recent_promotions)

        # ---- Main loop -----------------------------------------------------------
        while not stop.is_set():
            iteration += 1
            cb.on_iteration_start(iteration, board_size)
            cb.on_status(f"Iteration {iteration} — self-play ({board_size}×{board_size})…")
            cb.on_self_play_progress(0, cfg.games_per_iteration)

            def _sp_progress(done: int, total: int) -> None:
                cb.on_self_play_progress(done, total)
                cb.on_buffer_updated(len(self.replay_buffer))

            samples, swap_games, games_played = run_self_play_parallel(
                cfg, best_path, self.device, board_size, cfg.games_per_iteration,
                progress_callback=_sp_progress, stop_event=stop,
            )
            cb.on_swap_rate(swap_games, games_played)
            for s in samples:
                self.replay_buffer.add(s)
            cb.on_buffer_updated(len(self.replay_buffer))
            cb.on_self_play_done(len(samples), swap_games, games_played)
            try:
                self.replay_buffer.save(buf_path)
            except Exception:
                pass

            if stop.is_set():
                break

            cb.on_status(f"Iteration {iteration} — training…")
            metrics_list = self.train_iteration(iteration, board_size)
            cb.on_train_done(metrics_list)

            if stop.is_set():
                break

            cand_path = self.save_checkpoint(iteration, board_size)

            cb.on_status(f"Iteration {iteration} — arena…")
            cw, chw, draws = run_arena(
                cand_path, best_path, cfg, board_size,
                progress_callback=lambda d, c, t: cb.on_arena_progress(d, c, t),
                stop_event=stop,
            )

            if stop.is_set():
                break

            cb.on_arena_done(cw, chw, draws)
            total    = cw + chw + draws
            win_rate = cw / total if total > 0 else 0.0

            iters_on_size += 1
            cb.on_curriculum_progress(iters_on_size, cfg.min_iters_per_size)

            promoted = candidate_is_better(cw, chw, total, cfg.arena_win_threshold)
            if promoted:
                ckpt_io.promote_to_best(cand_path, cfg.checkpoint_dir)
                best_path = ckpt_io.best_checkpoint_path(cfg.checkpoint_dir)
                cb.on_promoted(cand_path)
            else:
                cb.on_champion_retained()

            recent_promotions.append(promoted)
            recent_promotions = recent_promotions[-cfg.min_iters_per_size:]
            cb.on_promotion_freq(recent_promotions)

            ckpt_io.save_training_state(cfg.checkpoint_dir, {
                "iteration":         iteration,
                "board_size":        board_size,
                "size_idx":          size_idx,
                "iters_on_size":     iters_on_size,
                "recent_promotions": recent_promotions,
            })

            # Curriculum advancement
            next_idx    = size_idx + 1
            no_promos   = (iters_on_size >= cfg.min_iters_per_size
                           and not any(recent_promotions))
            max_reached = iters_on_size >= cfg.max_iters_per_size
            can_advance = (
                iters_on_size >= cfg.min_iters_per_size
                and next_idx < len(cfg.board_sizes)
                and (no_promos or max_reached)
            )

            if can_advance:
                size_idx          = next_idx
                board_size        = cfg.board_sizes[size_idx]
                iters_on_size     = 0
                recent_promotions = []
                self.reset_lr()
                reason = "no recent promotions" if no_promos else f"max {cfg.max_iters_per_size} iters"
                cb.on_board_size_advanced(board_size, reason, promoted, iteration)
                cb.on_promotion_freq(recent_promotions)
                prefix = "new champion + " if promoted else ""
                cb.on_status(
                    f"Iteration {iteration} — {prefix}curriculum advanced to "
                    f"{board_size}×{board_size}! ({reason})"
                )
            else:
                arena_str = (
                    f"new champion! ({cw}/{total}, {win_rate:.0%})" if promoted
                    else f"champion retained ({chw}/{total}, {win_rate:.0%} for candidate)"
                )
                if next_idx >= len(cfg.board_sizes):
                    suffix = "  (at largest board size)"
                elif iters_on_size < cfg.min_iters_per_size:
                    iters_left = cfg.min_iters_per_size - iters_on_size
                    suffix = f"  (need {iters_left} more iter(s) on {board_size}×{board_size})"
                else:
                    promos = sum(recent_promotions)
                    suffix = (f"  (promos {promos}/{len(recent_promotions)}, "
                              f"max in {cfg.max_iters_per_size - iters_on_size} iter(s))")
                cb.on_status(f"Iteration {iteration} — {arena_str}{suffix}")

            cb.on_iteration_done(iteration)
