"""
HexZero — AlphaZero-style self-play training for Hex.

Usage:
    python main.py                    # Launch GUI
    python main.py --headless         # Run training without GUI
    python main.py --board-size 9     # Override initial board size
    python main.py --checkpoint path  # Resume from checkpoint
"""

import argparse
import os
import sys

from config import HexZeroConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HexZero training suite")
    p.add_argument("--headless",    action="store_true",  help="Run without GUI")
    p.add_argument("--board-size",  type=int, default=None, help="Initial board size")
    p.add_argument("--checkpoint",  type=str, default=None, help="Path to checkpoint to load")
    p.add_argument("--simulations", type=int, default=None, help="MCTS simulations per move")
    p.add_argument("--workers",     type=int, default=None, help="Self-play worker processes")
    return p.parse_args()


def _loss_plateau(losses: list[float], window: int, threshold: float) -> bool:
    """Return True when loss improvement over `window` iters drops below `threshold`."""
    if len(losses) < window:
        return False
    recent = losses[-window:]
    start  = recent[0]
    if start <= 0:
        return True
    return (start - recent[-1]) / start < threshold


def run_headless(cfg: HexZeroConfig, resume_path: str = None) -> None:
    import torch

    import hexzero.checkpoint as ckpt_io
    from hexzero.arena import candidate_is_better, run_arena
    from hexzero.net import build_net
    from hexzero.self_play import run_self_play_parallel
    from hexzero.trainer import Trainer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    net     = build_net(cfg, device)
    trainer = Trainer(cfg, net, device)

    buf_path   = os.path.join(cfg.checkpoint_dir, "replay_buffer.pt.gz")
    best_path  = resume_path or ckpt_io.best_checkpoint_path(cfg.checkpoint_dir)
    board_size = cfg.initial_board_size

    # training_state.json is written atomically after every iteration and is
    # the single authoritative source for iteration counter and curriculum state.
    ts = ckpt_io.load_training_state(cfg.checkpoint_dir)

    if best_path is None:
        print("No checkpoint found — saving initial weights…")
        best_path = trainer.save_checkpoint(0, board_size)
        ckpt_io.promote_to_best(best_path, cfg.checkpoint_dir)
        iteration     = 0
        iters_on_size = 0
        size_losses: list[float] = []
    else:
        trainer.load_checkpoint(best_path)
        # Load once and reuse for board_size; avoids a second torch.load call.
        ckpt_data  = ckpt_io.load(best_path, device)
        saved_size = ckpt_data.get("metrics", {}).get("board_size")
        if saved_size and saved_size in cfg.board_sizes:
            board_size = saved_size

        if resume_path:
            # Explicit --checkpoint: the JSON may belong to a different run;
            # trust the checkpoint's own counter and reset curriculum state.
            iteration     = ckpt_data.get("iteration", 0)
            iters_on_size = 0
            size_losses   = []
        else:
            # Normal resume: JSON is consistent with best.pt (both updated
            # together at the end of each accepted iteration).
            iteration     = ts.get("iteration", 0)
            iters_on_size = ts.get("iters_on_size", 0)
            size_losses   = ts.get("size_losses", [])

        print(f"Resumed from {best_path}  (board {board_size}×{board_size}  iter {iteration})")

    if os.path.exists(buf_path):
        print(f"Loading replay buffer from {buf_path}…", end="", flush=True)
        trainer.replay_buffer.load(buf_path)
        print(f" {len(trainer.replay_buffer)} samples")

    try:
        while True:
            iteration += 1
            print(f"\n{'='*60}")
            print(f"Iteration {iteration}  |  board {board_size}×{board_size}")

            print("  Self-play…", end="", flush=True)
            samples, swap_games, games_played = run_self_play_parallel(
                cfg, best_path, device, board_size, cfg.games_per_iteration)
            for s in samples:
                trainer.replay_buffer.add(s)
            swap_info = ""
            if cfg.use_pie_rule and games_played > 0:
                swap_info = f"  swap {swap_games}/{games_played} ({100*swap_games//games_played}%)"
            print(f" {len(samples)} samples ({games_played} games × 2){swap_info}")
            trainer.replay_buffer.save(buf_path)

            print("  Training…", end="", flush=True)
            metrics_list = trainer.train_iteration(iteration, board_size)
            if metrics_list:
                last = metrics_list[-1]
                mean_loss = sum(m["policy_loss"] for m in metrics_list) / len(metrics_list)
                size_losses.append(mean_loss)
                plateau = _loss_plateau(
                    size_losses, cfg.min_iters_per_size, cfg.loss_plateau_threshold)
                print(f" loss={last['loss']:.4f}  policy_acc={last['policy_acc']:.3f}"
                      f"  plateau={plateau}")

            cand_path = trainer.save_checkpoint(iteration, board_size)

            print("  Arena…", end="", flush=True)
            cw, chw, draws = run_arena(cand_path, best_path, cfg, board_size)
            total = cw + chw + draws
            print(f" candidate {cw}/{total}  champion {chw}/{total}")

            iters_on_size += 1
            if candidate_is_better(cw, chw, total, cfg.arena_win_threshold):
                best_path = cand_path
                ckpt_io.promote_to_best(best_path, cfg.checkpoint_dir)
                print("  → New champion accepted.")
                size_idx  = cfg.board_sizes.index(board_size)
                next_idx  = size_idx + 1
                if (_loss_plateau(size_losses, cfg.min_iters_per_size,
                                  cfg.loss_plateau_threshold)
                        and iters_on_size >= cfg.min_iters_per_size
                        and next_idx < len(cfg.board_sizes)):
                    board_size    = cfg.board_sizes[next_idx]
                    iters_on_size = 0
                    size_losses   = []
                    trainer.reset_lr()
                    print(f"  → Curriculum advanced to {board_size}×{board_size}!")
            else:
                print("  → Champion retained.")

            ckpt_io.save_training_state(cfg.checkpoint_dir, {
                "iteration":    iteration,
                "board_size":   board_size,
                "size_idx":     cfg.board_sizes.index(board_size),
                "iters_on_size": iters_on_size,
                "size_losses":  size_losses,
            })

    except KeyboardInterrupt:
        print("\nTraining interrupted. Checkpoint saved at:", best_path)


def run_gui(cfg: HexZeroConfig, resume_path: str = None) -> None:
    from PyQt6.QtWidgets import QApplication

    from hexzero.gui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Dark palette
    from PyQt6.QtGui import QColor, QPalette
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window,          QColor(45,  45,  45))
    palette.setColor(QPalette.ColorRole.WindowText,      QColor(210, 210, 210))
    palette.setColor(QPalette.ColorRole.Base,            QColor(30,  30,  30))
    palette.setColor(QPalette.ColorRole.AlternateBase,   QColor(45,  45,  45))
    palette.setColor(QPalette.ColorRole.Text,            QColor(210, 210, 210))
    palette.setColor(QPalette.ColorRole.Button,          QColor(60,  60,  60))
    palette.setColor(QPalette.ColorRole.ButtonText,      QColor(210, 210, 210))
    palette.setColor(QPalette.ColorRole.Highlight,       QColor(80,  120, 200))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)

    window = MainWindow(cfg)
    if resume_path:
        import hexzero.checkpoint as ckpt_io
        try:
            import torch
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            data = ckpt_io.load(resume_path, device)
            ckpt_io.load_weights(window._net, data["model_state"])
        except Exception as e:
            print(f"Warning: could not load checkpoint: {e}")

    window.show()
    sys.exit(app.exec())


def main() -> None:
    args = parse_args()
    cfg  = HexZeroConfig()

    if args.board_size is not None:
        cfg.initial_board_size = args.board_size
    if args.simulations is not None:
        cfg.mcts_simulations = args.simulations
    if args.workers is not None:
        cfg.num_self_play_workers = args.workers

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    if args.headless:
        run_headless(cfg, resume_path=args.checkpoint)
    else:
        run_gui(cfg, resume_path=args.checkpoint)


if __name__ == "__main__":
    main()
