"""
HexZero — AlphaZero-style self-play training for Hex.

Usage:
    python main.py                    # Launch GUI
    python main.py --headless         # Run training without GUI
    python main.py --board-size 9     # Override initial board size
    python main.py --checkpoint path  # Resume from checkpoint
"""

import argparse
import sys
import os

from config import HexZeroConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HexZero training suite")
    p.add_argument("--headless",    action="store_true",  help="Run without GUI")
    p.add_argument("--board-size",  type=int, default=None, help="Initial board size")
    p.add_argument("--checkpoint",  type=str, default=None, help="Path to checkpoint to load")
    p.add_argument("--simulations", type=int, default=None, help="MCTS simulations per move")
    p.add_argument("--workers",     type=int, default=None, help="Self-play worker processes")
    return p.parse_args()


def run_headless(cfg: HexZeroConfig, resume_path: str = None) -> None:
    import torch
    from hexzero.net import build_net
    from hexzero.trainer import Trainer
    from hexzero.self_play import run_self_play_parallel
    from hexzero.arena import run_arena, candidate_is_better
    import hexzero.checkpoint as ckpt_io

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    net     = build_net(cfg, device)
    trainer = Trainer(cfg, net, device)

    best_path = resume_path or ckpt_io.best_checkpoint_path(cfg.checkpoint_dir)
    board_size = cfg.initial_board_size
    if best_path is None:
        print("No checkpoint found — saving initial weights…")
        best_path = trainer.save_checkpoint(0, board_size)
    else:
        trainer.load_checkpoint(best_path)
        saved_size = ckpt_io.load(best_path, device).get("metrics", {}).get("board_size")
        if saved_size and saved_size in cfg.board_sizes:
            board_size = saved_size
        print(f"Resumed from {best_path}  (board {board_size}×{board_size})")
    iteration  = ckpt_io.latest_iteration(cfg.checkpoint_dir)

    try:
        while True:
            iteration += 1
            print(f"\n{'='*60}")
            print(f"Iteration {iteration}  |  board {board_size}×{board_size}")

            print("  Self-play…", end="", flush=True)
            samples = run_self_play_parallel(cfg, best_path, device, board_size, cfg.games_per_iteration)
            for s in samples:
                trainer.replay_buffer.add(s)
            print(f" {len(samples)} samples ({cfg.games_per_iteration} games × 2 with augmentation)")

            print("  Training…", end="", flush=True)
            metrics_list = trainer.train_iteration(iteration, board_size)
            if metrics_list:
                last = metrics_list[-1]
                print(f" loss={last['loss']:.4f}  policy_acc={last['policy_acc']:.3f}")

            cand_path = trainer.save_checkpoint(iteration, board_size)

            print("  Arena…", end="", flush=True)
            cw, chw, draws = run_arena(cand_path, best_path, cfg, board_size)
            total = cw + chw + draws
            print(f" candidate {cw}/{total}  champion {chw}/{total}")

            if candidate_is_better(cw, chw, total, cfg.arena_win_threshold):
                best_path = cand_path
                print("  → New champion accepted.")
            else:
                print("  → Champion retained.")

    except KeyboardInterrupt:
        print("\nTraining interrupted. Checkpoint saved at:", best_path)


def run_gui(cfg: HexZeroConfig, resume_path: str = None) -> None:
    from PyQt6.QtWidgets import QApplication
    from hexzero.gui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Dark palette
    from PyQt6.QtGui import QPalette, QColor
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
            window._net.load_state_dict(data["model_state"])
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
