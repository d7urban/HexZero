"""
HexZero — AlphaZero-style self-play training for Hex.

Usage:
    python main.py                                   # Launch GUI
    python main.py --headless                        # Run training without GUI
    python main.py --board-size 9                    # Override initial board size
    python main.py --checkpoint path                 # Resume from checkpoint
    python main.py --checkpoint-dir checkpoints2     # Parallel experiment with separate state
"""

import argparse
import os
import sys

from config import HexZeroConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HexZero training suite")
    p.add_argument("--headless",        action="store_true",  help="Run without GUI")
    p.add_argument("--board-size",      type=int, default=None, help="Initial board size")
    p.add_argument("--checkpoint",      type=str, default=None, help="Path to checkpoint to load")
    p.add_argument("--checkpoint-dir",  type=str, default=None,
                   help="Checkpoint directory (default: checkpoints). Use a different dir to run parallel experiments.")
    p.add_argument("--simulations",     type=int, default=None,
                   help="MCTS simulations for 7×7 (default: 50); larger boards scale proportionally")
    p.add_argument("--workers",         type=int, default=None, help="Self-play worker threads")
    return p.parse_args()


def run_headless(cfg: HexZeroConfig, resume_path: str = None) -> None:
    import torch

    import hexzero.checkpoint as ckpt_io
    from hexzero.net import build_net
    from hexzero.trainer import LoopCallbacks, Trainer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    net     = build_net(cfg, device)
    trainer = Trainer(cfg, net, device)

    initial_path = None
    if resume_path:
        trainer.load_checkpoint(resume_path)
        initial_path = resume_path
        print(f"Resuming from {resume_path}")

    class _PrintCallbacks(LoopCallbacks):
        def on_iteration_start(self, i, bs):
            print(f"\n{'='*60}")
            print(f"Iteration {i}  |  board {bs}×{bs}")
        def on_self_play_done(self, n, sw, gp):
            swap_info = f"  swap {sw}/{gp} ({100*sw//gp}%)" if gp > 0 else ""
            print(f"  Self-play: {n} samples ({gp} games × 2){swap_info}")
        def on_train_done(self, ml):
            if ml:
                last = ml[-1]
                print(f"  Training: loss={last['loss']:.4f}  policy_acc={last['policy_acc']:.3f}")
        def on_arena_done(self, cw, chw, draws):
            print(f"  Arena: candidate {cw}/{cw+chw+draws}  champion {chw}/{cw+chw+draws}")
        def on_promoted(self, path):
            print("  → New champion accepted.")
        def on_champion_retained(self):
            print("  → Champion retained.")
        def on_board_size_advanced(self, bs, reason, promoted, iteration):
            prefix = "new champion + " if promoted else ""
            print(f"  → {prefix}Curriculum advanced to {bs}×{bs}! ({reason})")

    try:
        trainer.run_loop(callbacks=_PrintCallbacks(), initial_path=initial_path)
    except KeyboardInterrupt:
        print("\nTraining interrupted.")


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
    if args.checkpoint_dir is not None:
        cfg.checkpoint_dir = args.checkpoint_dir
    if args.simulations is not None:
        cfg.mcts_simulations = args.simulations
        # Scale all per-size entries proportionally from the 7×7 base.
        base = cfg.mcts_simulations_per_size[0] if cfg.mcts_simulations_per_size else 50
        if base > 0 and cfg.mcts_simulations_per_size:
            cfg.mcts_simulations_per_size = [
                max(1, round(s * args.simulations / base))
                for s in cfg.mcts_simulations_per_size
            ]
    if args.workers is not None:
        cfg.num_self_play_workers = args.workers

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    if args.headless:
        run_headless(cfg, resume_path=args.checkpoint)
    else:
        run_gui(cfg, resume_path=args.checkpoint)


if __name__ == "__main__":
    main()
