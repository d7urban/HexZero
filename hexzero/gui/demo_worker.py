"""
DemoWorker: a background QThread that continuously plays Hex games with the
current best model and emits game_step signals so the board stays live.

Behaviour:
  - Reloads the best checkpoint at the start of each new game, so the board
    automatically reflects the latest champion after each training iteration.
  - Falls back to a fresh (random-policy) network if no checkpoint exists yet,
    so the board shows something from the moment Start Training is clicked.
  - Uses 50 MCTS simulations (not the full training budget) for display speed.
  - Pauses briefly between moves so the human eye can follow the game.
"""

import numpy as np
import torch
from PyQt6.QtCore import QThread

from config import HexZeroConfig
from hexzero.features import extract_features
from hexzero.game import HexState, SWAP_MOVE
from hexzero.mcts import MCTSAgent
from hexzero.net import build_net
import hexzero.checkpoint as ckpt_io

_DISPLAY_SIMS   = 25     # MCTS simulations per displayed move
_MOVE_DELAY_MS  = 650    # pause between moves (ms)
_GAME_PAUSE_MS  = 1500   # pause after a game ends before starting the next


class DemoWorker(QThread):
    def __init__(self, cfg: HexZeroConfig, signals, parent=None):
        super().__init__(parent)
        self.cfg     = cfg
        self.signals = signals
        self._stop   = False
        self._size   = cfg.initial_board_size

    def set_board_size(self, size: int) -> None:
        self._size = size

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            self._run_loop(device)
        except Exception as exc:
            self.signals.status_message.emit(f"Demo worker crashed: {exc}")

    def _run_loop(self, device) -> None:
        while not self._stop:
            # ----------------------------------------------------------
            # Load current best model (or untrained weights if none yet)
            # ----------------------------------------------------------
            net = build_net(self.cfg, device)
            best_path = ckpt_io.best_checkpoint_path(self.cfg.checkpoint_dir)
            if best_path:
                try:
                    data = ckpt_io.load(best_path, device)
                    ckpt_io.load_weights(net, data["model_state"])
                except Exception:
                    pass  # checkpoint unreadable mid-write; use current weights
            net.eval()

            def infer_fn(state: HexState, _net=net, _dev=device):
                feat, size_norm = extract_features(state)
                x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(_dev)
                s = torch.tensor([[size_norm]], dtype=torch.float32).to(_dev)
                with torch.no_grad():
                    log_pi, v = _net(x, s)
                return torch.exp(log_pi).squeeze(0).cpu().numpy(), float(v.item())

            agent = MCTSAgent(
                infer_fn,
                simulations=_DISPLAY_SIMS,
                dirichlet_epsilon=0.0,   # no noise; greedy display
                temperature=1.0,
                temperature_moves=self.cfg.temperature_moves,
            )
            state = HexState(self._size, pie_rule=self.cfg.use_pie_rule)

            # ----------------------------------------------------------
            # Play one full game, emitting a signal after each move
            # ----------------------------------------------------------
            while not state.is_terminal() and not self._stop:
                pi, _, info = agent.search(state, add_noise=False)

                self.signals.game_step.emit(state.board.copy(), pi.copy(), info)

                idx  = int(np.argmax(pi))
                size = state.size
                move = SWAP_MOVE if idx == size * size else (idx // size, idx % size)
                agent.update_root(move)
                state.apply_move(move)
                self.msleep(_MOVE_DELAY_MS)

            if state.is_terminal() and not self._stop:
                # Show the finished board for a moment
                empty_pi = np.zeros(state.size ** 2, dtype=np.float32)
                self.signals.game_step.emit(state.board.copy(), empty_pi, {})
                self.signals.game_finished.emit(state.winner() or 0, state.move_count)
                self.msleep(_GAME_PAUSE_MS)
