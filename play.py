"""
play.py — Human vs best checkpoint.

Usage:
    python play.py                     # play as Black (first move)
    python play.py --color white       # play as White
    python play.py --sims 400          # stronger AI
    python play.py --board-size 9      # override board size
"""

import argparse
import sys

import numpy as np
import torch
from PyQt6.QtCore import QObject, Qt, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QAction, QColor, QPalette
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QLabel,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QToolBar,
)

import hexzero.checkpoint as ckpt_io
from config import HexZeroConfig
from hexzero.features import extract_features
from hexzero.game import BLACK, SWAP_MOVE, WHITE, HexState
from hexzero.gui.board_widget import BoardWidget
from hexzero.gui.mcts_widget import MCTSWidget
from hexzero.mcts import MCTSAgent
from hexzero.net import build_net

# ---------------------------------------------------------------------------
# Background AI worker
# ---------------------------------------------------------------------------

class AIWorker(QObject):
    """Runs one MCTS search in a background thread."""
    move_ready = pyqtSignal(object, dict)   # (move, mcts_info)

    def __init__(self, agent: MCTSAgent, state: HexState):
        super().__init__()
        self._agent = agent
        self._state = state

    @pyqtSlot()
    def run(self) -> None:
        pi, _, info = self._agent.search(self._state, add_noise=False)
        size = self._state.size
        idx  = int(np.argmax(pi))
        move = SWAP_MOVE if idx == size * size else (idx // size, idx % size)
        self.move_ready.emit(move, info)


# ---------------------------------------------------------------------------
# Play window
# ---------------------------------------------------------------------------

class PlayWindow(QMainWindow):
    def __init__(self, cfg: HexZeroConfig, human_color: int, sims: int):
        super().__init__()
        self.cfg           = cfg
        self._human_color  = human_color
        self._sims         = sims
        self._device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._net          = build_net(cfg, self._device)
        self._state:     HexState | None  = None
        self._agent:     MCTSAgent | None = None
        self._ai_thread: QThread | None   = None
        self._status_override: str | None = None  # shown once in _refresh_ui

        msg = self._load_checkpoint()
        self._build_ui()
        self._status_lbl.setText(msg or "Ready")
        self._new_game()

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def _load_checkpoint(self, path: str | None = None) -> str:
        if path is None:
            path = ckpt_io.best_checkpoint_path(self.cfg.checkpoint_dir)
        if path is None:
            return "No checkpoint found — AI plays with random weights."
        try:
            data = ckpt_io.load(path, self._device)
            ckpt_io.load_weights(self._net, data["model_state"])
            board_size = data.get("metrics", {}).get("board_size")
            if board_size and board_size in self.cfg.board_sizes:
                self.cfg.initial_board_size = board_size
            iteration = data.get("iteration", "?")
            return (f"Loaded iter {iteration}  "
                    f"board {self.cfg.initial_board_size}×{self.cfg.initial_board_size}")
        except Exception as e:
            return f"Could not load checkpoint: {e}"
        finally:
            self._net.eval()

    def _make_infer_fn(self):
        net, device = self._net, self._device
        def infer_fn(state: HexState):
            feat, size_norm = extract_features(state)
            x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(device)
            s = torch.tensor([[size_norm]], dtype=torch.float32).to(device)
            with torch.no_grad():
                log_pi, v = net(x, s)
            return torch.exp(log_pi).squeeze(0).cpu().numpy(), float(v.item())
        return infer_fn

    # ------------------------------------------------------------------
    # Game control
    # ------------------------------------------------------------------

    def _cancel_ai(self) -> None:
        if self._ai_thread and self._ai_thread.isRunning():
            self._ai_thread.quit()
            self._ai_thread.wait(2000)
        self._ai_thread = None

    def _new_game(self) -> None:
        self._cancel_ai()
        self._status_override = None
        size = self.cfg.initial_board_size
        self._state = HexState(size, pie_rule=self.cfg.use_pie_rule)
        self._agent = MCTSAgent(
            infer_fn=self._make_infer_fn(),
            simulations=self._sims,
            cpuct=self.cfg.cpuct,
            dirichlet_epsilon=0.0,          # no noise in play mode
            temperature=1.0,
            temperature_moves=self.cfg.temperature_moves,
        )
        self._board.set_state(self._state)
        self._mcts_widget.clear()
        self._refresh_ui()
        if self._state.current_player != self._human_color:
            self._start_ai_turn()

    def _apply_move(self, move) -> None:
        self._agent.update_root(move)
        self._state.apply_move(move)

        if move == SWAP_MOVE:
            # Both players' identities switch; keep tracking consistent.
            self._human_color = -self._human_color
            self._color_combo.blockSignals(True)
            self._color_combo.setCurrentIndex(0 if self._human_color == BLACK else 1)
            self._color_combo.blockSignals(False)

        self._board.set_state(self._state)
        self._refresh_ui()

        if not self._state.is_terminal() and self._state.current_player != self._human_color:
            self._start_ai_turn()

    def _start_ai_turn(self) -> None:
        self._refresh_ui()
        worker = AIWorker(self._agent, self._state)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.move_ready.connect(self._on_ai_move)
        worker.move_ready.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self._ai_thread = thread
        thread.start()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    @pyqtSlot(int, int)
    def _on_human_click(self, row: int, col: int) -> None:
        if self._state is None or self._state.is_terminal():
            return
        if self._state.current_player != self._human_color:
            return
        if (row, col) not in self._state.legal_moves():
            return
        self._apply_move((row, col))

    @pyqtSlot()
    def _on_swap_clicked(self) -> None:
        if self._state is None or self._state.current_player != self._human_color:
            return
        if not (self.cfg.use_pie_rule and self._state.move_count == 1):
            return
        self._apply_move(SWAP_MOVE)

    @pyqtSlot(object, dict)
    def _on_ai_move(self, move, info: dict) -> None:
        if self._state is None or self._state.is_terminal():
            return
        self._mcts_widget.update_info(info)
        if move == SWAP_MOVE:
            # Announce the swap; _refresh_ui will use this message once.
            new_color = "White" if self._human_color == WHITE else "Black"
            self._status_override = f"AI swapped — you now play {new_color}."
        self._apply_move(move)

    @pyqtSlot(int)
    def _on_size_changed(self, idx: int) -> None:
        self.cfg.initial_board_size = self._size_combo.itemData(idx)
        self._new_game()

    @pyqtSlot(int)
    def _on_color_changed(self, idx: int) -> None:
        self._human_color = self._color_combo.itemData(idx)
        self._new_game()

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------

    def _refresh_ui(self) -> None:
        if self._state is None:
            return

        is_terminal = self._state.is_terminal()
        is_human    = not is_terminal and self._state.current_player == self._human_color
        can_swap    = (is_human and self.cfg.use_pie_rule
                       and self._state.move_count == 1)
        self._swap_btn.setVisible(can_swap)

        # Status text — override takes priority (used for AI swap announcement)
        if self._status_override:
            self._status_lbl.setText(self._status_override)
            self._status_override = None
        elif is_terminal:
            winner = self._state.winner()
            if winner == self._human_color:
                self._status_lbl.setText("You win!")
            else:
                self._status_lbl.setText("AI wins.")
        elif is_human:
            color_name = "Black" if self._human_color == BLACK else "White"
            hint = " (or click Swap)" if can_swap else ""
            self._status_lbl.setText(f"Your turn ({color_name}){hint} — click a cell.")
        else:
            self._status_lbl.setText("AI is thinking…")

    def _build_ui(self) -> None:
        # ---- Toolbar ----
        toolbar = QToolBar("Controls")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        act_new = QAction("New Game", self)
        act_new.setShortcut("Ctrl+N")
        act_new.triggered.connect(self._new_game)
        toolbar.addAction(act_new)

        toolbar.addSeparator()
        toolbar.addWidget(QLabel("  Play as: "))
        self._color_combo = QComboBox()
        self._color_combo.addItem("Black (first)", BLACK)
        self._color_combo.addItem("White (second)", WHITE)
        self._color_combo.setCurrentIndex(0 if self._human_color == BLACK else 1)
        self._color_combo.currentIndexChanged.connect(self._on_color_changed)
        toolbar.addWidget(self._color_combo)

        toolbar.addSeparator()
        toolbar.addWidget(QLabel("  Board: "))
        self._size_combo = QComboBox()
        for s in self.cfg.board_sizes:
            self._size_combo.addItem(f"{s}×{s}", s)
        cur = self.cfg.initial_board_size
        self._size_combo.setCurrentIndex(
            self.cfg.board_sizes.index(cur) if cur in self.cfg.board_sizes else 0
        )
        self._size_combo.currentIndexChanged.connect(self._on_size_changed)
        toolbar.addWidget(self._size_combo)

        toolbar.addSeparator()
        toolbar.addWidget(QLabel("  AI sims: "))
        self._sims_spin = QSpinBox()
        self._sims_spin.setRange(10, 2000)
        self._sims_spin.setValue(self._sims)
        self._sims_spin.setSingleStep(50)
        self._sims_spin.setToolTip("MCTS simulations per AI move (takes effect next game)")
        self._sims_spin.valueChanged.connect(lambda v: setattr(self, "_sims", v))
        toolbar.addWidget(self._sims_spin)

        toolbar.addSeparator()
        self._swap_btn = QPushButton("Swap sides")
        self._swap_btn.setToolTip(
            "Pie rule: take Black's first stone as your own and play as Black."
        )
        self._swap_btn.setVisible(False)
        self._swap_btn.clicked.connect(self._on_swap_clicked)
        toolbar.addWidget(self._swap_btn)

        # ---- Central: board | MCTS info ----
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self._board = BoardWidget()
        self._board.move_clicked.connect(self._on_human_click)
        splitter.addWidget(self._board)

        self._mcts_widget = MCTSWidget()
        splitter.addWidget(self._mcts_widget)
        splitter.setSizes([660, 340])

        self.setCentralWidget(splitter)

        # ---- Status bar ----
        status_bar = QStatusBar()
        self._status_lbl = QLabel("Ready")
        status_bar.addWidget(self._status_lbl)
        status_bar.addPermanentWidget(QLabel(f"  Device: {self._device}"))
        self.setStatusBar(status_bar)

        self.setWindowTitle("HexZero — Play vs AI")
        self.resize(1000, 680)

    def closeEvent(self, event) -> None:
        self._cancel_ai()
        event.accept()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Play Hex against the best trained model")
    p.add_argument("--board-size", type=int, default=None,
                   help="Board size override (default: from checkpoint)")
    p.add_argument("--sims",       type=int, default=200,
                   help="MCTS simulations per AI move (default: 200)")
    p.add_argument("--color",      choices=["black", "white"], default="black",
                   help="Your colour (default: black)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = HexZeroConfig()

    if args.board_size is not None:
        cfg.initial_board_size = args.board_size

    human_color = BLACK if args.color == "black" else WHITE

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window,           QColor(45,  45,  45))
    palette.setColor(QPalette.ColorRole.WindowText,       QColor(210, 210, 210))
    palette.setColor(QPalette.ColorRole.Base,             QColor(30,  30,  30))
    palette.setColor(QPalette.ColorRole.AlternateBase,    QColor(45,  45,  45))
    palette.setColor(QPalette.ColorRole.Text,             QColor(210, 210, 210))
    palette.setColor(QPalette.ColorRole.Button,           QColor(60,  60,  60))
    palette.setColor(QPalette.ColorRole.ButtonText,       QColor(210, 210, 210))
    palette.setColor(QPalette.ColorRole.Highlight,        QColor(80,  120, 200))
    palette.setColor(QPalette.ColorRole.HighlightedText,  QColor(255, 255, 255))
    app.setPalette(palette)

    window = PlayWindow(cfg, human_color, sims=args.sims)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
