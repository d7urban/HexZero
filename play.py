"""
play.py — Human vs best checkpoint.

Usage:
    python play.py                           # play as Blue (first move)
    python play.py --color red               # play as Red
    python play.py --sims 400                # stronger AI
    python play.py --board-size 9            # override board size
    python play.py --checkpoint-dir checkpoints2   # use a specific run
"""

import argparse
import io
import subprocess
import sys
import threading
import wave as _wave_mod

import numpy as np
import torch
from PyQt6.QtCore import QObject, Qt, pyqtSignal, pyqtSlot
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


# ---------------------------------------------------------------------------
# Audio  (stdlib-only: wave + aplay)
# ---------------------------------------------------------------------------

def _make_chime_bytes(freq: float, duration: float, decay: float) -> bytes:
    sr = 44100
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    samples = (np.exp(-decay * t) * np.sin(2 * np.pi * freq * t) * 8192).astype(np.int16)
    buf = io.BytesIO()
    with _wave_mod.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(samples.tobytes())
    return buf.getvalue()

_CHIME_SWAP = _make_chime_bytes(880,  0.4, 8.0)   # soft high ding
_CHIME_WIN  = _make_chime_bytes(1047, 0.8, 4.0)   # resonant bell


def _play_chime(wav_bytes: bytes) -> None:
    def _run() -> None:
        try:
            proc = subprocess.Popen(
                ["paplay", "--property=media.role=music", "/dev/stdin"],
                stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
            )
            proc.communicate(wav_bytes, timeout=3)
        except Exception:
            pass
    threading.Thread(target=_run, daemon=True).start()
from config import HexZeroConfig
from hexzero.features import extract_features
from hexzero.game import BLACK, SWAP_MOVE, WHITE, HexState
from hexzero.gui.board_widget import BoardWidget
from hexzero.gui.mcts_widget import MCTSWidget
from hexzero.mcts import MCTSAgent
from hexzero.net import build_net

# ---------------------------------------------------------------------------
# Signal carrier — lives in the main thread; emitted from the worker thread;
# delivered to the main thread via explicit QueuedConnection.
# ---------------------------------------------------------------------------

class _AISignals(QObject):
    move_ready = pyqtSignal(int, object, dict)  # (generation, move, info)
    error      = pyqtSignal(int, str)           # (generation, message)


# ---------------------------------------------------------------------------
# Play window
# ---------------------------------------------------------------------------

class PlayWindow(QMainWindow):
    def __init__(self, cfg: HexZeroConfig, human_color: int, sims: int):
        super().__init__()
        self.cfg              = cfg
        self._human_color     = human_color
        self._sims            = sims
        self._device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._net             = build_net(cfg, self._device, compile=False)
        self._state:  HexState | None   = None
        self._agent:  MCTSAgent | None  = None
        self._ai_thread: threading.Thread | None = None
        self._ai_generation: int = 0          # incremented on cancel; guards stale results
        self._status_override: str | None = None

        # Signal carrier: one instance for the lifetime of the window.
        # QueuedConnection ensures slots run in the main thread regardless of
        # which thread emits the signal.
        self._ai_signals = _AISignals()
        self._ai_signals.move_ready.connect(
            self._on_ai_move, Qt.ConnectionType.QueuedConnection)
        self._ai_signals.error.connect(
            self._on_ai_error, Qt.ConnectionType.QueuedConnection)

        msg = self._load_checkpoint()
        threading.Thread(target=self._warmup_net, daemon=True).start()
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

    def _warmup_net(self) -> None:
        """Forward pass on dummy input so CUDA kernels are compiled before the
        first real inference.  Runs in a daemon thread so startup is instant;
        if the AI turn starts before warm-up finishes, PyTorch serialises the
        two calls internally — no deadlock possible with plain threading.Thread."""
        size = self.cfg.initial_board_size
        with torch.no_grad():
            x = torch.zeros(1, self.cfg.num_input_planes, size, size,
                            dtype=torch.float32, device=self._device)
            s = torch.zeros(1, 1, dtype=torch.float32, device=self._device)
            self._net(x, s)
            if self._device.type == "cuda":
                torch.cuda.synchronize()

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
        # Bump the generation counter so any in-flight result is silently
        # ignored when it arrives via the queued signal.  The daemon thread
        # runs to completion on its own — we don't block on it.
        self._ai_generation += 1
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
            _play_chime(_CHIME_SWAP)

        self._board.set_state(self._state)
        self._refresh_ui()

        if self._state.is_terminal():
            _play_chime(_CHIME_WIN)
        elif self._state.current_player != self._human_color:
            self._start_ai_turn()

    def _start_ai_turn(self) -> None:
        self._refresh_ui()
        gen   = self._ai_generation
        agent = self._agent
        state = self._state
        sigs  = self._ai_signals

        def _run() -> None:
            try:
                pi, _, info = agent.search(state, add_noise=False)
                size = state.size
                idx  = int(np.argmax(pi))
                move = SWAP_MOVE if idx == size * size else (idx // size, idx % size)
                sigs.move_ready.emit(gen, move, info)
            except Exception as exc:
                sigs.error.emit(gen, str(exc))

        self._ai_thread = threading.Thread(target=_run, daemon=True)
        self._ai_thread.start()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    @pyqtSlot(int, int)
    def _on_human_click(self, row: int, col: int) -> None:
        if self._state is None or self._state.is_terminal():
            return
        if self._state.current_player != self._human_color:
            return
        # Clicking the last-placed stone while swap is legal → pie rule swap
        if (self.cfg.use_pie_rule
                and self._state.move_count == 1
                and self._state.last_move == (row, col)):
            self._apply_move(SWAP_MOVE)
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

    @pyqtSlot(int, object, dict)
    def _on_ai_move(self, gen: int, move, info: dict) -> None:
        if gen != self._ai_generation:
            return  # result from a cancelled turn — discard
        if self._state is None or self._state.is_terminal():
            return
        self._mcts_widget.update_info(info)
        if move == SWAP_MOVE:
            new_color = "Red" if self._human_color == WHITE else "Blue"
            self._status_override = f"AI swapped — you now play {new_color}."
        self._apply_move(move)

    @pyqtSlot(int, str)
    def _on_ai_error(self, gen: int, msg: str) -> None:
        if gen != self._ai_generation:
            return
        self._status_lbl.setText(f"AI error: {msg}")

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
            color_name = "Blue" if self._human_color == BLACK else "Red"
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
        self._color_combo.addItem("Blue (first)", BLACK)
        self._color_combo.addItem("Red (second)", WHITE)
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
        self._sims_spin.setRange(10, 100_000)
        self._sims_spin.setValue(self._sims)
        self._sims_spin.setSingleStep(50)
        self._sims_spin.setToolTip("MCTS simulations per AI move (takes effect next game)")
        self._sims_spin.valueChanged.connect(lambda v: setattr(self, "_sims", v))
        toolbar.addWidget(self._sims_spin)

        toolbar.addSeparator()
        self._swap_btn = QPushButton("Swap sides")
        self._swap_btn.setToolTip(
            "Pie rule: take Blue's first stone as your own and play as Blue."
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

        self.setWindowTitle(f"HexZero — Play vs AI  [{self.cfg.checkpoint_dir}]")
        self.resize(1000, 680)

    def closeEvent(self, event) -> None:
        self._cancel_ai()
        event.accept()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Play Hex against the best trained model")
    p.add_argument("--checkpoint-dir", default=None,
                   help="Checkpoint directory to load best.pt from (default: from config)")
    p.add_argument("--board-size", type=int, default=None,
                   help="Board size override (default: from checkpoint)")
    p.add_argument("--sims",       type=int, default=200,
                   help="MCTS simulations per AI move (default: 200)")
    p.add_argument("--color",      choices=["blue", "red"], default="blue",
                   help="Your colour (default: blue)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = HexZeroConfig()

    if args.checkpoint_dir is not None:
        cfg.checkpoint_dir = args.checkpoint_dir
    if args.board_size is not None:
        cfg.initial_board_size = args.board_size

    human_color = BLACK if args.color == "blue" else WHITE

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
