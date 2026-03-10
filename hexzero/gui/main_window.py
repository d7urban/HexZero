"""
MainWindow: assembles the full GUI.

Layout:
  ┌─────────────────┬──────────────────────┐
  │   BoardWidget   │   ChartWidget        │
  │   (left, 60%)   │   (right-top, 60%)   │
  │                 ├──────────────────────┤
  │                 │   MCTSWidget         │
  │                 │   (right-bottom, 40%)│
  └─────────────────┴──────────────────────┘
  [ StatsWidget: Iter | Phase progress | Buffer | Arena ]
  [Status bar]

The DemoWorker thread drives the board continuously from the moment
Start Training is clicked, independent of the training loop.
The TrainingWorker thread runs self-play → train → arena in the background.
"""

import os
import torch

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QSplitter,
    QToolBar, QStatusBar, QLabel, QFileDialog,
    QComboBox, QSpinBox, QMessageBox,
)
from PyQt6.QtCore import Qt, QThread, pyqtSlot, QObject, pyqtSignal
from PyQt6.QtGui import QAction

from config import HexZeroConfig
from hexzero.game import HexState
from hexzero.net import build_net
from hexzero.trainer import Trainer
from hexzero.gui.signals import TrainingSignals
from hexzero.gui.board_widget import BoardWidget
from hexzero.gui.chart_widget import ChartWidget
from hexzero.gui.mcts_widget import MCTSWidget
from hexzero.gui.stats_widget import StatsWidget
from hexzero.gui.demo_worker import DemoWorker
import hexzero.checkpoint as ckpt_io


# ---------------------------------------------------------------------------
# Training worker thread
# ---------------------------------------------------------------------------

class TrainingWorker(QObject):
    finished = pyqtSignal()

    def __init__(self, trainer: Trainer, cfg: HexZeroConfig, signals: TrainingSignals):
        super().__init__()
        self.trainer    = trainer
        self.cfg        = cfg
        self.signals    = signals
        self._stop      = False
        self._iteration = 0

    def stop(self) -> None:
        self._stop = True

    @pyqtSlot()
    def run(self) -> None:
        from hexzero.self_play import run_self_play_parallel
        from hexzero.arena import run_arena, candidate_is_better

        cfg        = self.cfg
        board_size = cfg.initial_board_size
        size_idx   = (cfg.board_sizes.index(board_size)
                      if board_size in cfg.board_sizes else 0)

        # Bootstrap: save initial checkpoint if none exists yet
        best_path = ckpt_io.best_checkpoint_path(cfg.checkpoint_dir)
        if best_path is None:
            self.signals.status_message.emit("Saving initial checkpoint…")
            best_path = self.trainer.save_checkpoint(0)
            self.signals.checkpoint_saved.emit(best_path)

        while not self._stop:
            self._iteration += 1
            self.signals.iteration_started.emit(self._iteration)

            # ----------------------------------------------------------
            # 1. Self-play
            # ----------------------------------------------------------
            self.signals.status_message.emit(
                f"Iteration {self._iteration} — self-play ({board_size}×{board_size})…"
            )
            self.signals.self_play_progress.emit(0, cfg.games_per_iteration)

            def _sp_progress(done: int, total: int) -> None:
                self.signals.self_play_progress.emit(done, total)
                self.signals.buffer_updated.emit(len(self.trainer.replay_buffer))

            samples = run_self_play_parallel(
                cfg, best_path, board_size, cfg.games_per_iteration,
                progress_callback=_sp_progress,
            )
            for s in samples:
                self.trainer.replay_buffer.add(s)
            self.signals.buffer_updated.emit(len(self.trainer.replay_buffer))

            if self._stop:
                break

            # ----------------------------------------------------------
            # 2. Train
            # ----------------------------------------------------------
            self.signals.status_message.emit(
                f"Iteration {self._iteration} — training…"
            )
            self.trainer.train_iteration(self._iteration, board_size)

            if self._stop:
                break

            # ----------------------------------------------------------
            # 3. Save candidate checkpoint
            # ----------------------------------------------------------
            cand_path = self.trainer.save_checkpoint(self._iteration)

            # ----------------------------------------------------------
            # 4. Arena
            # ----------------------------------------------------------
            self.signals.status_message.emit(
                f"Iteration {self._iteration} — arena…"
            )
            cw, chw, draws = run_arena(cand_path, best_path, cfg, board_size)
            self.signals.arena_result.emit(cw, chw, draws)

            total    = cw + chw + draws
            win_rate = cw / total if total > 0 else 0.0

            if candidate_is_better(cw, chw, total, cfg.arena_win_threshold):
                best_path = cand_path
                self.signals.checkpoint_saved.emit(best_path)

                # Curriculum: advance board size if win rate clears the threshold
                next_idx = size_idx + 1
                if win_rate >= cfg.curriculum_threshold and next_idx < len(cfg.board_sizes):
                    size_idx   = next_idx
                    board_size = cfg.board_sizes[size_idx]
                    self.signals.board_size_advanced.emit(board_size)
                    self.signals.status_message.emit(
                        f"Iteration {self._iteration} — curriculum advanced to "
                        f"{board_size}×{board_size}! ({cw}/{total}, {win_rate:.0%})"
                    )
                else:
                    self.signals.status_message.emit(
                        f"Iteration {self._iteration} — new champion! "
                        f"({cw}/{total}, {win_rate:.0%})"
                    )
            else:
                self.signals.status_message.emit(
                    f"Iteration {self._iteration} — champion retained "
                    f"({chw}/{total}, {win_rate:.0%} for candidate)"
                )

            self.signals.iteration_finished.emit(self._iteration)

        self.finished.emit()


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self, cfg: HexZeroConfig):
        super().__init__()
        self.cfg     = cfg
        self.signals = TrainingSignals()

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._net    = build_net(cfg, self._device)

        self._trainer:      Trainer | None        = None
        self._worker:       TrainingWorker | None = None
        self._train_thread: QThread | None        = None
        self._demo:         DemoWorker | None     = None

        self._build_ui()
        self._connect_signals()
        self.setWindowTitle("HexZero — AlphaZero for Hex")
        self.resize(1200, 740)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        # ------ main splitter: board | right column ------
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(4)

        self._board = BoardWidget()
        splitter.addWidget(self._board)

        right_splitter = QSplitter(Qt.Orientation.Vertical)
        right_splitter.setHandleWidth(4)

        self._chart = ChartWidget()
        right_splitter.addWidget(self._chart)

        self._mcts = MCTSWidget()
        right_splitter.addWidget(self._mcts)
        right_splitter.setSizes([420, 280])

        splitter.addWidget(right_splitter)
        splitter.setSizes([540, 660])

        # ------ stats bar below the splitter ------
        self._stats = StatsWidget(sizes=self.cfg.board_sizes)

        # ------ central widget: splitter + stats ------
        central = QWidget()
        layout  = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(splitter, stretch=1)
        layout.addWidget(self._stats)
        self.setCentralWidget(central)

        # ------ toolbar ------
        toolbar = QToolBar("Controls")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        self._act_start = QAction("▶ Start Training", self)
        self._act_start.setToolTip("Begin self-play training loop")
        toolbar.addAction(self._act_start)

        self._act_stop = QAction("■ Stop", self)
        self._act_stop.setEnabled(False)
        toolbar.addAction(self._act_stop)

        toolbar.addSeparator()

        self._act_load = QAction("Load Checkpoint", self)
        toolbar.addAction(self._act_load)

        toolbar.addSeparator()

        toolbar.addWidget(QLabel("  Board size: "))
        self._size_combo = QComboBox()
        for s in self.cfg.board_sizes:
            self._size_combo.addItem(f"{s}×{s}", s)
        self._size_combo.setCurrentIndex(
            self.cfg.board_sizes.index(self.cfg.initial_board_size)
            if self.cfg.initial_board_size in self.cfg.board_sizes else 0
        )
        toolbar.addWidget(self._size_combo)

        toolbar.addSeparator()
        toolbar.addWidget(QLabel("  Training sims: "))
        self._sims_spin = QSpinBox()
        self._sims_spin.setRange(10, 2000)
        self._sims_spin.setValue(self.cfg.mcts_simulations)
        self._sims_spin.setSingleStep(50)
        self._sims_spin.setToolTip(
            "MCTS simulations per move during self-play and arena.\n"
            "The live board demo always uses 50 sims for display speed."
        )
        toolbar.addWidget(self._sims_spin)

        # ------ status bar ------
        self._status_bar   = QStatusBar()
        self._status_label = QLabel("Ready")
        self._status_bar.addWidget(self._status_label)
        self._device_label = QLabel(f"  Device: {self._device}")
        self._status_bar.addPermanentWidget(self._device_label)
        self.setStatusBar(self._status_bar)

        # Initial board display
        self._board.set_state(HexState(self.cfg.initial_board_size))

    # ------------------------------------------------------------------
    # Signal wiring
    # ------------------------------------------------------------------

    def _connect_signals(self) -> None:
        self._act_start.triggered.connect(self._start_training)
        self._act_stop.triggered.connect(self._stop_training)
        self._act_load.triggered.connect(self._load_checkpoint)
        self._size_combo.currentIndexChanged.connect(self._on_size_changed)
        self._sims_spin.valueChanged.connect(self._on_sims_changed)

        sig = self.signals
        sig.metrics_updated.connect(self._chart.update_metrics)
        sig.game_step.connect(self._on_game_step)
        sig.arena_result.connect(self._on_arena_result)
        sig.checkpoint_saved.connect(self._on_checkpoint_saved)
        sig.status_message.connect(self._status_label.setText)

        # Stats bar
        sig.iteration_started.connect(self._stats.on_iteration_started)
        sig.self_play_progress.connect(self._stats.on_self_play_progress)
        sig.arena_result.connect(self._stats.on_arena_result)
        sig.buffer_updated.connect(self._stats.on_buffer_updated)
        sig.metrics_updated.connect(self._stats.on_metrics)
        sig.board_size_advanced.connect(self._on_size_advanced)
        sig.board_size_advanced.connect(self._stats.on_board_size_advanced)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    @pyqtSlot()
    def _start_training(self) -> None:
        self._act_start.setEnabled(False)
        self._act_stop.setEnabled(True)

        self._trainer = Trainer(self.cfg, self._net, self._device, self.signals)

        best = ckpt_io.best_checkpoint_path(self.cfg.checkpoint_dir)
        if best:
            try:
                self._trainer.load_checkpoint(best)
                self._status_label.setText(f"Loaded checkpoint: {os.path.basename(best)}")
            except Exception as e:
                self._status_label.setText(f"Could not load checkpoint: {e}")

        # Training thread
        self._worker = TrainingWorker(self._trainer, self.cfg, self.signals)
        self._train_thread = QThread()
        self._worker.moveToThread(self._train_thread)
        self._train_thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._train_thread.quit)
        self._worker.finished.connect(self._on_training_finished)
        self._train_thread.start()

        # Demo thread — starts immediately so the board is never blank
        self._demo = DemoWorker(self.cfg, self.signals)
        self._demo.start()

    @pyqtSlot()
    def _stop_training(self) -> None:
        if self._worker:
            self._worker.stop()
        if self._demo:
            self._demo.stop()
        self._act_stop.setEnabled(False)
        self._status_label.setText("Stopping after current iteration…")

    @pyqtSlot()
    def _on_training_finished(self) -> None:
        self._act_start.setEnabled(True)
        self._act_stop.setEnabled(False)
        self._status_label.setText("Training stopped.")

    @pyqtSlot()
    def _load_checkpoint(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Checkpoint", self.cfg.checkpoint_dir, "PyTorch (*.pt)"
        )
        if not path:
            return
        try:
            data = ckpt_io.load(path, self._device)
            self._net.load_state_dict(data["model_state"])
            self._status_label.setText(f"Loaded: {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.critical(self, "Load failed", str(e))

    @pyqtSlot(object, object, dict)
    def _on_game_step(self, board_array, policy, mcts_info) -> None:
        size  = int(board_array.shape[0])
        state = HexState(size)
        state.board[:] = board_array
        self._board.set_state(state, policy)
        if mcts_info:
            self._mcts.update_info(mcts_info)

    @pyqtSlot(int, int, int)
    def _on_arena_result(self, cw: int, chw: int, draws: int) -> None:
        total = cw + chw + draws
        self._status_label.setText(
            f"Arena: candidate {cw}/{total}  champion {chw}/{total}  draws {draws}"
        )

    @pyqtSlot(str)
    def _on_checkpoint_saved(self, path: str) -> None:
        self._status_label.setText(f"Checkpoint saved: {os.path.basename(path)}")

    @pyqtSlot(int)
    def _on_size_changed(self, idx: int) -> None:
        size = self._size_combo.itemData(idx)
        self.cfg.initial_board_size = size
        self._board.set_state(HexState(size))
        if self._demo:
            self._demo.set_board_size(size)

    @pyqtSlot(int)
    def _on_size_advanced(self, size: int) -> None:
        """Curriculum advanced — sync the combo box and demo worker."""
        idx = self._size_combo.findData(size)
        if idx >= 0:
            # Block the signal so _on_size_changed doesn't also fire
            self._size_combo.blockSignals(True)
            self._size_combo.setCurrentIndex(idx)
            self._size_combo.blockSignals(False)
        self._board.set_state(HexState(size))
        if self._demo:
            self._demo.set_board_size(size)

    @pyqtSlot(int)
    def _on_sims_changed(self, value: int) -> None:
        self.cfg.mcts_simulations = value

    # ------------------------------------------------------------------
    # Close
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        if self._worker:
            self._worker.stop()
        if self._demo:
            self._demo.stop()
        for t in (self._train_thread, self._demo):
            if t and t.isRunning():
                t.quit()
                t.wait(2000)
        event.accept()
