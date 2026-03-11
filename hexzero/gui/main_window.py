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
import threading

import torch
from PyQt6.QtCore import QObject, Qt, QThread, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

import hexzero.checkpoint as ckpt_io
from config import HexZeroConfig
from hexzero.game import HexState
from hexzero.gui.board_widget import BoardWidget
from hexzero.gui.chart_widget import ChartWidget
from hexzero.gui.demo_worker import DemoWorker
from hexzero.gui.mcts_widget import MCTSWidget
from hexzero.gui.signals import TrainingSignals
from hexzero.gui.stats_widget import StatsWidget
from hexzero.net import build_net
from hexzero.trainer import Trainer


def _loss_plateau(losses: list[float], window: int, threshold: float) -> bool:
    """Return True when policy loss has stopped improving significantly.

    Compares the first and last value of the most recent `window` per-iteration
    losses.  If the relative improvement is below `threshold` the network has
    plateaued on the current board size.
    """
    if len(losses) < window:
        return False
    recent = losses[-window:]
    start  = recent[0]
    if start <= 0:
        return True
    return (start - recent[-1]) / start < threshold


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
        self._stop_event = threading.Event()
        self._iteration  = 0

    def stop(self) -> None:
        self._stop_event.set()

    @pyqtSlot()
    def run(self) -> None:
        from hexzero.arena import candidate_is_better, run_arena
        from hexzero.self_play import run_self_play_parallel

        cfg        = self.cfg
        board_size = cfg.initial_board_size
        size_idx   = (cfg.board_sizes.index(board_size)
                      if board_size in cfg.board_sizes else 0)

        # Bootstrap: save initial checkpoint if none exists yet
        best_path = ckpt_io.best_checkpoint_path(cfg.checkpoint_dir)
        if best_path is None:
            self.signals.status_message.emit("Saving initial checkpoint…")
            best_path = self.trainer.save_checkpoint(0, board_size)
            ckpt_io.promote_to_best(best_path, cfg.checkpoint_dir)
            self.signals.checkpoint_saved.emit(best_path)

        # Resume iteration counter from disk so we never overwrite / prune a
        # freshly-saved candidate checkpoint on the very first save.
        self._iteration = ckpt_io.latest_iteration(cfg.checkpoint_dir)

        # Restore replay buffer from disk if available
        buf_path = os.path.join(cfg.checkpoint_dir, "replay_buffer.pt.gz")
        if os.path.exists(buf_path):
            try:
                self.signals.status_message.emit("Loading replay buffer…")
                self.trainer.replay_buffer.load(buf_path)
                self.signals.buffer_updated.emit(len(self.trainer.replay_buffer))
            except Exception as e:
                self.signals.status_message.emit(f"Could not load replay buffer: {e}")

        # Restore board size and per-size iteration count from persisted state
        ts = ckpt_io.load_training_state(cfg.checkpoint_dir)
        if ts.get("board_size") and ts["board_size"] in cfg.board_sizes:
            board_size = ts["board_size"]
            size_idx   = cfg.board_sizes.index(board_size)
        iters_on_size = ts.get("iters_on_size", 0)
        size_losses: list[float] = ts.get("size_losses", [])

        stop = self._stop_event
        while not stop.is_set():
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

            device = self.trainer.device
            samples, swap_games, games_played = run_self_play_parallel(
                cfg, best_path, device, board_size, cfg.games_per_iteration,
                progress_callback=_sp_progress,
                stop_event=stop,
            )
            self.signals.swap_rate_updated.emit(swap_games, games_played)
            for s in samples:
                self.trainer.replay_buffer.add(s)
            self.signals.buffer_updated.emit(len(self.trainer.replay_buffer))

            # Persist buffer so it survives restarts
            try:
                self.trainer.replay_buffer.save(buf_path)
            except Exception:
                pass  # non-fatal; training continues without saving

            if stop.is_set():
                break

            # ----------------------------------------------------------
            # 2. Train
            # ----------------------------------------------------------
            self.signals.status_message.emit(
                f"Iteration {self._iteration} — training…"
            )
            metrics_list = self.trainer.train_iteration(self._iteration, board_size)
            if metrics_list:
                mean_loss = sum(m["policy_loss"] for m in metrics_list) / len(metrics_list)
                size_losses.append(mean_loss)

            if stop.is_set():
                break

            # ----------------------------------------------------------
            # 3. Save candidate checkpoint
            # ----------------------------------------------------------
            cand_path = self.trainer.save_checkpoint(self._iteration, board_size)

            # ----------------------------------------------------------
            # 4. Arena
            # ----------------------------------------------------------
            self.signals.status_message.emit(
                f"Iteration {self._iteration} — arena…"
            )
            def _arena_progress(done: int, cw_so_far: int, total: int) -> None:
                self.signals.arena_progress.emit(done, cw_so_far, total)

            cw, chw, draws = run_arena(cand_path, best_path, cfg, board_size,
                                       progress_callback=_arena_progress,
                                       stop_event=stop)

            if stop.is_set():
                break

            self.signals.arena_result.emit(cw, chw, draws)

            total    = cw + chw + draws
            win_rate = cw / total if total > 0 else 0.0

            iters_on_size += 1
            self.signals.curriculum_progress.emit(iters_on_size, cfg.min_iters_per_size)
            ckpt_io.save_training_state(cfg.checkpoint_dir, {
                "iteration":    self._iteration,
                "board_size":   board_size,
                "size_idx":     size_idx,
                "iters_on_size": iters_on_size,
                "size_losses":  size_losses,
            })

            if candidate_is_better(cw, chw, total, cfg.arena_win_threshold):
                best_path = cand_path
                ckpt_io.promote_to_best(best_path, cfg.checkpoint_dir)
                self.signals.checkpoint_saved.emit(best_path)

                next_idx = size_idx + 1
                can_advance = (
                    iters_on_size >= cfg.min_iters_per_size
                    and _loss_plateau(size_losses, cfg.min_iters_per_size,
                                      cfg.loss_plateau_threshold)
                    and next_idx < len(cfg.board_sizes)
                )
                if can_advance:
                    size_idx      = next_idx
                    board_size    = cfg.board_sizes[size_idx]
                    iters_on_size = 0
                    size_losses   = []
                    self.trainer.reset_lr()
                    self.signals.board_size_advanced.emit(board_size)
                    self.signals.status_message.emit(
                        f"Iteration {self._iteration} — curriculum advanced to "
                        f"{board_size}×{board_size}! ({cw}/{total})"
                    )
                else:
                    iters_left = max(0, cfg.min_iters_per_size - iters_on_size)
                    if iters_left > 0:
                        suffix = f"  (need {iters_left} more iter(s) on {board_size}×{board_size})"
                    elif next_idx >= len(cfg.board_sizes):
                        suffix = "  (at largest board size)"
                    else:
                        suffix = "  (loss still converging)"
                    self.signals.status_message.emit(
                        f"Iteration {self._iteration} — new champion! "
                        f"({cw}/{total}, {win_rate:.0%}){suffix}"
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
        # compile=False: torch.compile is not thread-safe during JIT compilation;
        # multiple background threads (demo, self-play, arena) running simultaneously
        # would race on dynamo's global bytecode patcher. TF32 still applies.
        self._net    = build_net(cfg, self._device, compile=False)

        # Auto-load best checkpoint before building UI so the board and
        # curriculum ladder reflect the state training was left in.
        self._startup_msg = self._autoload_best()

        self._shutting_down = False

        self._trainer:      Trainer | None        = None
        self._worker:       TrainingWorker | None = None
        self._train_thread: QThread | None        = None
        self._demo:         DemoWorker | None     = None

        self._build_ui()
        self._connect_signals()
        self.setWindowTitle("HexZero — AlphaZero for Hex")
        self.resize(1200, 740)

        # Restore curriculum ladder from persisted training state
        ts = ckpt_io.load_training_state(self.cfg.checkpoint_dir)
        if ts:
            self._stats.restore_state(
                ts.get("board_size", self.cfg.initial_board_size),
                ts.get("iters_on_size", 0),
            )
        elif self.cfg.initial_board_size != self.cfg.board_sizes[0]:
            self._stats.restore_state(self.cfg.initial_board_size, 0)

        if self._startup_msg:
            self._status_label.setText(self._startup_msg)

    # ------------------------------------------------------------------
    # Auto-load
    # ------------------------------------------------------------------

    def _autoload_best(self) -> str:
        """Load best.pt into self._net and restore board size. Returns status string."""
        best = ckpt_io.best_checkpoint_path(self.cfg.checkpoint_dir)
        if best is None:
            return ""
        try:
            data = ckpt_io.load(best, self._device)
            ckpt_io.load_weights(self._net, data["model_state"])
            board_size = data.get("metrics", {}).get("board_size")
            if board_size and board_size in self.cfg.board_sizes:
                self.cfg.initial_board_size = board_size
            iteration = data.get("iteration", 0)
            sz = self.cfg.initial_board_size
            return f"Loaded checkpoint: iter {iteration}, board {sz}×{sz}"
        except Exception as e:
            return f"Could not load checkpoint: {e}"

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
        self._stats = StatsWidget(
            sizes=self.cfg.board_sizes,
            min_iters=self.cfg.min_iters_per_size,
            use_pie_rule=self.cfg.use_pie_rule,
        )

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
        self._sims_spin.setValue(self.cfg.mcts_simulations_per_size[0]
                                  if self.cfg.mcts_simulations_per_size
                                  else self.cfg.mcts_simulations)
        self._sims_spin.setSingleStep(50)
        self._sims_spin.setToolTip(
            "MCTS simulations per move for the smallest board size.\n"
            "Larger board sizes scale up proportionally (see mcts_simulations_per_size).\n"
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
        sig.arena_progress.connect(self._stats.on_arena_progress)
        sig.buffer_updated.connect(self._stats.on_buffer_updated)
        sig.metrics_updated.connect(self._stats.on_metrics)
        sig.board_size_advanced.connect(self._on_size_advanced)
        sig.board_size_advanced.connect(self._stats.on_board_size_advanced)
        sig.curriculum_progress.connect(self._stats.on_curriculum_progress)
        sig.swap_rate_updated.connect(self._stats.on_swap_rate_updated)

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
            ckpt_io.load_weights(self._net, data["model_state"])
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
        # Scale all per-size entries proportionally so relative ratios are preserved.
        per_size = self.cfg.mcts_simulations_per_size
        base = per_size[0] if per_size else value
        if base > 0 and per_size:
            self.cfg.mcts_simulations_per_size = [
                max(1, round(s * value / base)) for s in per_size
            ]

    # ------------------------------------------------------------------
    # Close
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        if self._shutting_down:
            # Second call: threads are done (or force-terminated), allow close.
            event.accept()
            return

        event.ignore()
        self._start_shutdown()

    def _start_shutdown(self) -> None:
        self._shutting_down = True
        self.setEnabled(False)
        self.setWindowTitle("HexZero — Stopping…")

        if self._worker:
            self._worker.stop()
        if self._demo:
            self._demo.stop()

        self._shutdown_ms = 0
        self._shutdown_timer = QTimer(self)
        self._shutdown_timer.timeout.connect(self._poll_shutdown)
        self._shutdown_timer.start(100)

    def _poll_shutdown(self) -> None:
        self._shutdown_ms += 100
        train_alive = self._train_thread is not None and self._train_thread.isRunning()
        demo_alive  = self._demo         is not None and self._demo.isRunning()

        if not train_alive and not demo_alive:
            self._shutdown_timer.stop()
            self.close()
            return

        # After 15 s give up waiting and force-terminate
        if self._shutdown_ms >= 15_000:
            self._shutdown_timer.stop()
            for t in (self._train_thread, self._demo):
                if t and t.isRunning():
                    t.terminate()
                    t.wait(1_000)
            self.close()
