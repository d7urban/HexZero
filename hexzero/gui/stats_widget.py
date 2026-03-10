"""
StatsWidget: compact horizontal strip showing live training progress.

┌──────────────────────────────────────────────────────────────────┐
│ Iter 3  │  Self-play ████████░░ 80/100  │  Buffer 8,400  │  Arena 24/40 (60%)  │
└──────────────────────────────────────────────────────────────────┘
"""

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QLabel, QProgressBar, QFrame
)
from PyQt6.QtCore import pyqtSlot


def _sep() -> QFrame:
    """Vertical separator line."""
    f = QFrame()
    f.setFrameShape(QFrame.Shape.VLine)
    f.setStyleSheet("color: #555;")
    return f


def _label(text: str, bold: bool = False) -> QLabel:
    lbl = QLabel(text)
    if bold:
        f = lbl.font()
        f.setBold(True)
        lbl.setFont(f)
    lbl.setStyleSheet("color: #c0c0c0;")
    return lbl


class StatsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(34)
        self.setStyleSheet("background: #252525;")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 0, 12, 0)
        layout.setSpacing(10)

        # Iteration
        self._iter_lbl = _label("Iter —", bold=True)
        layout.addWidget(self._iter_lbl)
        layout.addWidget(_sep())

        # Phase + progress bar
        self._phase_lbl = _label("Idle")
        layout.addWidget(self._phase_lbl)

        self._progress = QProgressBar()
        self._progress.setFixedWidth(160)
        self._progress.setFixedHeight(14)
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        self._progress.setTextVisible(True)
        self._progress.setFormat("%v / %m")
        self._progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #444;
                border-radius: 3px;
                background: #333;
                color: #aaa;
                font-size: 10px;
            }
            QProgressBar::chunk {
                background: #4a7acc;
                border-radius: 2px;
            }
        """)
        layout.addWidget(self._progress)
        layout.addWidget(_sep())

        # Buffer fill
        self._buffer_lbl = _label("Buffer  0")
        layout.addWidget(self._buffer_lbl)
        layout.addWidget(_sep())

        # Arena result
        self._arena_lbl = _label("Arena  —")
        layout.addWidget(self._arena_lbl)

        layout.addStretch()

        # Board size indicator (right-aligned)
        self._size_lbl = _label("7 × 7")
        layout.addWidget(self._size_lbl)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    @pyqtSlot(int)
    def on_iteration_started(self, n: int) -> None:
        self._iter_lbl.setText(f"Iter {n}")
        self._set_phase("Self-play", 0, 1)

    @pyqtSlot(int, int)
    def on_self_play_progress(self, done: int, total: int) -> None:
        self._set_phase("Self-play", done, total)

    def on_training_started(self, total_steps: int) -> None:
        self._set_phase("Training", 0, total_steps)

    @pyqtSlot(dict)
    def on_metrics(self, _metrics: dict) -> None:
        # Charts handle loss display; progress bar is driven by self_play_progress.
        pass

    def on_arena_started(self, total: int) -> None:
        self._set_phase("Arena", 0, total)

    @pyqtSlot(int, int, int)
    def on_arena_result(self, cw: int, chw: int, draws: int) -> None:
        total = cw + chw + draws
        pct   = int(100 * cw / total) if total > 0 else 0
        self._arena_lbl.setText(f"Arena  {cw}/{total} ({pct}%)")
        colour = "#6acc6a" if pct >= 55 else "#cc6a6a"
        self._arena_lbl.setStyleSheet(f"color: {colour};")
        self._set_phase("Idle", 0, 1)

    @pyqtSlot(int)
    def on_buffer_updated(self, n: int) -> None:
        if n >= 1_000:
            self._buffer_lbl.setText(f"Buffer  {n/1000:.1f}k")
        else:
            self._buffer_lbl.setText(f"Buffer  {n}")

    def on_board_size_changed(self, size: int) -> None:
        self._size_lbl.setText(f"{size} × {size}")

    # ------------------------------------------------------------------

    def _set_phase(self, phase: str, done: int, total: int) -> None:
        self._phase_lbl.setText(phase)
        self._progress.setRange(0, max(1, total))
        self._progress.setValue(done)
