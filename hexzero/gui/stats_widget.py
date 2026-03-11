"""
StatsWidget: compact horizontal strip showing live training progress.

┌──────────────────────────────────────────────────────────────────────────┐
│ Iter 3  │  Self-play ████████░░ 80/100  │  Buffer 8,400  │  Arena 24/40  │  ✓7×7 → ●9×9 → ○11×11  │
└──────────────────────────────────────────────────────────────────────────┘
"""

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QLabel, QProgressBar, QFrame
)
from PyQt6.QtCore import pyqtSlot


def _sep() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.Shape.VLine)
    f.setStyleSheet("color: #555;")
    return f


def _label(text: str, bold: bool = False) -> QLabel:
    lbl = QLabel(text)
    if bold:
        font = lbl.font()
        font.setBold(True)
        lbl.setFont(font)
    lbl.setStyleSheet("color: #c0c0c0;")
    return lbl


# Styles for each curriculum state
_STYLE_DONE    = "color: #6acc6a; font-size: 10px; font-weight: bold;"
_STYLE_CURRENT = "color: #ffffff; font-size: 10px; font-weight: bold;"
_STYLE_LOCKED  = "color: #505050; font-size: 10px;"
_STYLE_ARROW   = "color: #505050; font-size: 10px;"
_STYLE_ARROW_DONE = "color: #6acc6a; font-size: 10px;"


class CurriculumWidget(QWidget):
    """
    Displays the board-size curriculum as a compact progression ladder:

        ✓ 7×7  →  ● 9×9  48% / 60%  →  ○ 11×11

    ✓ = completed (green), ● = current with live progress, ○ = locked (grey).
    The current-size label shows "current% / goal%" and shifts colour as the
    model approaches the advancement threshold:
        grey  → below 50 % of goal
        amber → 50–89 % of goal
        lime  → 90–99 % of goal
        green → at or above goal (advancement is imminent / just happened)
    At the last (largest) board size there is no goal, so only the win rate
    is shown without a target.
    """

    def __init__(self, sizes: list[int], goal_pct: int, parent=None):
        super().__init__(parent)
        self._sizes       = sizes
        self._goal_pct    = goal_pct   # e.g. 60 for 60%
        self._current_idx = 0
        self._win_pct     = 0          # last known arena win % for current size

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._size_labels:  list[QLabel] = []
        self._arrow_labels: list[QLabel] = []

        for i, _ in enumerate(sizes):
            if i > 0:
                arrow = QLabel("→")
                layout.addWidget(arrow)
                self._arrow_labels.append(arrow)

            lbl = QLabel()
            layout.addWidget(lbl)
            self._size_labels.append(lbl)

        self._refresh()

    def advance(self, new_size: int) -> None:
        if new_size in self._sizes:
            self._current_idx = self._sizes.index(new_size)
            self._win_pct = 0   # reset progress for the new size
            self._refresh()

    def update_progress(self, win_rate: float) -> None:
        """Call after each arena result with the candidate's win rate (0–1)."""
        self._win_pct = int(round(win_rate * 100))
        self._refresh()

    def _refresh(self) -> None:
        is_last = (self._current_idx == len(self._sizes) - 1)

        for i, lbl in enumerate(self._size_labels):
            size = self._sizes[i]
            if i < self._current_idx:
                lbl.setStyleSheet(_STYLE_DONE)
                lbl.setText(f"✓ {size}×{size}")
            elif i == self._current_idx:
                if is_last:
                    # No advancement threshold at the final size
                    suffix = f"  {self._win_pct}%" if self._win_pct > 0 else ""
                    lbl.setStyleSheet(_STYLE_CURRENT)
                    lbl.setText(f"● {size}×{size}{suffix}")
                else:
                    lbl.setStyleSheet(self._progress_style())
                    lbl.setText(self._progress_text(size))
            else:
                lbl.setStyleSheet(_STYLE_LOCKED)
                lbl.setText(f"○ {size}×{size}")

        for i, arrow in enumerate(self._arrow_labels):
            arrow.setStyleSheet(
                _STYLE_ARROW_DONE if i < self._current_idx else _STYLE_ARROW
            )

    def _progress_text(self, size: int) -> str:
        if self._win_pct > 0:
            return f"● {size}×{size}  {self._win_pct}% / {self._goal_pct}%"
        return f"● {size}×{size}  — / {self._goal_pct}%"

    def _progress_style(self) -> str:
        if self._goal_pct == 0:
            return _STYLE_CURRENT
        ratio = self._win_pct / self._goal_pct
        if ratio >= 1.0:
            return "color: #6acc6a; font-size: 10px; font-weight: bold;"
        if ratio >= 0.9:
            return "color: #aacc44; font-size: 10px; font-weight: bold;"
        if ratio >= 0.5:
            return "color: #ccaa44; font-size: 10px; font-weight: bold;"
        return _STYLE_CURRENT


class StatsWidget(QWidget):
    def __init__(self, sizes: list[int] | None = None, goal_pct: int = 60, parent=None):
        super().__init__(parent)
        self.setFixedHeight(34)
        self.setStyleSheet("background: #252525;")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 0, 12, 0)
        layout.setSpacing(10)

        # Iteration counter
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
                color: #ffffff;
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

        # Curriculum ladder (right-aligned)
        layout.addWidget(_sep())
        self._curriculum = CurriculumWidget(sizes or [7, 9, 11], goal_pct)
        layout.addWidget(self._curriculum)

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

    @pyqtSlot(dict)
    def on_metrics(self, _metrics: dict) -> None:
        if self._phase_lbl.text() != "Training":
            # First metrics emission signals transition into training; reset bar.
            self._set_phase("Training", 0, 1)
        else:
            self._phase_lbl.setText("Training")

    @pyqtSlot(int, int, int)
    def on_arena_progress(self, done: int, cw: int, total: int) -> None:
        pct = int(100 * cw / done) if done > 0 else 0
        colour = "#6acc6a" if pct >= 55 else "#cc6a6a"
        self._arena_lbl.setText(f"Arena  {cw}/{done} ({pct}%)")
        self._arena_lbl.setStyleSheet(f"color: {colour};")
        self._set_phase("Arena", done, total)

    @pyqtSlot(int, int, int)
    def on_arena_result(self, cw: int, chw: int, draws: int) -> None:
        total    = cw + chw + draws
        win_rate = cw / total if total > 0 else 0.0
        pct      = int(100 * win_rate)
        self._arena_lbl.setText(f"Arena  {cw}/{total} ({pct}%)")
        colour = "#6acc6a" if pct >= 55 else "#cc6a6a"
        self._arena_lbl.setStyleSheet(f"color: {colour};")
        self._curriculum.update_progress(win_rate)
        self._set_phase("Idle", 0, 1)

    @pyqtSlot(int)
    def on_buffer_updated(self, n: int) -> None:
        if n >= 1_000:
            self._buffer_lbl.setText(f"Buffer  {n / 1000:.1f}k")
        else:
            self._buffer_lbl.setText(f"Buffer  {n}")

    @pyqtSlot(int)
    def on_board_size_advanced(self, size: int) -> None:
        self._curriculum.advance(size)
        # Also flip the progress bar to a fresh state for the new size
        self._set_phase("Self-play", 0, 1)

    # ------------------------------------------------------------------

    def restore_state(self, board_size: int, win_rate: float) -> None:
        """Restore curriculum position and last known win rate (called on startup)."""
        self._curriculum.advance(board_size)
        if win_rate > 0:
            self._curriculum.update_progress(win_rate)

    def _set_phase(self, phase: str, done: int, total: int) -> None:
        self._phase_lbl.setText(phase)
        self._progress.setRange(0, max(1, total))
        self._progress.setValue(done)
