"""
ChartWidget: live training curves using pyqtgraph.

Displays two plots side by side:
  Left:  Policy loss + Value loss (raw + MA overlay)
  Right: Policy top-1 accuracy (raw + MA overlay)

Plot titles update with the current moving-average value.
"""

from collections import deque

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import pyqtSlot
from PyQt6.QtWidgets import QHBoxLayout, QWidget

_WINDOW    = 2000   # number of recent steps kept in ring buffer
_MA_WINDOW = 50     # moving-average window (steps)


def _moving_avg(data: np.ndarray, w: int) -> np.ndarray | None:
    """Convolution-based moving average.  Returns None when len(data) < w."""
    if len(data) < w:
        return None
    return np.convolve(data, np.ones(w) / w, mode="valid")


class ChartWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        pg.setConfigOptions(antialias=True, background=(40, 40, 40), foreground=(200, 200, 200))

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # -- Loss plot --
        self._loss_plot = pg.PlotWidget(title="Loss")
        self._loss_plot.addLegend()
        self._loss_plot.setLabel("left", "Loss")
        self._loss_plot.setLabel("bottom", "Step")

        # Raw curves — thin, semi-transparent
        self._policy_loss_curve = self._loss_plot.plot(
            pen=pg.mkPen(color=(100, 180, 255, 80), width=1), name="Policy Loss"
        )
        self._value_loss_curve = self._loss_plot.plot(
            pen=pg.mkPen(color=(255, 120, 60, 80), width=1), name="Value Loss"
        )
        # MA overlay curves — same colour, thicker and opaque
        self._policy_loss_ma = self._loss_plot.plot(
            pen=pg.mkPen(color=(100, 180, 255), width=2), name=f"Policy MA-{_MA_WINDOW}"
        )
        self._value_loss_ma = self._loss_plot.plot(
            pen=pg.mkPen(color=(255, 120, 60), width=2), name=f"Value MA-{_MA_WINDOW}"
        )
        layout.addWidget(self._loss_plot)

        # -- Accuracy plot --
        self._acc_plot = pg.PlotWidget(title="Policy Accuracy")
        self._acc_plot.setLabel("left", "Accuracy")
        self._acc_plot.setLabel("bottom", "Step")
        self._acc_plot.setYRange(0, 1)

        # Raw curve — thin, semi-transparent
        self._acc_curve = self._acc_plot.plot(
            pen=pg.mkPen(color=(100, 220, 100, 80), width=1)
        )
        # MA overlay curve
        self._acc_ma = self._acc_plot.plot(
            pen=pg.mkPen(color=(100, 220, 100), width=2)
        )
        layout.addWidget(self._acc_plot)

        # Ring buffers
        self._steps        = deque(maxlen=_WINDOW)
        self._policy_loss  = deque(maxlen=_WINDOW)
        self._value_loss   = deque(maxlen=_WINDOW)
        self._accuracy     = deque(maxlen=_WINDOW)

    @pyqtSlot(dict)
    def update_metrics(self, metrics: dict) -> None:
        self._steps.append(metrics["step"])
        self._policy_loss.append(metrics["policy_loss"])
        self._value_loss.append(metrics["value_loss"])
        self._accuracy.append(metrics["policy_acc"])

        steps = np.array(self._steps)
        pl    = np.array(self._policy_loss)
        vl    = np.array(self._value_loss)
        ac    = np.array(self._accuracy)

        # Raw curves
        self._policy_loss_curve.setData(steps, pl)
        self._value_loss_curve.setData(steps, vl)
        self._acc_curve.setData(steps, ac)

        # Moving averages
        pl_ma = _moving_avg(pl, _MA_WINDOW)
        vl_ma = _moving_avg(vl, _MA_WINDOW)
        ac_ma = _moving_avg(ac, _MA_WINDOW)

        ma_x = steps[_MA_WINDOW - 1:]   # x positions for mode='valid' output
        if pl_ma is not None:
            self._policy_loss_ma.setData(ma_x, pl_ma)
        if vl_ma is not None:
            self._value_loss_ma.setData(ma_x, vl_ma)
        if ac_ma is not None:
            self._acc_ma.setData(ma_x, ac_ma)

        # Tag titles with current MA value
        p_str = f"{pl_ma[-1]:.3f}" if pl_ma is not None else "—"
        v_str = f"{vl_ma[-1]:.3f}" if vl_ma is not None else "—"
        a_str = f"{ac_ma[-1]:.1%}" if ac_ma is not None else "—"
        self._loss_plot.setTitle(
            f"Loss  ·  Policy {p_str}  ·  Value {v_str}"
        )
        self._acc_plot.setTitle(f"Policy Accuracy  ·  {a_str}")

    def clear(self) -> None:
        self._steps.clear()
        self._policy_loss.clear()
        self._value_loss.clear()
        self._accuracy.clear()
        self._policy_loss_curve.setData([], [])
        self._value_loss_curve.setData([], [])
        self._acc_curve.setData([], [])
        self._policy_loss_ma.setData([], [])
        self._value_loss_ma.setData([], [])
        self._acc_ma.setData([], [])
        self._loss_plot.setTitle("Loss")
        self._acc_plot.setTitle("Policy Accuracy")
