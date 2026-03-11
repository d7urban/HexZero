"""
ChartWidget: live training curves using pyqtgraph.

Displays two plots side by side:
  Left:  Policy loss + Value loss (two lines)
  Right: Policy top-1 accuracy
"""

from collections import deque

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import pyqtSlot
from PyQt6.QtWidgets import QHBoxLayout, QWidget

_WINDOW = 2000   # number of recent steps shown


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
        self._policy_loss_curve = self._loss_plot.plot(
            pen=pg.mkPen(color=(100, 180, 255), width=1.5), name="Policy Loss"
        )
        self._value_loss_curve = self._loss_plot.plot(
            pen=pg.mkPen(color=(255, 120, 60), width=1.5), name="Value Loss"
        )
        layout.addWidget(self._loss_plot)

        # -- Accuracy plot --
        self._acc_plot = pg.PlotWidget(title="Policy Accuracy")
        self._acc_plot.setLabel("left", "Accuracy")
        self._acc_plot.setLabel("bottom", "Step")
        self._acc_plot.setYRange(0, 1)
        self._acc_curve = self._acc_plot.plot(
            pen=pg.mkPen(color=(100, 220, 100), width=1.5)
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
        self._policy_loss_curve.setData(steps, np.array(self._policy_loss))
        self._value_loss_curve.setData(steps, np.array(self._value_loss))
        self._acc_curve.setData(steps, np.array(self._accuracy))

    def clear(self) -> None:
        self._steps.clear()
        self._policy_loss.clear()
        self._value_loss.clear()
        self._accuracy.clear()
        self._policy_loss_curve.setData([], [])
        self._value_loss_curve.setData([], [])
        self._acc_curve.setData([], [])
