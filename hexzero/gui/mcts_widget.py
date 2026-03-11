"""
MCTSWidget: simplified MCTS tree viewer.

Shows the top-N moves by visit count at depth 1 (the root's children),
rendered as a bar-chart-style tree since a full tree is too large to
display usefully during live play.

For depth > 1 a QGraphicsScene approach would be needed; this focuses on
the information most useful for human viewers: what move MCTS prefers and
why (visits, Q-value, prior).
"""


from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QBrush, QColor
from PyQt6.QtWidgets import (
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

_COLS = ["Move", "Visits (N)", "Value (Q)", "Prior (P)"]


class MCTSWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self._title = QLabel("MCTS — Top Moves")
        self._title.setStyleSheet("color: #c8c8c8; font-weight: bold;")
        layout.addWidget(self._title)

        self._table = QTableWidget(0, len(_COLS))
        self._table.setHorizontalHeaderLabels(_COLS)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setStyleSheet("""
            QTableWidget { background: #2a2a2a; color: #d0d0d0; gridline-color: #444; }
            QHeaderView::section { background: #383838; color: #aaa; padding: 4px; }
        """)
        layout.addWidget(self._table)

        self._root_label = QLabel("Root: N=0, Q=0.00")
        self._root_label.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self._root_label)

    @pyqtSlot(dict)
    def update_info(self, info: dict) -> None:
        """
        Accepts the `info` dict returned by MCTSAgent.search():
            info["top_moves"]  : list of dicts with move, N, Q, P
            info["root_N"]     : total visit count at root
            info["root_Q"]     : root Q value
        """
        top_moves: list[dict] = info.get("top_moves", [])
        root_N = info.get("root_N", 0)
        root_Q = info.get("root_Q", 0.0)

        self._root_label.setText(f"Root: N={root_N},  Q={root_Q:+.3f}")

        self._table.setRowCount(len(top_moves))
        max_N = max((m["N"] for m in top_moves), default=1) or 1

        for row, move_info in enumerate(top_moves):
            m = move_info["move"]
            move_str = move_info.get("move_str") or (
                "swap" if m == (-1, -1) else f"{chr(ord('A') + m[0])}{m[1] + 1}"
            )
            n_val = move_info["N"]
            q_val = move_info["Q"]
            p_val = move_info["P"]

            items = [
                QTableWidgetItem(move_str),
                QTableWidgetItem(str(n_val)),
                QTableWidgetItem(f"{q_val:+.4f}"),
                QTableWidgetItem(f"{p_val:.4f}"),
            ]

            # Colour Q value: green if positive, red if negative
            q_color = QColor(100, 230, 100) if q_val >= 0 else QColor(255, 110, 110)

            # Shade rows by visit count fraction
            intensity = int(40 + 60 * n_val / max_N)
            bg = QColor(intensity, intensity, intensity)

            for col, item in enumerate(items):
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                item.setBackground(QBrush(bg))
                if col == 2:
                    item.setForeground(QBrush(q_color))
                self._table.setItem(row, col, item)

    def clear(self) -> None:
        self._table.setRowCount(0)
        self._root_label.setText("Root: N=0, Q=0.00")
