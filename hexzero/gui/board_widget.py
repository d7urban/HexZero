"""
BoardWidget: renders the Hex board using QPainter with hexagonal cells.

Hex grid orientation:
  - Rows run left-to-right, slightly offset downward (standard pointy-top layout
    rotated 30°, giving the traditional parallelogram board shape).
  - BLACK connects top to bottom (marked with coloured borders at top/bottom).
  - WHITE connects left to right (marked with coloured borders at left/right).

Supports:
  - Displaying any HexState
  - Policy heatmap overlay (semi-transparent)
  - Last-move highlight
  - Mouse click to emit a move (for human play mode)
"""

import math

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import QSizePolicy, QWidget

from hexzero.game import BLACK, EMPTY, WHITE, HexState

# Colours
_COL_BLACK      = QColor(30,  90,  200)   # blue player
_COL_WHITE      = QColor(200, 40,  40)    # red player
_COL_EMPTY      = QColor(200, 190, 160)
_COL_EMPTY_HVR  = QColor(220, 215, 185)
_COL_BLACK_EDGE = QColor(15,  55,  150)   # top/bottom border (blue)
_COL_WHITE_EDGE = QColor(155, 20,  20)    # left/right border (red)
_COL_LAST_MOVE  = QColor(80,  200, 80,  200)
_COL_SWAP_RING  = QColor(255, 210, 0,   220)  # golden ring: "click to swap"
_COL_POLICY_HOT = QColor(255, 80,  0,   160)
_COL_POLICY_COLD= QColor(0,   80,  255, 40)
_COL_WIN_DOT    = QColor(60,  220, 80)    # green dot on winning-path stones


def _hex_corners(cx: float, cy: float, r: float) -> list:
    """Return 6 QPointF corners for a pointy-top hexagon centred at (cx,cy).

    Pointy-top: flat edges on left/right, pointed ends at top/bottom.
    This is the correct orientation for a Hex game board where rows go
    horizontally and adjacent rows share slanted edges.
    Corners start at 30° so the top vertex is at 90°.
    """
    return [
        QPointF(cx + r * math.cos(math.radians(30 + 60 * i)),
                cy + r * math.sin(math.radians(30 + 60 * i)))
        for i in range(6)
    ]


class BoardWidget(QWidget):
    move_clicked = pyqtSignal(int, int)   # row, col

    def __init__(self, parent=None):
        super().__init__(parent)
        self._state: HexState | None = None
        self._policy: np.ndarray | None = None   # (H*W,) float32
        self._hover: tuple[int, int] | None = None
        self._win_path: set[tuple[int, int]] = set()
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(300, 300)

    # ------------------------------------------------------------------
    # Public slots
    # ------------------------------------------------------------------

    def set_state(self, state: HexState, policy: np.ndarray | None = None) -> None:
        self._state = state
        self._policy = policy
        self._win_path = state.winning_path() if state.is_terminal() else set()
        self.update()

    def set_policy(self, policy: np.ndarray) -> None:
        self._policy = policy
        self.update()

    def clear(self) -> None:
        self._state = None
        self._policy = None
        self.update()

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _cell_radius(self) -> float:
        if self._state is None:
            return 20.0
        size = self._state.size
        w, h = self.width(), self.height()
        # Pointy-top hex tiling (size×size board, parallelogram shape):
        #   board_w = sqrt(3)*r * (1.5*(size-1) + 1)   [cols + row offsets]
        #   board_h = r * (1.5*(size-1) + 2)            [rows + top/bottom caps]
        # Leave ~15% margin each side for edge strips and labels.
        avail_w = w * 0.82
        avail_h = h * 0.82
        r_from_w = avail_w / (math.sqrt(3) * (1.5 * (size - 1) + 1))
        r_from_h = avail_h / (1.5 * (size - 1) + 2)
        return max(8.0, min(r_from_w, r_from_h))

    def _board_origin(self, radius: float) -> QPointF:
        """Pixel position of the centre of cell (0, 0), centred in the widget."""
        if self._state is None:
            return QPointF(radius, radius)
        size    = self._state.size
        dx      = radius * math.sqrt(3)
        board_w = dx * (1.5 * (size - 1) + 1)
        board_h = radius * (1.5 * (size - 1) + 2)
        ox = (self.width()  - board_w) / 2 + dx / 2
        oy = (self.height() - board_h) / 2 + radius
        return QPointF(ox, oy)

    def _cell_center(self, row: int, col: int, radius: float) -> QPointF:
        """Return pixel centre for cell (row, col).

        Pointy-top layout:
          dx = sqrt(3)*r  — center-to-center same row (cells share flat edge)
          dy = 1.5*r      — center-to-center adjacent rows (cells share slanted edge)
          row offset      — each row shifts right by dx/2
        """
        dx = radius * math.sqrt(3)
        dy = radius * 1.5
        origin = self._board_origin(radius)
        px = origin.x() + col * dx + row * dx * 0.5
        py = origin.y() + row * dy
        return QPointF(px, py)

    def _cell_at_pos(self, px: float, py: float) -> tuple[int, int] | None:
        if self._state is None:
            return None
        radius = self._cell_radius()
        size   = self._state.size
        dx     = radius * math.sqrt(3)
        dy     = radius * 1.5
        origin = self._board_origin(radius)   # computed once, not per cell
        ox, oy = origin.x(), origin.y()
        best   = None
        best_d = radius * 1.2
        for row in range(size):
            for col in range(size):
                cpx = ox + col * dx + row * dx * 0.5
                cpy = oy + row * dy
                d = math.hypot(px - cpx, py - cpy)
                if d < best_d:
                    best_d = d
                    best = (row, col)
        return best

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self._state is None:
            painter.fillRect(self.rect(), QColor(50, 50, 50))
            painter.setPen(QColor(200, 200, 200))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No game loaded")
            return

        state  = self._state
        size   = state.size
        radius = self._cell_radius()

        # Background
        painter.fillRect(self.rect(), QColor(60, 55, 50))

        # Draw edge zone markers (coloured strips)
        self._draw_edge_markers(painter, size, radius)

        # Draw cells
        policy_max = float(self._policy.max()) if self._policy is not None and self._policy.max() > 0 else 1.0

        for row in range(size):
            for col in range(size):
                self._draw_cell(painter, state, row, col, radius, policy_max)

        # Coordinate labels (optional, only when cells are large enough)
        if radius >= 18:
            self._draw_coords(painter, size, radius)

        painter.end()

    def _draw_edge_markers(self, painter: QPainter, size: int, radius: float) -> None:
        """Coloured strips along the four board edges.

        Pointy-top hex geometry:
          dx = sqrt(3)*r  (cell width, flat-to-flat)
          Top/bottom point of each cell is at ±radius from its centre.
          Left/right flat edge of each cell is at ±dx/2 from its centre,
          spanning ±radius/2 vertically.

        BLACK connects top→bottom: blue strips above row 0 and below row S-1.
        WHITE connects left→right: red strips left of col 0 and right of col S-1.
        Strips are wide enough to merge into a continuous band.
        """
        strip = max(5, int(radius * 0.35))
        dx    = radius * math.sqrt(3)
        half  = int(dx / 2) + 1   # half-cell-width in pixels (ensures no gap)

        # BLACK — top strip: above each cell in row 0
        for col in range(size):
            c = self._cell_center(0, col, radius)
            painter.fillRect(
                int(c.x() - half), int(c.y() - radius - strip),
                half * 2, strip,
                _COL_BLACK_EDGE,
            )
        # BLACK — bottom strip: below each cell in row size-1
        for col in range(size):
            c = self._cell_center(size - 1, col, radius)
            painter.fillRect(
                int(c.x() - half), int(c.y() + radius),
                half * 2, strip,
                _COL_BLACK_EDGE,
            )

        # WHITE — left strip: left of each cell in col 0
        # Flat edge of pointy-top hex is at cx ± dx/2, spanning cy ± r/2
        half_v = int(radius / 2) + 1
        for row in range(size):
            c = self._cell_center(row, 0, radius)
            painter.fillRect(
                int(c.x() - dx / 2 - strip), int(c.y() - half_v),
                strip, half_v * 2,
                _COL_WHITE_EDGE,
            )
        # WHITE — right strip: right of each cell in col size-1
        for row in range(size):
            c = self._cell_center(row, size - 1, radius)
            painter.fillRect(
                int(c.x() + dx / 2), int(c.y() - half_v),
                strip, half_v * 2,
                _COL_WHITE_EDGE,
            )

    def _draw_cell(
        self,
        painter: QPainter,
        state: HexState,
        row: int,
        col: int,
        radius: float,
        policy_max: float,
    ) -> None:
        cell_val = state.board[row, col]
        center   = self._cell_center(row, col, radius)
        corners  = _hex_corners(center.x(), center.y(), radius * 0.95)
        polygon  = QPolygonF(corners)

        # Determine fill colour
        if cell_val == BLACK:
            fill = _COL_BLACK
        elif cell_val == WHITE:
            fill = _COL_WHITE
        # Empty cell: maybe tint by policy heatmap
        elif self._policy is not None:
            idx = row * state.size + col
            intensity = float(self._policy[idx]) / (policy_max + 1e-8)
            r = int(_COL_POLICY_COLD.red()   + intensity * (_COL_POLICY_HOT.red()   - _COL_POLICY_COLD.red()))
            g = int(_COL_POLICY_COLD.green() + intensity * (_COL_POLICY_HOT.green() - _COL_POLICY_COLD.green()))
            b = int(_COL_POLICY_COLD.blue()  + intensity * (_COL_POLICY_HOT.blue()  - _COL_POLICY_COLD.blue()))
            a = int(_COL_POLICY_COLD.alpha() + intensity * (_COL_POLICY_HOT.alpha() - _COL_POLICY_COLD.alpha()))
            policy_colour = QColor(r, g, b, a)
            # Blend with base empty colour
            base = _COL_EMPTY_HVR if self._hover == (row, col) else _COL_EMPTY
            fill = _blend_colours(base, policy_colour, intensity * 0.8)
        else:
            fill = _COL_EMPTY_HVR if self._hover == (row, col) else _COL_EMPTY

        painter.setBrush(QBrush(fill))

        # Highlight last move
        if state.last_move == (row, col):
            painter.setPen(QPen(_COL_LAST_MOVE, max(2, radius * 0.12)))
        else:
            painter.setPen(QPen(QColor(80, 75, 70), max(1, radius * 0.06)))

        painter.drawPolygon(polygon)

        # Stone dot for occupied cells
        if cell_val != EMPTY:
            dot_r = radius * 0.30
            if (row, col) in self._win_path:
                dot_c = _COL_WIN_DOT
            else:
                dot_c = QColor(140, 185, 255) if cell_val == BLACK else QColor(255, 140, 140)
            painter.setBrush(QBrush(dot_c))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(center, dot_r, dot_r)

        # Swap hint: golden ring around the last stone when pie-rule swap is available.
        # Signals "click this stone to swap sides".
        if (state.pie_rule and state.move_count == 1
                and state.last_move == (row, col)):
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(_COL_SWAP_RING, max(2.0, radius * 0.14)))
            painter.drawEllipse(center, radius * 0.68, radius * 0.68)

    def _draw_coords(self, painter: QPainter, size: int, radius: float) -> None:
        dx = radius * math.sqrt(3)
        font = QFont("Monospace", max(7, int(radius * 0.38)))
        painter.setFont(font)
        painter.setPen(QColor(160, 160, 140))
        lw = radius * 1.4   # label box width
        lh = radius * 0.9   # label box height
        for i in range(size):
            # Row labels (A, B, C, ...) — to the left of each col-0 cell's flat edge
            c = self._cell_center(i, 0, radius)
            painter.drawText(
                QRectF(c.x() - dx / 2 - lw - 2, c.y() - lh / 2, lw, lh),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                chr(ord("A") + i),
            )
            # Col labels (1, 2, 3, ...) — above each row-0 cell's top point
            c2 = self._cell_center(0, i, radius)
            painter.drawText(
                QRectF(c2.x() - lw / 2, c2.y() - radius - lh - 2, lw, lh),
                Qt.AlignmentFlag.AlignCenter,
                str(i + 1),
            )

    # ------------------------------------------------------------------
    # Mouse events
    # ------------------------------------------------------------------

    def mouseMoveEvent(self, event):
        cell = self._cell_at_pos(event.position().x(), event.position().y())
        if cell != self._hover:
            self._hover = cell
            self.update()
        # Pointer cursor when hovering over a stone that can be swapped
        if (cell is not None and self._state is not None
                and self._state.pie_rule and self._state.move_count == 1
                and cell == self._state.last_move):
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        else:
            self.unsetCursor()

    def leaveEvent(self, event):
        self._hover = None
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            cell = self._cell_at_pos(event.position().x(), event.position().y())
            if cell is not None and self._state is not None and not self._state.is_terminal():
                self.move_clicked.emit(*cell)


def _blend_colours(base: QColor, overlay: QColor, alpha: float) -> QColor:
    a = max(0.0, min(1.0, alpha))
    r = int(base.red()   * (1 - a) + overlay.red()   * a)
    g = int(base.green() * (1 - a) + overlay.green() * a)
    b = int(base.blue()  * (1 - a) + overlay.blue()  * a)
    return QColor(r, g, b)
