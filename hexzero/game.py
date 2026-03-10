"""
Hex game engine.

Board layout: H rows × W cols (standard: square, H=W=size).
Player 1 (BLACK = 1) connects TOP to BOTTOM (row 0 to row H-1).
Player 2 (WHITE = -1) connects LEFT to RIGHT (col 0 to col W-1).

Cells are indexed (row, col). Neighbour connectivity is the standard
6-connected hex grid:
    (r-1,c), (r-1,c+1),
    (r,  c-1),           (r,  c+1),
    (r+1,c-1),(r+1,c)

Win detection uses union-find with two virtual nodes per player:
    black_top (id = H*W),   black_bottom (id = H*W+1)
    white_left(id = H*W+2), white_right (id = H*W+3)
"""

import numpy as np


BLACK = 1
WHITE = -1
EMPTY = 0

_NEIGHBOUR_DELTAS = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path halving
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1

    def connected(self, x: int, y: int) -> bool:
        return self.find(x) == self.find(y)


class HexState:
    """
    Immutable-ish game state. Clone before mutating for MCTS.
    """

    def __init__(self, size: int):
        self.size = size
        self.board = np.zeros((size, size), dtype=np.int8)
        self.current_player = BLACK
        self.move_count = 0
        self.last_move: tuple[int, int] | None = None
        self._winner: int | None = None

        # 4 virtual nodes: black_top, black_bottom, white_left, white_right
        n_cells = size * size
        self._uf = UnionFind(n_cells + 4)
        self._black_top    = n_cells
        self._black_bottom = n_cells + 1
        self._white_left   = n_cells + 2
        self._white_right  = n_cells + 3

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def clone(self) -> "HexState":
        s = HexState.__new__(HexState)
        s.size = self.size
        s.board = self.board.copy()
        s.current_player = self.current_player
        s.move_count = self.move_count
        s.last_move = self.last_move
        s._winner = self._winner
        s._black_top    = self._black_top
        s._black_bottom = self._black_bottom
        s._white_left   = self._white_left
        s._white_right  = self._white_right
        uf = UnionFind(0)
        uf.parent = self._uf.parent[:]
        uf.rank   = self._uf.rank[:]
        s._uf = uf
        return s

    def legal_moves(self) -> list[tuple[int, int]]:
        if self._winner is not None:
            return []
        rows, cols = np.where(self.board == EMPTY)
        return list(zip(rows.tolist(), cols.tolist(), strict=True))

    def apply_move(self, move: tuple[int, int]) -> None:
        r, c = move
        assert self.board[r, c] == EMPTY, f"Cell ({r},{c}) already occupied"
        player = self.current_player
        self.board[r, c] = player
        self.last_move = move
        self.move_count += 1

        cell_id = self._cell(r, c)
        self._connect_neighbours(r, c, player, cell_id)

        if player == BLACK:
            if r == 0:
                self._uf.union(cell_id, self._black_top)
            if r == self.size - 1:
                self._uf.union(cell_id, self._black_bottom)
            if self._uf.connected(self._black_top, self._black_bottom):
                self._winner = BLACK
        else:
            if c == 0:
                self._uf.union(cell_id, self._white_left)
            if c == self.size - 1:
                self._uf.union(cell_id, self._white_right)
            if self._uf.connected(self._white_left, self._white_right):
                self._winner = WHITE

        self.current_player = -player

    def is_terminal(self) -> bool:
        return self._winner is not None

    def winner(self) -> int | None:
        return self._winner

    def result_for(self, player: int) -> float:
        """1.0 if player won, -1.0 if lost, 0.0 if not terminal."""
        if self._winner is None:
            return 0.0
        return 1.0 if self._winner == player else -1.0

    # ------------------------------------------------------------------
    # Symmetry: Hex has 180° rotational symmetry (swap players + rotate)
    # ------------------------------------------------------------------

    def apply_symmetry(self) -> "HexState":
        """
        Return a new state equivalent under 180° rotation + player swap.
        The resulting state has the same strategic content from the
        opposite player's perspective.
        """
        s = HexState(self.size)
        s.board = -np.rot90(self.board, 2).copy()
        s.current_player = -self.current_player
        s.move_count = self.move_count
        s._winner = -self._winner if self._winner is not None else None
        # Rebuild union-find from the rotated board
        for r in range(self.size):
            for c in range(self.size):
                if s.board[r, c] != EMPTY:
                    s._reconnect_cell(r, c)
        return s

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _cell(self, r: int, c: int) -> int:
        return r * self.size + c

    def _connect_neighbours(self, r: int, c: int, player: int, cell_id: int) -> None:
        for dr, dc in _NEIGHBOUR_DELTAS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.size and 0 <= nc < self.size:
                if self.board[nr, nc] == player:
                    self._uf.union(cell_id, self._cell(nr, nc))

    def _reconnect_cell(self, r: int, c: int) -> None:
        """Re-apply union-find connections for a cell (used after board copy)."""
        player = self.board[r, c]
        cell_id = self._cell(r, c)
        self._connect_neighbours(r, c, player, cell_id)
        if player == BLACK:
            if r == 0:
                self._uf.union(cell_id, self._black_top)
            if r == self.size - 1:
                self._uf.union(cell_id, self._black_bottom)
        else:
            if c == 0:
                self._uf.union(cell_id, self._white_left)
            if c == self.size - 1:
                self._uf.union(cell_id, self._white_right)

    def __repr__(self) -> str:
        symbols = {BLACK: "B", WHITE: "W", EMPTY: "."}
        lines = []
        for r in range(self.size):
            indent = " " * r
            row = " ".join(symbols[self.board[r, c]] for c in range(self.size))
            lines.append(indent + row)
        return "\n".join(lines)
