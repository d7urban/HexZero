"""
Feature plane construction for the neural network.

Input tensor shape: (C, H, W) where C = num_input_planes (default 8).

Plane layout:
  0  black_stones        — 1 where BLACK stone, 0 elsewhere
  1  white_stones        — 1 where WHITE stone, 0 elsewhere
  2  black_2bridge       — 1 where an empty cell completes a 2-bridge for BLACK
  3  white_2bridge       — 1 where an empty cell completes a 2-bridge for WHITE
  4  black_edge_dist     — min BFS distance to either of BLACK's two target edges (normalised)
  5  white_edge_dist     — min BFS distance to either of WHITE's two target edges (normalised)
  6  black_components    — normalised connected-component label for BLACK stones
  7  white_components    — normalised connected-component label for WHITE stones

All planes are float32 in [0, 1].

Returns:
    features  : np.ndarray shape (8, H, W) float32
    size_norm : float  board size normalised to [0,1] for value head conditioning
                       using min=5 max=19
"""

import numpy as np
from collections import deque

from hexzero.game import HexState, BLACK, WHITE, EMPTY, _NEIGHBOUR_DELTAS


_SIZE_MIN = 5
_SIZE_MAX = 19

# Two-bridge patterns: for each cell (r,c) and player, a 2-bridge exists when
# there is a pair of that player's stones at specific offsets and the two
# intermediate cells (the "carrier") are empty.  We enumerate all 3 canonical
# 2-bridge patterns in a hex grid.
#
# A 2-bridge between cells A and B with carrier {c1, c2} exists when:
#   A and B are both owned by the same player
#   c1 and c2 are empty
# The 3 carrier pairs for offset (A→B) are:
#   (+2,-1) carriers: (+1,-1),(+1, 0)
#   (+2, 1) carriers: (+1, 0),(+1, 1)   [actually (+1,+1) for the second]
#   ( 0,+2) carriers: (-1,+1),(+1,+1) — wait, this depends on orientation
#
# Simpler approach: enumerate all pairs of cells that are exactly 2 hex-steps
# apart and share the same carrier pair.  For a hex grid the canonical list is:
_TWO_BRIDGE_PATTERNS = [
    # (anchor_delta, carrier1_delta, carrier2_delta)
    # Each pattern: if board[r+ar, c+ac] == player AND board[r+c1r,c+c1c] == EMPTY
    #               AND board[r+c2r,c+c2c] == EMPTY, mark (r,c) as 2-bridge cell.
    # But more useful: mark the EMPTY cells in the carrier as 2-bridge threats.
    # Pattern 1: bridge over (r,c)↔(r-2,c+1) via carriers (r-1,c) and (r-1,c+1)
    ((-2,  1), (-1,  0), (-1,  1)),
    # Pattern 2: bridge over (r,c)↔(r-2,c-1) via carriers (r-1,c-1) and (r-1,c)
    ((-2, -1), (-1, -1), (-1,  0)),
    # Pattern 3: bridge over (r,c)↔(r, c-2) via carriers (r-1,c-1) and (r+1,c-1) — wrong
    # Actually let's use the standard 6-direction pairs:
    # For each pair of hex neighbours-of-neighbours sharing exactly 2 common neighbours:
]

# Re-derive cleanly.  In a hex grid, two cells are "2-bridge distance" apart if
# they share exactly 2 common neighbours.  All such pairs and their shared
# neighbours:
_BRIDGE_PAIRS = [
    # (dr_b, dc_b, dr_c1, dc_c1, dr_c2, dc_c2)
    # "b" is the OTHER stone; c1,c2 are the two shared neighbours (the carrier)
    (-2,  1,  -1,  0,  -1,  1),
    (-2, -1,  -1, -1,  -1,  0),
    ( 0,  2,  -1,  1,   1,  1),
    ( 0, -2,  -1, -1,   1, -1),
    ( 2, -1,   1, -1,   1,  0),
    ( 2,  1,   1,  0,   1,  1),
]


def _two_bridge_plane(board: np.ndarray, player: int) -> np.ndarray:
    """
    Returns a plane where cell (r,c) == 1 if it is an empty carrier cell
    of a two-bridge between two of `player`'s stones.
    """
    size = board.shape[0]
    plane = np.zeros((size, size), dtype=np.float32)
    player_cells = np.argwhere(board == player)

    for (r, c) in player_cells:
        for dr_b, dc_b, dr_c1, dc_c1, dr_c2, dc_c2 in _BRIDGE_PAIRS:
            rb, cb   = r + dr_b, c + dc_b
            rc1, cc1 = r + dr_c1, c + dc_c1
            rc2, cc2 = r + dr_c2, c + dc_c2
            if not (0 <= rb < size and 0 <= cb < size):
                continue
            if not (0 <= rc1 < size and 0 <= cc1 < size):
                continue
            if not (0 <= rc2 < size and 0 <= cc2 < size):
                continue
            if board[rb, cb] == player and board[rc1, cc1] == EMPTY and board[rc2, cc2] == EMPTY:
                plane[rc1, cc1] = 1.0
                plane[rc2, cc2] = 1.0
    return plane


def _bfs_from_seeds(board: np.ndarray, seeds: list, player: int) -> np.ndarray:
    """BFS distance from a set of seed cells, ignoring opponent-occupied cells."""
    size = board.shape[0]
    INF  = size * size + 1
    dist = np.full((size, size), INF, dtype=np.float32)
    q    = deque()
    for r, c in seeds:
        dist[r, c] = 0.0
        q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in _NEIGHBOUR_DELTAS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < size and 0 <= nc < size:
                if board[nr, nc] != -player and dist[nr, nc] == INF:
                    dist[nr, nc] = dist[r, c] + 1.0
                    q.append((nr, nc))
    return dist


def _edge_distance_plane(board: np.ndarray, player: int) -> np.ndarray:
    """
    Minimum BFS distance from each cell to either of the player's two target
    edges, ignoring opponent-occupied cells.

    BLACK connects top (row 0) ↔ bottom (row size-1).
    WHITE connects left (col 0) ↔ right (col size-1).

    Using min(dist_edge1, dist_edge2) instead of a single edge gives a
    symmetric, non-monotonic plane: both edge rows/cols score 0, the centre
    scores highest.  This prevents random network weights from producing a
    systematic first-row / first-column bias.
    """
    size = board.shape[0]

    if player == BLACK:
        seeds1 = [(0,      c) for c in range(size) if board[0,      c] != -player]
        seeds2 = [(size-1, c) for c in range(size) if board[size-1, c] != -player]
    else:
        seeds1 = [(r, 0)      for r in range(size) if board[r, 0]      != -player]
        seeds2 = [(r, size-1) for r in range(size) if board[r, size-1] != -player]

    dist1 = _bfs_from_seeds(board, seeds1, player)
    dist2 = _bfs_from_seeds(board, seeds2, player)

    # Normalise; cells unreachable from an edge (blocked by opponent) stay at 1.0
    plane = np.minimum(np.minimum(dist1, dist2) / size, 1.0)
    return plane


def _component_plane(board: np.ndarray, player: int) -> np.ndarray:
    """
    Labels each player stone with its connected-component index,
    normalised by total number of components so values are in (0, 1].
    Empty and opponent cells are 0.
    """
    size = board.shape[0]
    plane = np.zeros((size, size), dtype=np.float32)
    visited = np.zeros((size, size), dtype=bool)
    component_id = 0

    for start_r in range(size):
        for start_c in range(size):
            if board[start_r, start_c] == player and not visited[start_r, start_c]:
                component_id += 1
                stack = [(start_r, start_c)]
                cells = []
                while stack:
                    r, c = stack.pop()
                    if visited[r, c]:
                        continue
                    visited[r, c] = True
                    cells.append((r, c))
                    for dr, dc in _NEIGHBOUR_DELTAS:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < size and 0 <= nc < size:
                            if board[nr, nc] == player and not visited[nr, nc]:
                                stack.append((nr, nc))
                for r, c in cells:
                    plane[r, c] = float(component_id)

    if component_id > 0:
        plane /= component_id
    return plane


def extract_features(state: HexState) -> tuple[np.ndarray, float]:
    """
    Build the (8, H, W) feature tensor and board-size scalar for `state`.

    Returns:
        features  : np.ndarray float32 shape (8, size, size)
        size_norm : float in [0, 1]
    """
    board = state.board
    size  = state.size

    black_stones = (board == BLACK).astype(np.float32)
    white_stones = (board == WHITE).astype(np.float32)

    black_2b = _two_bridge_plane(board, BLACK)
    white_2b = _two_bridge_plane(board, WHITE)

    black_ed = _edge_distance_plane(board, BLACK)
    white_ed = _edge_distance_plane(board, WHITE)

    black_cc = _component_plane(board, BLACK)
    white_cc = _component_plane(board, WHITE)

    features = np.stack([
        black_stones,
        white_stones,
        black_2b,
        white_2b,
        black_ed,
        white_ed,
        black_cc,
        white_cc,
    ], axis=0)  # (8, size, size)

    size_norm = (size - _SIZE_MIN) / (_SIZE_MAX - _SIZE_MIN)
    size_norm = float(np.clip(size_norm, 0.0, 1.0))

    return features, size_norm
