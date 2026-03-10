"""
Monte Carlo Tree Search — PUCT variant (AlphaZero style).

Deliberately decoupled from HexNet: takes an `infer_fn` callable so it
can be used with any backend (trained net, random prior, batched GPU server).

infer_fn signature:
    infer_fn(state: HexState) -> (policy: np.ndarray shape (H*W,), value: float)
    policy is a probability distribution over all cells in row-major order.
    value is from the current player's perspective in [-1, 1].
"""

import math
import numpy as np
from collections.abc import Callable

from hexzero.game import HexState


class Node:
    __slots__ = ("state", "prior", "N", "W", "children", "is_expanded")

    def __init__(self, state: HexState, prior: float = 0.0):
        self.state = state
        self.prior = prior        # P(s, a) from policy network
        self.N: int = 0           # visit count
        self.W: float = 0.0       # total value
        self.children: dict[tuple[int, int], Node] = {}
        self.is_expanded: bool = False

    @property
    def Q(self) -> float:
        return self.W / self.N if self.N > 0 else 0.0

    def puct_score(self, parent_N: int, cpuct: float) -> float:
        """PUCT = Q + cpuct * P * sqrt(parent_N) / (1 + N)"""
        return self.Q + cpuct * self.prior * math.sqrt(parent_N) / (1 + self.N)


class MCTSAgent:
    def __init__(
        self,
        infer_fn: Callable,
        simulations: int = 200,
        cpuct: float = 1.25,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        temperature: float = 1.0,
        temperature_moves: int = 20,
    ):
        self.infer_fn = infer_fn
        self.simulations = simulations
        self.cpuct = cpuct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.temperature = temperature
        self.temperature_moves = temperature_moves
        self._root: Node | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._root = None

    def search(
        self,
        state: HexState,
        add_noise: bool = True,
    ) -> tuple[np.ndarray, float, dict]:
        """
        Run MCTS simulations from `state`.

        Returns:
            pi      : policy vector (H*W,) — visit count distribution
            value   : estimated value for current player
            info    : dict with debug info (top moves, root N, etc.)
        """
        size = state.size
        n_cells = size * size

        if self._root is None or self._root.state is not state:
            self._root = Node(state)

        root = self._root
        if not root.is_expanded:
            self._expand(root)

        if add_noise:
            self._add_dirichlet_noise(root)

        for _ in range(self.simulations):
            self._simulate(root)

        # Build policy from visit counts
        pi = np.zeros(n_cells, dtype=np.float32)
        for move, child in root.children.items():
            r, c = move
            pi[r * size + c] = child.N

        if pi.sum() > 0:
            if state.move_count < self.temperature_moves and self.temperature > 0:
                pi = pi ** (1.0 / self.temperature)
            else:
                # Greedy: put all mass on the best move
                best = np.argmax(pi)
                pi[:] = 0
                pi[best] = 1.0
            pi /= pi.sum()

        info = {
            "root_N": root.N,
            "root_Q": root.Q,
            "top_moves": self._top_moves(root, size, k=5),
        }
        return pi, root.Q, info

    def select_move(self, state: HexState, add_noise: bool = False) -> tuple[int, int]:
        pi, _, _ = self.search(state, add_noise=add_noise)
        size = state.size
        idx = np.random.choice(len(pi), p=pi)
        return (idx // size, idx % size)

    def update_root(self, move: tuple[int, int]) -> None:
        """Reuse subtree after a move is played (tree reuse)."""
        if self._root is not None and move in self._root.children:
            self._root = self._root.children[move]
        else:
            self._root = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _simulate(self, root: Node) -> None:
        path: list[tuple[Node, tuple[int, int] | None]] = []
        node = root

        # Selection: descend until unexpanded or terminal
        while node.is_expanded and not node.state.is_terminal():
            move, node = self._select_child(node)
            path.append((node, move))

        # Expansion + evaluation
        if node.state.is_terminal():
            # Winner is opposite of current player (last player to move won)
            value = node.state.result_for(node.state.current_player)
        else:
            # _expand calls infer_fn internally; reuse its value directly
            value = self._expand(node)

        # Backup
        self._backup(root, path, value)

    def _select_child(self, node: Node) -> tuple[tuple[int, int], Node]:
        parent_N = node.N
        best_score = -float("inf")
        best_move = None
        best_child = None
        for move, child in node.children.items():
            score = child.puct_score(parent_N, self.cpuct)
            if score > best_score:
                best_score = score
                best_move = move
                best_child = child
        return best_move, best_child

    def _expand(self, node: Node) -> float:
        """Expand node, returning the network value for backup."""
        state = node.state
        legal = state.legal_moves()
        if not legal:
            node.is_expanded = True
            return 0.0

        policy, value = self.infer_fn(state)
        size = state.size

        for move in legal:
            r, c = move
            prior = float(policy[r * size + c])
            child_state = state.clone()
            child_state.apply_move(move)
            node.children[move] = Node(child_state, prior=prior)

        node.is_expanded = True
        return value

    def _backup(self, root: Node, path: list, leaf_value: float) -> None:
        # leaf_value is from the perspective of the player at the leaf node.
        # As we go back up, alternate sign each ply.
        root.N += 1
        root.W += leaf_value  # approximate; root W not used for move selection

        v = leaf_value
        for child, _ in reversed(path):
            v = -v
            child.N += 1
            child.W += v

    def _add_dirichlet_noise(self, node: Node) -> None:
        if not node.children:
            return
        moves = list(node.children.keys())
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(moves))
        eps = self.dirichlet_epsilon
        for move, n in zip(moves, noise, strict=True):
            child = node.children[move]
            child.prior = (1 - eps) * child.prior + eps * float(n)

    def _top_moves(self, node: Node, size: int, k: int = 5) -> list[dict]:
        children = sorted(node.children.items(), key=lambda kv: -kv[1].N)[:k]
        return [
            {
                "move": move,
                "N": child.N,
                "Q": round(child.Q, 4),
                "P": round(child.prior, 4),
            }
            for move, child in children
        ]
