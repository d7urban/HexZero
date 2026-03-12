"""
Monte Carlo Tree Search — PUCT variant (AlphaZero style).

Deliberately decoupled from HexNet: takes an `infer_fn` callable so it
can be used with any backend (trained net, random prior, batched GPU server).

infer_fn signature:
    infer_fn(state: HexState) -> (policy: np.ndarray shape (H*W+1,), value: float)
    policy is a probability distribution over all cells in row-major order,
    with the final element being the probability of the pie-rule swap move.
    value is from the current player's perspective in [-1, 1].

Performance note: Node objects do NOT store a copy of the game state.
_simulate clones the root state once per simulation and applies moves
in-place as it descends the tree.  This replaces the original design
that cloned the state for every legal move at expansion time (O(moves)
allocations per expansion → hours at 11×11/150 sims).
"""

import math
from collections.abc import Callable

import numpy as np

from hexzero.game import SWAP_MOVE, HexState


class Node:
    __slots__ = ("N", "W", "children", "is_expanded", "prior")

    def __init__(self, prior: float = 0.0):
        self.prior = prior        # P(s, a) from policy network
        self.N: int = 0           # visit count
        self.W: float = 0.0       # total value
        self.children: dict = {}
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
        # move_count of the game state at the current root — used for tree-reuse
        # validity check without storing a copy of the board.
        self._root_move_count: int | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._root = None
        self._root_move_count = None

    def search(
        self,
        state: HexState,
        add_noise: bool = True,
    ) -> tuple[np.ndarray, float, dict]:
        """
        Run MCTS simulations from `state`.

        Returns:
            pi      : policy vector (H*W+1,) — visit count distribution
            value   : estimated value for current player
            info    : dict with debug info (top moves, root N, etc.)
        """
        size = state.size
        n_cells = size * size

        # Tree reuse: keep the existing root if move_count matches.
        if (self._root is None
                or self._root_move_count != state.move_count):
            self._root = Node()
            self._root_move_count = state.move_count

        root = self._root
        if not root.is_expanded:
            self._expand(root, state)

        if add_noise:
            self._add_dirichlet_noise(root)

        for _ in range(self.simulations):
            self._simulate(root, state)

        # Build policy from visit counts (always n_cells+1 to include swap slot)
        pi = np.zeros(n_cells + 1, dtype=np.float32)
        for move, child in root.children.items():
            if move == SWAP_MOVE:
                pi[n_cells] = child.N
            else:
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
        if idx == size * size:
            return SWAP_MOVE
        return (idx // size, idx % size)

    def update_root(self, move) -> None:
        """Reuse subtree after a move is played (tree reuse)."""
        if self._root is not None and move in self._root.children:
            self._root = self._root.children[move]
            if self._root_move_count is not None:
                self._root_move_count += 1
        else:
            self._root = None
            self._root_move_count = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _simulate(self, root: Node, root_state: HexState) -> None:
        """
        One MCTS simulation.

        Clones root_state once and applies moves in-place while descending
        the tree — no per-child state copies needed.
        """
        path: list[tuple[Node, object]] = []
        node = root
        state = root_state.clone()   # single clone for the entire simulation

        # Selection: descend until unexpanded or terminal
        while node.is_expanded and not state.is_terminal():
            move, child = self._select_child(node)
            state.apply_move(move)
            path.append((child, move))   # child, not node — backup walks visited nodes
            node = child

        # Expansion + evaluation
        if state.is_terminal():
            value = state.result_for(state.current_player)
        else:
            value = self._expand(node, state)

        # Backup
        self._backup(root, path, value)

    def _select_child(self, node: Node) -> tuple[object, Node]:
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

    def _expand(self, node: Node, state: HexState) -> float:
        """
        Expand node using the given state.  Child nodes store only the prior
        probability — no state clone per child.  Returns the network value.
        """
        legal = state.legal_moves()
        if not legal:
            node.is_expanded = True
            return 0.0

        policy, value = self.infer_fn(state)
        size = state.size

        # Extract raw priors for legal moves only, then renormalise.
        raw: dict = {}
        for move in legal:
            if move == SWAP_MOVE:
                raw[move] = float(policy[size * size])
            else:
                r, c = move
                raw[move] = float(policy[r * size + c])

        prior_sum = sum(raw.values())
        if prior_sum > 0:
            scale = 1.0 / prior_sum
            raw = {m: p * scale for m, p in raw.items()}
        else:
            uniform = 1.0 / len(legal)
            raw = dict.fromkeys(legal, uniform)

        for move, prior in raw.items():
            node.children[move] = Node(prior=prior)   # no state clone

        node.is_expanded = True
        return value

    def _backup(self, root: Node, path: list, leaf_value: float) -> None:
        # leaf_value is from the perspective of the player at the leaf node.
        # Negate once per ply as we walk back toward root; after the loop v has
        # been negated len(path) times, which puts it in root's player's frame.
        root.N += 1

        v = leaf_value
        for child, _ in reversed(path):
            v = -v
            child.N += 1
            child.W += v

        root.W += v  # v is now from root.current_player's perspective

    def _add_dirichlet_noise(self, node: Node) -> None:
        if not node.children:
            return
        moves = list(node.children.keys())
        alphas = [
            self.dirichlet_alpha * len(moves) if move == SWAP_MOVE
            else self.dirichlet_alpha
            for move in moves
        ]
        noise = np.random.dirichlet(alphas)
        eps = self.dirichlet_epsilon
        for move, n in zip(moves, noise, strict=True):
            child = node.children[move]
            child.prior = (1 - eps) * child.prior + eps * float(n)

    def _top_moves(self, node: Node, size: int, k: int = 5) -> list[dict]:
        children = sorted(node.children.items(), key=lambda kv: -kv[1].N)[:k]
        result = []
        for move, child in children:
            result.append({
                "move":     move,
                "move_str": ("swap" if move == SWAP_MOVE
                             else f"{chr(ord('A') + move[0])}{move[1] + 1}"),
                "N":        child.N,
                "Q":        round(child.Q, 4),
                "P":        round(child.prior, 4),
            })
        return result
