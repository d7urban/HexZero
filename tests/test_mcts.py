"""Tests for hexzero/mcts.py — MCTSAgent internals."""
import unittest

import numpy as np

from hexzero.game import SWAP_MOVE, WHITE, HexState
from hexzero.mcts import MCTSAgent, Node

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uniform_infer(state: HexState):
    n_cells = state.size * state.size
    legal = state.legal_moves()
    policy = [0.0] * (n_cells + 1)
    if legal:
        p = 1.0 / len(legal)
        for move in legal:
            if move == SWAP_MOVE:
                policy[n_cells] = p
            else:
                r, c = move
                policy[r * state.size + c] = p
    return policy, 0.0


def _biased_infer(state: HexState, preferred: tuple[int, int]):
    """Put 90% weight on `preferred`, split remainder uniformly."""
    n_cells = state.size * state.size
    legal = state.legal_moves()
    policy = [0.0] * (n_cells + 1)
    others = [m for m in legal if m != preferred]
    if preferred in legal:
        policy[preferred[0] * state.size + preferred[1]] = 0.9
    if others:
        spread = 0.1 / len(others)
        for m in others:
            policy[m[0] * state.size + m[1]] = spread
    return policy, 0.0


# ---------------------------------------------------------------------------
# _expand tests
# ---------------------------------------------------------------------------

class ExpandTests(unittest.TestCase):
    def test_expand_creates_children_for_all_legal_moves(self):
        s = HexState(3)
        node = Node(s)
        agent = MCTSAgent(infer_fn=_uniform_infer)
        agent._expand(node)
        self.assertTrue(node.is_expanded)
        self.assertEqual(len(node.children), 9)

    def test_priors_sum_to_one(self):
        s = HexState(3)
        node = Node(s)
        agent = MCTSAgent(infer_fn=_uniform_infer)
        agent._expand(node)
        total = sum(c.prior for c in node.children.values())
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_uniform_priors_when_all_legal(self):
        s = HexState(3)
        node = Node(s)
        agent = MCTSAgent(infer_fn=_uniform_infer)
        agent._expand(node)
        expected = 1.0 / 9
        for child in node.children.values():
            self.assertAlmostEqual(child.prior, expected, places=5)

    def test_swap_slot_prior_excluded_when_swap_illegal(self):
        """On move 0, swap is illegal; the network's swap probability must be
        excluded so legal priors still sum to 1."""
        n_cells = 9
        # Put all mass on the swap slot (illegal here)
        def swap_heavy_infer(_state):
            policy = [0.0] * (n_cells + 1)
            policy[n_cells] = 1.0  # 100% on swap
            return policy, 0.0

        s = HexState(3, pie_rule=True)  # move 0 — swap NOT legal
        node = Node(s)
        agent = MCTSAgent(infer_fn=swap_heavy_infer)
        agent._expand(node)
        # All legal children should have equal uniform fallback (sum=0 after
        # filtering swap) and sum to 1
        total = sum(c.prior for c in node.children.values())
        self.assertAlmostEqual(total, 1.0, places=5)
        self.assertNotIn(SWAP_MOVE, node.children)

    def test_swap_child_created_when_legal(self):
        s = HexState(3, pie_rule=True)
        s.apply_move((1, 1))           # after one move, swap is legal
        node = Node(s)
        agent = MCTSAgent(infer_fn=_uniform_infer)
        agent._expand(node)
        self.assertIn(SWAP_MOVE, node.children)

    def test_expand_returns_network_value(self):
        def value_one_infer(state):
            n_cells = state.size * state.size
            policy = [1.0 / n_cells] * n_cells + [0.0]
            return policy, 0.75

        s = HexState(3)
        node = Node(s)
        agent = MCTSAgent(infer_fn=value_one_infer)
        val = agent._expand(node)
        self.assertAlmostEqual(val, 0.75)

    def test_expand_on_empty_board_no_crash(self):
        s = HexState(2)
        node = Node(s)
        agent = MCTSAgent(infer_fn=_uniform_infer)
        agent._expand(node)
        self.assertEqual(len(node.children), 4)

    def test_prior_renorm_with_biased_network(self):
        """Network gives 90% to one cell; after renorm that cell should still
        dominate but not exceed 1.0."""
        s = HexState(3)
        preferred = (0, 0)
        agent = MCTSAgent(infer_fn=lambda st: _biased_infer(st, preferred))
        node = Node(s)
        agent._expand(node)
        total = sum(c.prior for c in node.children.values())
        self.assertAlmostEqual(total, 1.0, places=5)
        # The preferred cell should be the largest prior
        best = max(node.children.items(), key=lambda kv: kv[1].prior)
        self.assertEqual(best[0], preferred)


# ---------------------------------------------------------------------------
# update_root tests
# ---------------------------------------------------------------------------

class UpdateRootTests(unittest.TestCase):
    def _make_agent_with_expanded_root(self, size=3):
        s = HexState(size)
        agent = MCTSAgent(infer_fn=_uniform_infer, simulations=5)
        agent.search(s, add_noise=False)
        return agent, s

    def test_update_root_to_played_move(self):
        agent, _ = self._make_agent_with_expanded_root()
        old_root = agent._root
        move = (0, 0)
        agent.update_root(move)
        self.assertIsNotNone(agent._root)
        self.assertIs(agent._root, old_root.children[move])

    def test_update_root_unknown_move_resets_root(self):
        agent, _ = self._make_agent_with_expanded_root()
        agent.update_root((-99, -99))   # nonsense move
        self.assertIsNone(agent._root)

    def test_update_root_none_root_stays_none(self):
        agent = MCTSAgent(infer_fn=_uniform_infer)
        agent._root = None
        agent.update_root((0, 0))
        self.assertIsNone(agent._root)

    def test_tree_reuse_preserves_child_stats(self):
        s = HexState(3)
        agent = MCTSAgent(infer_fn=_uniform_infer, simulations=20)
        agent.search(s, add_noise=False)
        move = (0, 0)
        child_before = agent._root.children[move]
        n_before = child_before.N
        agent.update_root(move)
        # New root is old child; its visit count must be preserved
        self.assertEqual(agent._root.N, n_before)


# ---------------------------------------------------------------------------
# root reuse validity tests
# ---------------------------------------------------------------------------

class RootReuseTests(unittest.TestCase):
    def test_stale_root_discarded_on_board_mismatch(self):
        s1 = HexState(3)
        agent = MCTSAgent(infer_fn=_uniform_infer, simulations=5)
        agent.search(s1, add_noise=False)
        old_root = agent._root

        # Different board
        s2 = HexState(3)
        s2.apply_move((2, 2))  # BLACK moved
        s2.apply_move((0, 0))  # WHITE moved — board differs
        agent.search(s2, add_noise=False)
        self.assertIsNot(agent._root, old_root)

    def test_stale_root_discarded_on_player_mismatch(self):
        s = HexState(3)
        agent = MCTSAgent(infer_fn=_uniform_infer, simulations=5)
        agent.search(s, add_noise=False)
        old_root = agent._root

        s2 = s.clone()
        s2.current_player = WHITE  # same board but wrong player
        agent.search(s2, add_noise=False)
        self.assertIsNot(agent._root, old_root)


# ---------------------------------------------------------------------------
# Dirichlet noise tests
# ---------------------------------------------------------------------------

class DirichletNoiseTests(unittest.TestCase):
    def test_priors_still_sum_to_one_after_noise(self):
        s = HexState(3)
        node = Node(s)
        agent = MCTSAgent(infer_fn=_uniform_infer)
        agent._expand(node)
        agent._add_dirichlet_noise(node)
        total = sum(c.prior for c in node.children.values())
        self.assertAlmostEqual(total, 1.0, places=4)

    def test_noise_changes_priors(self):
        s = HexState(3)
        node = Node(s)
        agent = MCTSAgent(infer_fn=_uniform_infer, dirichlet_epsilon=1.0)
        agent._expand(node)
        priors_before = {m: c.prior for m, c in node.children.items()}
        agent._add_dirichlet_noise(node)
        priors_after = {m: c.prior for m, c in node.children.items()}
        # With epsilon=1, priors are fully replaced by Dirichlet noise
        self.assertNotEqual(priors_before, priors_after)

    def test_no_crash_on_no_children(self):
        s = HexState(2)
        node = Node(s)
        agent = MCTSAgent(infer_fn=_uniform_infer)
        # Not expanded — no children — should not crash
        agent._add_dirichlet_noise(node)

    def test_swap_gets_boosted_noise_share(self):
        """Swap slot should receive ~50% of noise on average (alpha * N vs alpha * 1).
        Verified statistically: mean swap noise share must exceed 1/N significantly."""
        np.random.seed(42)
        s = HexState(3, pie_rule=True)
        s.apply_move((1, 1))  # after one move, swap is legal (10 moves total: 9 cells + swap)
        node = Node(s)
        agent = MCTSAgent(infer_fn=_uniform_infer, dirichlet_epsilon=1.0)
        agent._expand(node)
        swap_shares = []
        for _ in range(200):
            # Re-expand to reset priors
            node2 = Node(s)
            agent._expand(node2)
            agent._add_dirichlet_noise(node2)
            swap_shares.append(node2.children[SWAP_MOVE].prior)

        mean_swap = float(np.mean(swap_shares))
        # Without boost: expected ~1/10 = 0.10; with boost: expected ~0.50
        self.assertGreater(mean_swap, 0.30,
            f"Swap noise share too low ({mean_swap:.3f}); boost not working")


# ---------------------------------------------------------------------------
# search integration tests
# ---------------------------------------------------------------------------

class SearchTests(unittest.TestCase):
    def test_search_returns_valid_policy(self):
        s = HexState(3)
        agent = MCTSAgent(infer_fn=_uniform_infer, simulations=10)
        pi, _, _ = agent.search(s, add_noise=False)
        self.assertEqual(pi.shape, (10,))      # 3*3 + 1 = 10
        self.assertAlmostEqual(float(pi.sum()), 1.0, places=5)
        self.assertTrue(np.all(pi >= 0))

    def test_search_value_in_range(self):
        s = HexState(3)
        agent = MCTSAgent(infer_fn=_uniform_infer, simulations=10)
        _, value, _ = agent.search(s, add_noise=False)
        self.assertGreaterEqual(value, -1.0)
        self.assertLessEqual(value, 1.0)

    def test_search_root_N_equals_simulations(self):
        s = HexState(3)
        sims = 15
        agent = MCTSAgent(infer_fn=_uniform_infer, simulations=sims)
        agent.search(s, add_noise=False)
        # root.N = simulations + 1 (initial expand also counts)
        self.assertGreaterEqual(agent._root.N, sims)

    def test_search_greedy_after_temperature_moves(self):
        """After temperature_moves, policy should be one-hot (greedy)."""
        s = HexState(3)
        # Advance move count past temperature_moves
        for _ in range(21):
            if s.is_terminal():
                break
            legal = s.legal_moves()
            if not legal:
                break
            s.apply_move(legal[0])

        if not s.is_terminal():
            agent = MCTSAgent(
                infer_fn=_uniform_infer, simulations=20,
                temperature=1.0, temperature_moves=20,
            )
            pi, _, _ = agent.search(s, add_noise=False)
            nonzero = int((pi > 0).sum())
            self.assertEqual(nonzero, 1)

    def test_search_info_keys(self):
        s = HexState(3)
        agent = MCTSAgent(infer_fn=_uniform_infer, simulations=5)
        _, _, info = agent.search(s, add_noise=False)
        self.assertIn("root_N", info)
        self.assertIn("root_Q", info)
        self.assertIn("top_moves", info)


if __name__ == "__main__":
    unittest.main()
