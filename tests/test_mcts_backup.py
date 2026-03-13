import unittest

from hexzero.game import HexState
from hexzero.mcts import MCTSAgent, Node


def _uniform_infer(state: HexState):
    n_cells = state.size * state.size
    legal = state.legal_moves()
    policy = [0.0] * (n_cells + 1)
    if legal:
        p = 1.0 / len(legal)
        for move in legal:
            if move == (-1, -1):
                policy[n_cells] = p
            else:
                r, c = move
                policy[r * state.size + c] = p
    return policy, 0.0


class BackupSignInvariantTests(unittest.TestCase):
    def test_backup_stores_child_q_in_parent_player_frame(self):
        # Construct root -> A where A is WHITE-to-move and leaf value is +1 for WHITE.
        root = Node()
        a_child = Node()

        agent = MCTSAgent(infer_fn=_uniform_infer)
        path = [(a_child, (0, 0))]

        agent._backup(root, path, leaf_value=1.0)

        # From root (BLACK) choosing A should be bad if A is winning for WHITE.
        self.assertEqual(a_child.N, 1)
        self.assertEqual(a_child.W, -1.0)
        self.assertEqual(a_child.Q, -1.0)
        self.assertEqual(root.W, -1.0)

    def test_select_child_avoids_losing_move_for_root_player(self):
        root = Node()
        root.N = 2

        good_child = Node(prior=0.5)
        bad_child = Node(prior=0.5)

        # Equal visits and priors; Q drives the choice.
        good_child.N, good_child.W = 1, 1.0   # good for root player
        bad_child.N, bad_child.W = 1, -1.0    # bad for root player

        root.children = {(0, 0): good_child, (0, 1): bad_child}

        agent = MCTSAgent(infer_fn=_uniform_infer)
        move, _ = agent._select_child(root)
        self.assertEqual(move, (0, 0))


if __name__ == "__main__":
    unittest.main()

