"""
Arena: pit two checkpoints against each other to decide whether the
candidate model should replace the current champion.

Games alternate colours to neutralise the first-move advantage.
"""

import torch
import numpy as np

from config import HexZeroConfig
from hexzero.game import HexState, BLACK, WHITE
from hexzero.features import extract_features
from hexzero.net import HexNet, build_net
from hexzero.mcts import MCTSAgent
import hexzero.checkpoint as ckpt_io


def _make_infer_fn(net: HexNet, device: torch.device):
    net.eval()

    def infer_fn(state: HexState):
        features, size_norm = extract_features(state)
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
        s = torch.tensor([[size_norm]], dtype=torch.float32).to(device)
        with torch.no_grad():
            log_pi, v = net(x, s)
        return torch.exp(log_pi).squeeze(0).cpu().numpy(), float(v.item())

    return infer_fn


def _play_game(
    state: HexState,
    black_agent: MCTSAgent,
    white_agent: MCTSAgent,
) -> int:
    """
    Play a game to completion.  black_agent plays BLACK, white_agent plays WHITE.
    Returns the winner (BLACK or WHITE constant).
    """
    from hexzero.game import BLACK as BLK, WHITE as WHT
    agents = {BLK: black_agent, WHT: white_agent}

    while not state.is_terminal():
        player  = state.current_player
        agent   = agents[player]
        size    = state.size
        pi, _, _ = agent.search(state, add_noise=False)
        idx     = int(np.argmax(pi))  # greedy in arena
        move    = (idx // size, idx % size)
        agent.update_root(move)
        # Update the other agent's tree too
        other = agents[-player]
        other.update_root(move)
        state.apply_move(move)

    return state.winner()


def run_arena(
    candidate_path: str,
    champion_path: str,
    cfg: HexZeroConfig,
    board_size: int,
) -> tuple[int, int, int]:
    """
    Run cfg.arena_games games between candidate and champion.
    Returns (candidate_wins, champion_wins, draws).
    """
    device = torch.device("cpu")

    candidate = build_net(cfg, device)
    champion  = build_net(cfg, device)

    cand_data = ckpt_io.load(candidate_path, device)
    champ_data = ckpt_io.load(champion_path, device)
    candidate.load_state_dict(cand_data["model_state"])
    champion.load_state_dict(champ_data["model_state"])

    cand_infer  = _make_infer_fn(candidate, device)
    champ_infer = _make_infer_fn(champion,  device)

    cand_wins = 0
    champ_wins = 0
    draws = 0

    for game_idx in range(cfg.arena_games):
        # Alternate who plays which colour
        if game_idx % 2 == 0:
            black_infer, white_infer = cand_infer, champ_infer
            cand_color = BLACK
        else:
            black_infer, white_infer = champ_infer, cand_infer
            cand_color = WHITE

        black_agent = MCTSAgent(infer_fn=black_infer, simulations=cfg.mcts_simulations,
                                cpuct=cfg.cpuct, dirichlet_epsilon=0.0)
        white_agent = MCTSAgent(infer_fn=white_infer, simulations=cfg.mcts_simulations,
                                cpuct=cfg.cpuct, dirichlet_epsilon=0.0)

        state  = HexState(board_size)
        winner = _play_game(state, black_agent, white_agent)

        if winner == cand_color:
            cand_wins += 1
        elif winner is not None:
            champ_wins += 1
        else:
            draws += 1

    return cand_wins, champ_wins, draws


def candidate_is_better(
    cand_wins: int,
    champ_wins: int,
    total: int,
    threshold: float,
) -> bool:
    if total == 0:
        return False
    return cand_wins / total >= threshold
