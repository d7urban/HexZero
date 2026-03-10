"""
Self-play: run games between the current best model and itself,
feeding completed records into a shared list.

Uses worker threads backed by a single InferenceServer on the target
device (GPU when available).  Tree traversal runs in parallel across
threads; all inference is serialised and batched on the device for
maximum throughput.
"""

import threading

import numpy as np
import torch

from config import HexZeroConfig
from hexzero.game import HexState, SWAP_MOVE
from hexzero.features import extract_features
from hexzero.net import build_net
from hexzero.mcts import MCTSAgent
from hexzero.inference_server import InferenceServer
import hexzero.checkpoint as ckpt_io


def _play_one_game(
    cfg: HexZeroConfig,
    infer_fn,
    board_size: int,
) -> list:
    """
    Play one full self-play game using `infer_fn` for policy/value.
    Returns a list of training samples (original + 180° augmented).
    """
    agent = MCTSAgent(
        infer_fn=infer_fn,
        simulations=cfg.mcts_simulations,
        cpuct=cfg.cpuct,
        dirichlet_alpha=cfg.dirichlet_alpha,
        dirichlet_epsilon=cfg.dirichlet_epsilon,
        temperature=cfg.temperature,
        temperature_moves=cfg.temperature_moves,
    )

    state   = HexState(board_size, pie_rule=cfg.use_pie_rule)
    history = []   # (features, size_norm, pi, player)

    while not state.is_terminal():
        features, size_norm = extract_features(state)
        pi, _, _ = agent.search(state, add_noise=True)
        history.append((features, size_norm, pi, state.current_player))
        size = state.size
        idx  = int(np.random.choice(len(pi), p=pi))
        move = SWAP_MOVE if idx == size * size else (idx // size, idx % size)
        agent.update_root(move)
        state.apply_move(move)

    winner = state.winner()
    samples: list = []
    aug_samples: list = []
    for features, size_norm, pi, player in history:
        value = 1.0 if winner == player else -1.0
        samples.append({
            "features":   features,
            "policy":     pi,
            "value":      value,
            "board_size": board_size,
            "size_norm":  size_norm,
        })
        aug_features = np.rot90(features, 2, axes=(1, 2)).copy()
        # Reverse only the board positions; preserve the swap slot at the end
        n_board  = board_size * board_size
        aug_pi   = np.concatenate([pi[:n_board][::-1], pi[n_board:]]).copy()
        aug_samples.append({
            "features":   aug_features,
            "policy":     aug_pi,
            "value":      value,
            "board_size": board_size,
            "size_norm":  size_norm,
        })

    return samples + aug_samples


def run_self_play_parallel(
    cfg: HexZeroConfig,
    checkpoint_path: str,
    device: torch.device,
    board_size: int,
    total_games: int,
    progress_callback=None,
    stop_event=None,
) -> list:
    """
    Load `checkpoint_path` into an InferenceServer on `device`, then
    play `total_games` games across `cfg.num_self_play_workers` threads.

    progress_callback: optional callable(games_done, games_total).
    """
    net  = build_net(cfg, device)
    data = ckpt_io.load(checkpoint_path, device)
    ckpt_io.load_weights(net, data["model_state"])

    server = InferenceServer(net, device)

    all_samples: list = []
    lock      = threading.Lock()
    completed = [0]

    games_per_worker = max(1, total_games // cfg.num_self_play_workers)
    remainder = total_games - games_per_worker * cfg.num_self_play_workers

    def worker(n_games: int) -> None:
        def infer_fn(state: HexState):
            features, size_norm = extract_features(state)
            return server.infer(features, size_norm)

        for _ in range(n_games):
            if stop_event is not None and stop_event.is_set():
                break
            try:
                game_samples = _play_one_game(cfg, infer_fn, board_size)
            except Exception:
                game_samples = []
            with lock:
                all_samples.extend(game_samples)
                completed[0] += 1
                if progress_callback is not None:
                    progress_callback(completed[0], total_games)

    threads = []
    for i in range(cfg.num_self_play_workers):
        n = games_per_worker + (1 if i < remainder else 0)
        t = threading.Thread(target=worker, args=(n,), daemon=True)
        threads.append(t)
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    server.stop()
    return all_samples
