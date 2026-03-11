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

import hexzero.checkpoint as ckpt_io
from config import HexZeroConfig
from hexzero.features import extract_features
from hexzero.game import SWAP_MOVE, HexState
from hexzero.inference_server import InferenceServer
from hexzero.mcts import MCTSAgent
from hexzero.net import build_net


def _play_one_game(
    cfg: HexZeroConfig,
    infer_fn,
    board_size: int,
) -> tuple[list, bool]:
    """
    Play one full self-play game using `infer_fn` for policy/value.
    Returns (samples, swapped) where samples is a list of training records
    (original + 180° augmented) and swapped indicates whether WHITE used the
    pie rule swap move.
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
    swapped = False

    while not state.is_terminal():
        features, size_norm = extract_features(state)
        pi, _, _ = agent.search(state, add_noise=True)
        history.append((features, size_norm, pi, state.current_player))
        size = state.size
        idx  = int(np.random.choice(len(pi), p=pi))
        move = SWAP_MOVE if idx == size * size else (idx // size, idx % size)
        if move == SWAP_MOVE:
            swapped = True
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

    return samples + aug_samples, swapped


def run_self_play_parallel(
    cfg: HexZeroConfig,
    checkpoint_path: str,
    device: torch.device,
    board_size: int,
    total_games: int,
    progress_callback=None,
    stop_event=None,
) -> tuple[list, int, int]:
    """
    Load `checkpoint_path` into an InferenceServer on `device`, then
    play `total_games` games across `cfg.num_self_play_workers` threads.

    progress_callback: optional callable(games_done, games_total).

    Returns (all_samples, swap_games, games_played) where swap_games is the
    number of games in which WHITE used the pie rule swap move.
    """
    net  = build_net(cfg, device, compile=False)
    data = ckpt_io.load(checkpoint_path, device)
    ckpt_io.load_weights(net, data["model_state"])

    server = InferenceServer(net, device)

    all_samples: list = []
    lock        = threading.Lock()
    completed   = [0]
    swap_count  = [0]
    first_error: list[BaseException] = []   # at most one entry

    n_workers        = min(cfg.num_self_play_workers, total_games)
    games_per_worker = total_games // n_workers if n_workers else 0
    remainder        = total_games % n_workers if n_workers else 0

    def worker(n_games: int) -> None:
        def infer_fn(state: HexState):
            features, size_norm = extract_features(state)
            return server.infer(features, size_norm)

        for _ in range(n_games):
            if stop_event is not None and stop_event.is_set():
                break
            game_samples, swapped = _play_one_game(cfg, infer_fn, board_size)
            with lock:
                all_samples.extend(game_samples)
                if swapped:
                    swap_count[0] += 1
                completed[0] += 1
                if progress_callback is not None:
                    progress_callback(completed[0], total_games)

    def worker_wrapper(n_games: int) -> None:
        try:
            worker(n_games)
        except Exception as exc:
            with lock:
                if not first_error:
                    first_error.append(exc)

    threads = []
    for i in range(n_workers):
        n = games_per_worker + (1 if i < remainder else 0)
        t = threading.Thread(target=worker_wrapper, args=(n,), daemon=True)
        threads.append(t)
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    server.stop()

    if first_error:
        raise RuntimeError(f"Self-play worker failed: {first_error[0]}") from first_error[0]

    return all_samples, swap_count[0], completed[0]
