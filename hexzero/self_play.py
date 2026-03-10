"""
Self-play worker: runs games between the current best model and itself,
feeding completed game records into a shared queue.

Runs in a subprocess to avoid the Python GIL.  One worker = one game at
a time; spawn cfg.num_self_play_workers of them in parallel.
"""

import multiprocessing as mp
import queue as _queue
import numpy as np
import torch

from config import HexZeroConfig
from hexzero.game import HexState
from hexzero.features import extract_features
from hexzero.net import HexNet, build_net
from hexzero.mcts import MCTSAgent
import hexzero.checkpoint as ckpt_io


def _make_infer_fn(net: HexNet, device: torch.device):
    """Return a callable (state) -> (policy np, value float)."""
    net.eval()

    def infer_fn(state: HexState):
        features, size_norm = extract_features(state)
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
        s = torch.tensor([[size_norm]], dtype=torch.float32).to(device)
        with torch.no_grad():
            log_pi, v = net(x, s)
        policy = torch.exp(log_pi).squeeze(0).cpu().numpy()
        value  = float(v.item())
        return policy, value

    return infer_fn


def _play_one_game(
    cfg: HexZeroConfig,
    net: HexNet,
    device: torch.device,
    board_size: int,
) -> list:
    """
    Play a full self-play game.  Returns a list of training samples:
        [{"features": ..., "policy": ..., "value": ...,
          "board_size": ..., "size_norm": ...}, ...]
    """
    infer_fn = _make_infer_fn(net, device)
    agent = MCTSAgent(
        infer_fn=infer_fn,
        simulations=cfg.mcts_simulations,
        cpuct=cfg.cpuct,
        dirichlet_alpha=cfg.dirichlet_alpha,
        dirichlet_epsilon=cfg.dirichlet_epsilon,
        temperature=cfg.temperature,
        temperature_moves=cfg.temperature_moves,
    )

    state = HexState(board_size)
    history = []  # (features, size_norm, pi, player)

    while not state.is_terminal():
        features, size_norm = extract_features(state)
        pi, _, _ = agent.search(state, add_noise=True)

        history.append((features, size_norm, pi, state.current_player))

        # Sample move from pi
        size = state.size
        idx  = int(np.random.choice(size * size, p=pi))
        move = (idx // size, idx % size)
        agent.update_root(move)
        state.apply_move(move)

    winner = state.winner()
    samples = []
    for features, size_norm, pi, player in history:
        value = 1.0 if winner == player else -1.0
        samples.append({
            "features":   features,
            "policy":     pi,
            "value":      value,
            "board_size": board_size,
            "size_norm":  size_norm,
        })

    # Data augmentation: 180° rotation
    aug_samples = []
    for features, size_norm, pi, player in history:
        # Rotate features and policy
        aug_features = np.rot90(features, 2, axes=(1, 2)).copy()
        aug_pi = pi[::-1].copy()  # 180° rotation reverses cell order in row-major
        value = 1.0 if winner == player else -1.0
        aug_samples.append({
            "features":   aug_features,
            "policy":     aug_pi,
            "value":      value,
            "board_size": board_size,
            "size_norm":  size_norm,
        })

    return samples + aug_samples


def self_play_worker(
    cfg: HexZeroConfig,
    checkpoint_path: str,
    result_queue: mp.Queue,
    board_size: int,
    games_to_play: int,
    worker_id: int = 0,
) -> None:
    """
    Entry point for a subprocess worker.
    Plays `games_to_play` games and puts records into `result_queue`.
    """
    # Each worker runs in its own process; pin to 1 thread so N parallel
    # workers share cores instead of all competing for the full thread pool.
    torch.set_num_threads(1)

    device = torch.device("cpu")  # workers always on CPU
    net = build_net(cfg, device)

    try:
        data = ckpt_io.load(checkpoint_path, device)
        net.load_state_dict(data["model_state"])
    except Exception as e:
        result_queue.put({"error": f"checkpoint load failed: {e}", "worker_id": worker_id})
        return

    for game_idx in range(games_to_play):
        try:
            samples = _play_one_game(cfg, net, device, board_size)
            result_queue.put(samples)
        except Exception as e:
            result_queue.put({"error": str(e), "worker_id": worker_id, "game_idx": game_idx})


def run_self_play_parallel(
    cfg: HexZeroConfig,
    checkpoint_path: str,
    board_size: int,
    total_games: int,
    progress_callback=None,
) -> list:
    """
    Spawn worker processes, collect all game samples, return flat list.

    progress_callback: optional callable(games_done: int, games_total: int)
    called after each completed game, from the calling thread.
    """
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    games_per_worker = max(1, total_games // cfg.num_self_play_workers)
    remainder = total_games - games_per_worker * cfg.num_self_play_workers

    workers = []
    for i in range(cfg.num_self_play_workers):
        n = games_per_worker + (1 if i < remainder else 0)
        p = ctx.Process(
            target=self_play_worker,
            args=(cfg, checkpoint_path, result_queue, board_size, n, i),
            daemon=True,
        )
        p.start()
        workers.append(p)

    all_samples = []
    collected = 0
    # 10-minute per-game timeout prevents an infinite hang if a worker crashes
    # before putting anything on the queue.
    timeout_s = 600
    while collected < total_games:
        try:
            item = result_queue.get(timeout=timeout_s)
        except _queue.Empty:
            break   # worker probably died; return whatever we have
        if isinstance(item, list):
            all_samples.extend(item)
            collected += 1
            if progress_callback is not None:
                progress_callback(collected, total_games)
        # error dicts are silently counted so we don't spin forever
        else:
            collected += 1

    for p in workers:
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()

    return all_samples
