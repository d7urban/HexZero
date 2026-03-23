"""
tournament.py — Round-robin tournament with Glicko-2 ratings.

Every model plays every other model once per round.  All matches within a
round run in parallel.  Glicko-2 ratings are updated after each round and
persisted to BOTB.rating.

Usage:
    python tournament.py checkpoints/ checkpoints2/ checkpoints3/
    python tournament.py checkpoints/ checkpoints2/ --rounds 3 --games 10
    python tournament.py checkpoints/ checkpoints2/ --sims 200 --board-size 11
"""

import argparse
import hashlib
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor

import torch

import hexzero.checkpoint as ckpt_io
from config import HexZeroConfig
from hexzero.arena import _make_infer_fn, _play_game
from hexzero.game import BLACK, WHITE, HexState
from hexzero.mcts import MCTSAgent
from hexzero.net import build_net


# ---------------------------------------------------------------------------
# Glicko-2
# ---------------------------------------------------------------------------

_SCALE = 173.7178   # converts between internal and display scale (r=1500, RD=350)
_TAU   = 0.5        # system constant; controls how fast volatility can change (0.3–1.2)


class Glicko2:
    """Single-player Glicko-2 rating state.  Internal scale: mu=0, phi≈2.015."""

    def __init__(self, mu: float = 0.0, phi: float = 2.0145, sigma: float = 0.06):
        self.mu    = mu
        self.phi   = phi
        self.sigma = sigma

    @property
    def rating(self) -> float:
        return _SCALE * self.mu + 1500

    @property
    def rd(self) -> float:
        return _SCALE * self.phi

    @staticmethod
    def _g(phi: float) -> float:
        return 1.0 / math.sqrt(1 + 3 * phi ** 2 / math.pi ** 2)

    @staticmethod
    def _E(mu: float, mu_j: float, phi_j: float) -> float:
        return 1.0 / (1 + math.exp(-Glicko2._g(phi_j) * (mu - mu_j)))

    def update(self, opp_snaps: list[tuple[float, float]], scores: list[float]) -> None:
        """Batch update for one rating period.

        opp_snaps : list of (mu, phi) snapshots taken before this period.
        scores    : per-game result (1=win, 0=loss) for each game.
        """
        if not opp_snaps:
            # No games played: inflate RD by volatility.
            self.phi = math.sqrt(self.phi ** 2 + self.sigma ** 2)
            return

        mu, phi, sigma = self.mu, self.phi, self.sigma
        gs = [self._g(phi_j) for _, phi_j in opp_snaps]
        Es = [self._E(mu, mu_j, phi_j) for mu_j, phi_j in opp_snaps]

        v_inv = sum(g ** 2 * E * (1 - E) for g, E in zip(gs, Es))
        if v_inv < 1e-12:
            return
        v     = 1.0 / v_inv
        delta = v * sum(g * (s - E) for g, E, s in zip(gs, Es, scores))

        # Volatility update — Illinois algorithm (Glickman 2012, step 5.4–5.7).
        a = math.log(sigma ** 2)

        def f(x: float) -> float:
            ex = math.exp(x)
            d2 = phi ** 2 + v + ex
            return ex * (delta ** 2 - d2) / (2 * d2 ** 2) - (x - a) / _TAU ** 2

        A, fA = a, f(a)
        if delta ** 2 > phi ** 2 + v:
            B = math.log(delta ** 2 - phi ** 2 - v)
        else:
            k = 1
            while f(a - k * _TAU) < 0:
                k += 1
            B = a - k * _TAU
        fB = f(B)

        for _ in range(200):
            C = A + (A - B) * fA / (fB - fA)
            fC = f(C)
            if fC * fB < 0:
                A, fA = B, fB
            else:
                fA *= 0.5
            B, fB = C, fC
            if abs(B - A) < 1e-6:
                break

        new_sigma = math.exp(A / 2)
        phi_star  = math.sqrt(phi ** 2 + new_sigma ** 2)
        new_phi   = 1.0 / math.sqrt(1.0 / phi_star ** 2 + 1.0 / v)
        new_mu    = mu + new_phi ** 2 * sum(
            g * (s - E) for g, E, s in zip(gs, Es, scores))

        self.mu    = new_mu
        self.phi   = new_phi
        self.sigma = new_sigma


# ---------------------------------------------------------------------------
# BOTB.rating persistence
# ---------------------------------------------------------------------------

_RATING_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "BOTB.rating")


def _load_ratings() -> dict:
    if os.path.exists(_RATING_FILE):
        with open(_RATING_FILE) as f:
            return json.load(f)
    return {}


def _save_ratings(ratings: dict) -> None:
    tmp = _RATING_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(ratings, f, indent=2)
    os.replace(tmp, _RATING_FILE)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rating_for(sha: str, label: str, ratings: dict) -> tuple[Glicko2, str]:
    """Return (Glicko2, key) for this sha256, allocating a new key if needed."""
    for key, entry in ratings.items():
        if key.startswith("_"):
            continue
        if entry.get("sha256") == sha:
            return Glicko2(mu=entry["mu"], phi=entry["phi"], sigma=entry["sigma"]), key

    next_id = ratings.get("_next_id", 1)
    key = str(next_id)
    ratings["_next_id"] = next_id + 1
    return Glicko2(), key


# ---------------------------------------------------------------------------
# Player
# ---------------------------------------------------------------------------

class Player:
    def __init__(self, label: str, net, board_size: int, iteration,
                 sha256: str, rating_key: str, glicko: Glicko2):
        self.label      = label
        self.net        = net
        self.board_size = board_size
        self.iteration  = iteration
        self.sha256     = sha256
        self.rating_key = rating_key
        self.glicko     = glicko


# ---------------------------------------------------------------------------
# Checkpoint loading & match play
# ---------------------------------------------------------------------------

def _label(checkpoint_dir: str) -> str:
    return os.path.basename(os.path.abspath(checkpoint_dir))


def _load_player(checkpoint_dir: str, cfg: HexZeroConfig,
                 device: torch.device, ratings: dict) -> Player:
    path = ckpt_io.best_checkpoint_path(checkpoint_dir)
    if path is None:
        raise FileNotFoundError(f"No best.pt found in {checkpoint_dir!r}")
    sha         = _sha256(path)
    data        = ckpt_io.load(path, device)
    net         = build_net(cfg, device, compile=False)
    ckpt_io.load_weights(net, data["model_state"])
    net.eval()
    board_size  = data.get("metrics", {}).get("board_size") or cfg.initial_board_size
    iteration   = data.get("iteration", "?")
    label       = _label(checkpoint_dir)
    glicko, key = _rating_for(sha, label, ratings)
    return Player(label, net, board_size, iteration, sha, key, glicko)


def _play_match(
    pa: Player, pb: Player,
    cfg: HexZeroConfig, board_size: int, games: int, sims: int,
) -> tuple[int, int]:
    """Play `games` alternating-colour games. Returns (pa_wins, pb_wins)."""
    device  = next(pa.net.parameters()).device
    infer_a = _make_infer_fn(pa.net, device)
    infer_b = _make_infer_fn(pb.net, device)

    a_wins = b_wins = 0
    for game_idx in range(games):
        if game_idx % 2 == 0:
            black_infer, white_infer, a_color = infer_a, infer_b, BLACK
        else:
            black_infer, white_infer, a_color = infer_b, infer_a, WHITE

        black_agent = MCTSAgent(infer_fn=black_infer, simulations=sims,
                                cpuct=cfg.cpuct, dirichlet_epsilon=0.0,
                                temperature_moves=cfg.arena_temperature_moves)
        white_agent = MCTSAgent(infer_fn=white_infer, simulations=sims,
                                cpuct=cfg.cpuct, dirichlet_epsilon=0.0,
                                temperature_moves=cfg.arena_temperature_moves)

        state  = HexState(board_size, pie_rule=cfg.use_pie_rule)
        winner = _play_game(state, black_agent, white_agent)
        if winner == a_color:
            a_wins += 1
        else:
            b_wins += 1

    return a_wins, b_wins


# ---------------------------------------------------------------------------
# Glicko-2 batch update
# ---------------------------------------------------------------------------

def _flush_ratings(players: list[Player], ratings: dict) -> None:
    for p in players:
        ratings[p.rating_key] = {
            "label":  p.label,
            "sha256": p.sha256,
            "mu":     p.glicko.mu,
            "phi":    p.glicko.phi,
            "sigma":  p.glicko.sigma,
            "rating": p.glicko.rating,
            "rd":     p.glicko.rd,
        }


def _update_ratings(
    round_results: list[tuple[Player, Player, int, int]],
    all_players:   list[Player],
) -> None:
    """Apply one Glicko-2 rating period using all games from this round."""
    snaps = {p.label: (p.glicko.mu, p.glicko.phi) for p in all_players}

    opp_snaps_by: dict[str, list[tuple[float, float]]] = {p.label: [] for p in all_players}
    scores_by:    dict[str, list[float]]               = {p.label: [] for p in all_players}

    for pa, pb, aw, bw in round_results:
        snap_a, snap_b = snaps[pa.label], snaps[pb.label]
        for _ in range(aw):
            opp_snaps_by[pa.label].append(snap_b);  scores_by[pa.label].append(1.0)
            opp_snaps_by[pb.label].append(snap_a);  scores_by[pb.label].append(0.0)
        for _ in range(bw):
            opp_snaps_by[pb.label].append(snap_a);  scores_by[pb.label].append(1.0)
            opp_snaps_by[pa.label].append(snap_b);  scores_by[pa.label].append(0.0)

    for p in all_players:
        p.glicko.update(opp_snaps_by[p.label], scores_by[p.label])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Round-robin tournament with Glicko-2 ratings between best checkpoints")
    p.add_argument("dirs",         nargs="+", metavar="DIR",
                   help="Checkpoint directories to include")
    p.add_argument("--rounds",     type=int, default=1,
                   help="How many times to run the full round-robin (default: 1)")
    p.add_argument("--games",      type=int, default=10,
                   help="Games per match, should be even (default: 10)")
    p.add_argument("--sims",       type=int, default=None,
                   help="MCTS simulations per move (default: from config for board size)")
    p.add_argument("--board-size", type=int, default=None,
                   help="Board size override (default: detected from checkpoints)")
    args = p.parse_args()

    cfg    = HexZeroConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    ratings = _load_ratings()

    players: list[Player] = []
    for d in args.dirs:
        if not os.path.isdir(d):
            print(f"Error: checkpoint directory '{d}' does not exist — skipping.")
            continue
        try:
            player = _load_player(d, cfg, device, ratings)
        except FileNotFoundError as e:
            print(f"Error: {e} — skipping.")
            continue
        players.append(player)
        print(f"  #{player.rating_key:<4} {player.label:30s}  iter {player.iteration}  "
              f"board {player.board_size}×{player.board_size}  "
              f"rating {player.glicko.rating:.0f} ± {player.glicko.rd:.0f}")

    if len(players) < 2:
        print("\nNeed at least two checkpoint directories.")
        return

    sizes = {p.board_size for p in players}
    if args.board_size is not None:
        board_size = args.board_size
    elif len(sizes) == 1:
        board_size = sizes.pop()
    else:
        board_size = min(sizes)
        print(f"\nWarning: mixed board sizes {sorted(sizes)} — "
              f"using smallest ({board_size}×{board_size})")

    sims   = args.sims if args.sims is not None else cfg.sims_for_size(board_size)
    pairs  = [(players[i], players[j])
              for i in range(len(players))
              for j in range(i + 1, len(players))]

    col = max(len(p.label) for p in players) + 2
    sep = "─" * (col + 36)

    print(f"\nBoard: {board_size}×{board_size}  |  Sims: {sims}  |  "
          f"Games/match: {args.games}  |  "
          f"Matches/round: {len(pairs)}  |  Rounds: {args.rounds}\n")

    t_total = time.monotonic()
    for rnd in range(1, args.rounds + 1):
        print(sep)
        print(f"Round {rnd}/{args.rounds}  ({len(pairs)} matches in parallel)")
        print(sep)

        t_round = time.monotonic()
        futures: dict = {}
        with ThreadPoolExecutor(max_workers=len(pairs)) as pool:
            for pa, pb in pairs:
                fut = pool.submit(_play_match, pa, pb, cfg, board_size, args.games, sims)
                futures[fut] = (pa, pb)

        round_results: list[tuple[Player, Player, int, int]] = []
        results_map = {(pa.label, pb.label): fut.result() for fut, (pa, pb) in futures.items()}
        for pa, pb in pairs:
            aw, bw = results_map[(pa.label, pb.label)]
            print(f"  {pa.label} vs {pb.label}:  "
                  f"{pa.label} {aw}/{args.games}  {pb.label} {bw}/{args.games}")
            round_results.append((pa, pb, aw, bw))
        print(f"  Round time: {time.monotonic() - t_round:.1f}s")

        _update_ratings(round_results, players)
        _flush_ratings(players, ratings)
        _save_ratings(ratings)

        ranked = sorted(players, key=lambda pl: -pl.glicko.rating)
        print(f"\n  {'Model':<{col}} {'Rating':>7}  {'RD':>6}")
        for pl in ranked:
            print(f"  {pl.label:<{col}} {pl.glicko.rating:>7.0f}  {pl.glicko.rd:>6.0f}")
        print()

    ranked = sorted(players, key=lambda pl: -pl.glicko.rating)
    print("═" * (col + 36))
    print("FINAL STANDINGS  (ranked by Glicko-2 rating)")
    print("═" * (col + 36))
    print(f"  {'#':<3} {'Model':<{col}} {'Rating':>7} ± {'RD':<6}  σ")
    print("─" * (col + 36))
    for i, pl in enumerate(ranked, 1):
        print(f"  {i:<3} {pl.label:<{col}} "
              f"{pl.glicko.rating:>7.0f} ± {pl.glicko.rd:<6.0f}  {pl.glicko.sigma:.4f}")
    print("═" * (col + 36))
    print(f"\nWinner: {ranked[0].label}  "
          f"(rating {ranked[0].glicko.rating:.0f} ± {ranked[0].glicko.rd:.0f})  "
          f"—  total time: {time.monotonic() - t_total:.1f}s")


if __name__ == "__main__":
    main()
