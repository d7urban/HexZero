# HexZero

An AlphaZero-style self-play training suite for the board game [Hex](https://en.wikipedia.org/wiki/Hex_(board_game)), built as a clean, readable showcase of the core ideas behind AlphaZero.

Hex is an ideal vehicle for this: the rules are simple (connect your two sides before your opponent connects theirs), draws are impossible, and the game has enough depth that a random player loses badly to even shallow search — so you can actually watch the agent improve.

---

## Why Hex?

| Property | Hex | Go | Chess |
|---|---|---|---|
| Rules complexity | Minimal | Low | High |
| Draw possible | No | Ko rules | Yes |
| Perfect information | Yes | Yes | Yes |
| Scalable difficulty | 5×5 → 19×19 | Fixed 19×19 | Fixed 8×8 |
| Rotational symmetry | 180° | 90° | — |

Tic-tac-toe is too small (negamax solves it trivially); Go and Chess have rule overhead that obscures the ML pipeline. Hex hits the sweet spot.

---

## Architecture

### Network

A single size-agnostic ResNet shared across all board sizes:

```
Input: (B, 8, H, W)   — 8 feature planes
  │
  ▼
Stem: Conv2d(3×3, 128) → BN → ReLU
  │
  ▼
Tower: 8 × ResBlock(128 filters, 3×3, skip connection)
  │
  ├─▶ Policy head: Conv2d(1×1, 1) → reshape (B, H×W)
  │                concat swap logit → (B, H×W+1) → log-softmax
  │
  └─▶ Value head:  GlobalAvgPool → concat(size_scalar) → FC(256) → ReLU → FC(1) → tanh
```

**Size-agnostic trick**: global average pooling before the FC layers means spatial dimensions never appear in any weight. A normalised board-size scalar is appended to the pooled vector so the network can calibrate value estimates across sizes.

**Swap logit**: the policy head outputs one extra scalar (from the global pool) for the pie-rule swap action, so the policy distribution is over H×W+1 actions.

The network is compiled with `torch.compile` at build time for fused kernel execution, with TF32 tensor cores enabled on Ampere+ GPUs.

### Input Feature Planes

| Plane | Description |
|---|---|
| `black_stones` | 1 where BLACK stone placed |
| `white_stones` | 1 where WHITE stone placed |
| `black_2bridge` | Empty cells that are carriers of a BLACK two-bridge |
| `white_2bridge` | Empty cells that are carriers of a WHITE two-bridge |
| `black_edge_dist` | min BFS distance to either of BLACK's two target edges (normalised) |
| `white_edge_dist` | min BFS distance to either of WHITE's two target edges (normalised) |
| `black_components` | Connected component label for BLACK stones (normalised) |
| `white_components` | Connected component label for WHITE stones (normalised) |

The edge-distance planes use the minimum distance to **either** target edge (not just one), giving a symmetric, non-monotonic signal that avoids biasing the untrained network toward a particular edge of the board.

Two-bridges are the fundamental tactical unit of Hex — a pair of stones with two shared empty neighbours forming a virtually unbreakable connection. Explicit planes let the network learn connection-based reasoning far faster than from raw board state alone.

### Pie Rule (Swap Rule)

After BLACK's first move, WHITE may invoke the swap rule: they take BLACK's stone as their own and play as BLACK for the rest of the game. This eliminates the first-mover advantage.

The swap action is the H×W+1-th output of the policy head — a global scalar logit, since the decision requires reasoning about the whole position rather than a specific cell. Controlled by `use_pie_rule` in config.

### MCTS

Standard PUCT (AlphaZero variant):

```
PUCT(s, a) = Q(s, a) + c_puct · P(s, a) · √N(s) / (1 + N(s, a))
```

- Dirichlet noise at the root during self-play (α=0.3, ε=0.25)
- Temperature-based move sampling for the first 20 moves, greedy thereafter
- Subtree reuse across moves within a game
- Policy vector is always H×W+1; swap slot is masked out when not legal

### Self-play

Games are played across worker threads (default: cpu\_count / 2) sharing a single **InferenceServer** — a background thread that collects MCTS state-evaluation requests from all workers, batches them, runs one GPU forward pass, and distributes results. This eliminates model duplication in VRAM and keeps the GPU saturated without spawning subprocesses.

### Training Loop

```
repeat:
  1. Self-play      — worker threads play games via shared InferenceServer
  2. Augment        — each game doubled via 180° rotation
  3. Buffer         — samples added to gzip-compressed replay buffer (100k cap, persisted)
  4. Train          — 200 gradient steps, dynamic batching by board size
  5. Arena          — candidate vs champion (40 games, alternating colours)
  6. Promote        — replace champion if candidate wins ≥ 55% of games
```

The replay buffer is saved to `checkpoints/replay_buffer.pt.gz` after each self-play phase and reloaded on restart, eliminating the cold-start penalty.

### Curriculum

Training starts on 7×7. When the agent achieves ≥ 60% win rate in the arena against the prior champion on the current size, the next size (9×9, then 11×11) is unlocked. Knowledge transfers because the convolutional weights are size-agnostic. The current board size and last arena win rate are persisted in `checkpoints/training_state.json` so the curriculum ladder is restored correctly on restart.

---

## Project Structure

```
HexZero/
├── main.py                  — entry point (GUI + headless modes)
├── play.py                  — human vs best checkpoint (interactive)
├── config.py                — HexZeroConfig dataclass (all hyperparameters)
├── requirements.txt
└── hexzero/
    ├── game.py              — Hex engine: union-find, winning path, pie rule
    ├── features.py          — 8-plane feature tensor construction
    ├── net.py               — HexNet: ResNet + policy/value heads, torch.compile
    ├── mcts.py              — PUCT MCTS: Node, MCTSAgent, tree reuse, swap action
    ├── inference_server.py  — batched GPU inference server for self-play workers
    ├── replay_buffer.py     — thread-safe circular buffer, gzip persistence
    ├── self_play.py         — thread-based self-play with shared InferenceServer
    ├── trainer.py           — gradient updates, loss computation, metrics
    ├── arena.py             — champion vs candidate evaluation
    ├── checkpoint.py        — atomic saves, rolling window, best.pt, state helpers
    └── gui/
        ├── signals.py           — Qt signal carriers (cross-thread comms)
        ├── board_widget.py      — pointy-top hex board, policy heatmap, winning path
        ├── chart_widget.py      — live loss / accuracy curves (pyqtgraph)
        ├── mcts_widget.py       — top-moves table with visit count shading
        ├── stats_widget.py      — stats bar: iteration, self-play, arena, curriculum
        ├── demo_worker.py       — background thread playing live demo games
        └── main_window.py       — main window, toolbar, training/demo QThreads
```

Dependency rule: `game.py` and `net.py` are sinks (no intra-project imports). `mcts.py` receives `infer_fn: Callable` so it never imports `net.py` directly — MCTS is testable without a GPU.

---

## Installation

```bash
git clone https://github.com/d7urban/HexZero.git
cd HexZero
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python ≥ 3.11, PyTorch ≥ 2.0 (for `torch.compile`).

---

## Usage

### Training GUI

```bash
python main.py
python main.py --board-size 9
python main.py --checkpoint checkpoints/best.pt
```

The GUI shows:
- **Board** — live demo games with policy heatmap, last-move highlight, and winning-path indicator (green dots)
- **Charts** — policy loss, value loss, and policy accuracy in real time
- **MCTS viewer** — top 5 moves by visit count, with Q-value and prior
- **Stats bar** — iteration counter, self-play / arena progress, replay buffer size, curriculum ladder

Click **▶ Start Training** to begin the self-play loop. The best checkpoint is auto-loaded on startup; training resumes from the correct iteration. Click **■ Stop** to stop cleanly at the next game boundary.

### Play vs AI

```bash
python play.py                    # play as Black (first move)
python play.py --color white      # play as White
python play.py --sims 400         # stronger AI
python play.py --board-size 9
```

Loads the best checkpoint automatically. The board and MCTS viewer are shared with the training GUI. When the pie rule is enabled and you play as White, a **Swap sides** button appears after Black's first move.

### Headless mode

```bash
python main.py --headless
python main.py --headless --board-size 11 --workers 8 --simulations 400
python main.py --headless --checkpoint checkpoints/best.pt
```

### CLI options

| Flag | Default | Description |
|---|---|---|
| `--headless` | off | Run without GUI |
| `--board-size N` | 7 | Initial board size |
| `--checkpoint PATH` | auto | Resume from checkpoint |
| `--simulations N` | 50 | MCTS simulations per move |
| `--workers N` | cpu\_count/2 | Self-play worker threads |

---

## Configuration

All hyperparameters live in `config.py` as a single `HexZeroConfig` dataclass — no config files, no CLI flags for obscure settings. Edit it directly.

| Parameter | Default | Effect |
|---|---|---|
| `num_res_blocks` | 8 | Tower depth |
| `num_filters` | 128 | Network width |
| `mcts_simulations` | 50 | Quality vs speed; 50 is fast, 400+ is strong |
| `games_per_iteration` | 100 | Self-play games before each training round |
| `arena_win_threshold` | 0.55 | Candidate win rate needed to replace champion |
| `curriculum_threshold` | 0.60 | Arena win rate to unlock the next board size |
| `use_pie_rule` | True | Enable swap rule (disable for very early training) |
| `num_self_play_workers` | cpu\_count/2 | Worker threads sharing the InferenceServer |

---

## Game Rules (Hex)

Two players — BLACK and WHITE — alternate placing stones on a rhombus-shaped board of hexagonal cells.

- **BLACK** wins by forming a connected chain from the **top edge** to the **bottom edge**.
- **WHITE** wins by connecting the **left edge** to the **right edge**.
- Every cell is eventually filled; draws are mathematically impossible (Hex is a [strategy-stealing](https://en.wikipedia.org/wiki/Strategy-stealing_argument) game).
- **Pie rule**: after BLACK's first move, WHITE may swap colours to neutralise the first-move advantage.

The first player has a theoretical winning advantage on all sizes, but the winning strategy is unknown for boards larger than a few cells, making self-play meaningful.

---

## References

- Silver et al., *Mastering the Game of Go without Human Knowledge* (AlphaZero), 2017
- Silver et al., *A General Reinforcement Learning Algorithm that Masters Chess, Shogi and Go*, 2018
- Browne et al., *A Survey of Monte Carlo Tree Search Methods*, 2012
- Anshelevich, *The Game of Hex: An Automatic Theorem Proving Approach to Game Programming*, 2000
