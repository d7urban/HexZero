# HexZero

An AlphaZero-style self-play training suite for the board game [Hex](https://en.wikipedia.org/wiki/Hex_(board_game)), built as a clean, readable showcase of the core ideas behind AlphaZero/AlphaStar.

Hex is an ideal vehicle for this: the rules are simple (connect your two sides before your opponent connects theirs), draws are impossible, and the game has enough depth that a random player loses badly to even shallow search — so you can actually watch the agent improve.

![HexZero GUI](docs/screenshot.png)

---

## Why Hex?

| Property | Hex | Go | Chess |
|---|---|---|---|
| Rules complexity | Minimal | Low | High |
| Draw possible | No | Ko rules | Yes |
| Perfect information | Yes | Yes | Yes |
| Scalable difficulty | 5×5 → 19×19 | Fixed 19×19 | Fixed 8×8 |
| Negamax solution | Only ≤5×5 | No | No |
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
  ├─▶ Policy head: Conv2d(1×1, 1) → reshape (B, H×W) → log-softmax
  │
  └─▶ Value head:  GlobalAvgPool → concat(size_scalar) → FC(256) → ReLU → FC(1) → tanh
```

**Size-agnostic trick**: the value head uses global average pooling before the FC layers, so the spatial dimensions never appear in any weight. A normalised board-size scalar is appended to the pooled vector so the network can calibrate value estimates across sizes.

### Input Feature Planes

| Plane | Description |
|---|---|
| `black_stones` | 1 where BLACK stone placed |
| `white_stones` | 1 where WHITE stone placed |
| `black_2bridge` | Empty cells that are carriers of a BLACK two-bridge |
| `white_2bridge` | Empty cells that are carriers of a WHITE two-bridge |
| `black_edge_dist` | BFS distance to BLACK's nearest winning edge (normalised) |
| `white_edge_dist` | BFS distance to WHITE's nearest winning edge (normalised) |
| `black_components` | Connected component label for BLACK stones (normalised) |
| `white_components` | Connected component label for WHITE stones (normalised) |

Two-bridges are the fundamental tactical unit of Hex — a pair of stones with two shared empty neighbours that form a virtually unbreakable connection. Including them as explicit planes lets the network learn connection-based reasoning far faster than from raw board state alone.

### MCTS

Standard PUCT (AlphaZero variant):

```
PUCT(s, a) = Q(s, a) + c_puct · P(s, a) · √N(s) / (1 + N(s, a))
```

- Dirichlet noise at the root during self-play (α=0.3, ε=0.25)
- Temperature-based move sampling for the first 20 moves, greedy thereafter
- Subtree reuse across moves within a game

### Training Loop

```
repeat:
  1. Self-play      — N workers play games with current best checkpoint
  2. Augment        — each game doubled via 180° rotation
  3. Buffer         — samples added to circular replay buffer (100k cap)
  4. Train          — 200 gradient steps, dynamic batching by board size
  5. Arena          — candidate vs champion (40 games, alternating colours)
  6. Promote        — replace champion if candidate wins ≥ 55% of games
```

### Curriculum

Training starts on 7×7. When the agent achieves ≥ 60% win rate in the arena against the prior champion on the current size, the next size (9×9, then 11×11) is unlocked. Knowledge transfers because the convolutional weights are size-agnostic.

---

## Project Structure

```
HexZero/
├── main.py               — entry point (GUI + headless modes)
├── config.py             — HexZeroConfig dataclass (all hyperparameters)
├── requirements.txt
└── hexzero/
    ├── game.py           — Hex engine: union-find win detection, cloning
    ├── features.py       — 8-plane feature tensor construction
    ├── net.py            — HexNet: ResNet stem + tower + policy/value heads
    ├── mcts.py           — PUCT MCTS: Node, MCTSAgent, tree reuse
    ├── replay_buffer.py  — thread-safe circular buffer, dynamic batching
    ├── self_play.py      — multiprocess self-play workers
    ├── trainer.py        — gradient updates, loss computation, metrics
    ├── arena.py          — champion vs candidate evaluation
    ├── checkpoint.py     — atomic saves, rolling window, best.pt
    └── gui/
        ├── signals.py        — Qt signal carriers (cross-thread comms)
        ├── board_widget.py   — pointy-top hex board, policy heatmap
        ├── chart_widget.py   — live loss / accuracy curves (pyqtgraph)
        ├── mcts_widget.py    — top-moves table with visit count shading
        └── main_window.py    — main window, toolbar, training QThread
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

Requires Python ≥ 3.11.

---

## Usage

### GUI mode (default)

```bash
python main.py
python main.py --board-size 9
python main.py --checkpoint checkpoints/best.pt
```

The GUI shows:
- **Board** — live hex board with policy heatmap overlay and last-move highlight
- **Charts** — policy loss, value loss, and policy accuracy in real time
- **MCTS viewer** — top 5 moves by visit count, with Q-value and prior

Click **▶ Start Training** to begin the self-play loop. Training runs in a background thread; the GUI stays responsive.

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
| `--simulations N` | 200 | MCTS simulations per move |
| `--workers N` | 4 | Self-play worker processes |

---

## Configuration

All hyperparameters live in `config.py` as a single `HexZeroConfig` dataclass — no config files, no CLI flags for obscure settings. Edit it directly.

Key values and their effect:

| Parameter | Default | Effect |
|---|---|---|
| `num_res_blocks` | 8 | Tower depth; 6 is fine for 7×7, 10+ for 11×11 |
| `num_filters` | 128 | Network width; bottleneck is self-play throughput, not model size |
| `mcts_simulations` | 200 | Quality vs speed trade-off; 50 is playable, 800 is strong |
| `games_per_iteration` | 100 | Self-play games before each training round |
| `arena_win_threshold` | 0.55 | How much better the candidate must be to replace the champion |
| `curriculum_threshold` | 0.60 | Win rate required to unlock the next board size |

---

## Game Rules (Hex)

Two players — BLACK and WHITE — alternate placing stones on a rhombus-shaped board of hexagonal cells.

- **BLACK** wins by forming a connected chain from the **top edge** to the **bottom edge**.
- **WHITE** wins by connecting the **left edge** to the **right edge**.
- Every cell is eventually filled; draws are mathematically impossible (Hex is a [strategy-stealing](https://en.wikipedia.org/wiki/Strategy-stealing_argument) game).

The first player has a theoretical winning advantage on all sizes, but the winning strategy is unknown for boards larger than a few cells, making self-play meaningful.

---

## References

- Silver et al., *Mastering the Game of Go without Human Knowledge* (AlphaZero), 2017
- Silver et al., *A General Reinforcement Learning Algorithm that Masters Chess, Shogi and Go*, 2018
- Browne et al., *A Survey of Monte Carlo Tree Search Methods*, 2012
- Anshelevich, *The Game of Hex: An Automatic Theorem Proving Approach to Game Programming*, 2000
