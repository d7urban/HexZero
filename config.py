import os
from dataclasses import dataclass, field


@dataclass
class HexZeroConfig:
    # Board
    board_sizes: list[int] = field(default_factory=lambda: [7, 9, 11])
    initial_board_size: int = 7

    # Network
    num_res_blocks: int = 8
    num_filters: int = 128
    value_fc_hidden: int = 256

    # Input feature planes (2 color + 6 feature + 1 to-move = 9 total)
    # Planes: black_stones, white_stones,
    #         black_2bridge, white_2bridge,
    #         black_edge_dist, white_edge_dist,
    #         black_components, white_components,
    #         to_move (1=BLACK to move, 0=WHITE to move)
    num_input_planes: int = 9

    # MCTS
    # Simulations scale with board area to keep visits-per-legal-move roughly
    # constant.  mcts_simulations_per_size is indexed in parallel with board_sizes;
    # if a size has no entry the last value is used as a fallback.
    mcts_simulations: int = 50   # used directly when pie_rule is off / for arena
    # Scaled as 1.5 × side², rounded to a nice number:  [7→75, 9→120, 11→180]
    mcts_simulations_per_size: list[int] = field(default_factory=lambda: [75, 120, 180])
    cpuct: float = 1.25
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature_moves: int = 20      # use temp=1 for first N moves, then 0
    temperature: float = 1.0

    # Self-play — workers spend most of their time blocked on GPU inference
    # (releasing the GIL), so more workers = larger inference batches without
    # proportionally more CPU pressure.  Cap at 32; raise if GPU is still idle.
    num_self_play_workers: int = field(
        default_factory=lambda: max(1, min(32, (os.cpu_count() or 2)))
    )
    games_per_iteration: int = 100

    # Inference server — larger batches and slightly longer accumulation window
    # keep the GPU busy across many concurrent MCTS workers.
    inference_max_batch: int = 256
    inference_max_wait_ms: float = 2.0

    # Training
    batch_size: int = 256
    learning_rate: float = 5e-4
    # Cosine LR schedule: decays from learning_rate to lr_min over lr_cosine_steps
    # gradient steps, then holds at lr_min until the next curriculum advance, at
    # which point it resets to learning_rate so the network can adapt quickly to
    # the new board size.  Default = min_iters_per_size * train_steps_per_iteration
    # so the LR reaches its floor right around when promotion first becomes possible.
    lr_cosine_steps: int = 3_000   # = 15 iters × 200 steps
    lr_min: float = 5e-5
    weight_decay: float = 1e-4
    value_loss_weight: float = 1.0
    train_steps_per_iteration: int = 200

    # Replay buffer
    replay_buffer_capacity: int = 100_000

    # Arena
    arena_games: int = 40
    arena_win_threshold: float = 0.55
    # Stochastic first N half-moves per arena game so each game has a different
    # opening; after that both agents play greedy.  Eliminates the degenerate
    # "all-or-nothing" outcome caused by fully deterministic play.
    arena_temperature_moves: int = 4

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    keep_last_n_checkpoints: int = 5

    # Curriculum: advance to the next board size when BOTH conditions hold:
    #   1. At least min_iters_per_size iterations completed on the current size.
    #   2. No arena promotion in the last min_iters_per_size iterations.
    # Arena promotion is the direct model-selection criterion, so its absence
    # is the cleanest saturation signal: if the candidate can no longer beat
    # the champion, the model has learned all it can at this board size.
    min_iters_per_size: int = 10
    # Hard cap: advance curriculum even without a loss plateau if we've spent
    # this many iterations on the current size with no further improvement.
    max_iters_per_size: int = 35

    # Pie rule (swap rule): after BLACK's first move WHITE may swap colours.
    # Disable for very early training runs before the net has learned to play.
    use_pie_rule: bool = True

    def sims_for_size(self, board_size: int) -> int:
        """Return the simulation count to use for a given board size."""
        try:
            idx = self.board_sizes.index(board_size)
            return self.mcts_simulations_per_size[idx]
        except (ValueError, IndexError):
            return self.mcts_simulations_per_size[-1] if self.mcts_simulations_per_size \
                else self.mcts_simulations
