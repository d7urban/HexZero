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

    # Input feature planes (2 color + 6 feature = 8 total)
    # Planes: black_stones, white_stones,
    #         black_2bridge, white_2bridge,
    #         black_edge_dist, white_edge_dist,
    #         black_components, white_components
    num_input_planes: int = 8

    # MCTS
    mcts_simulations: int = 50
    cpuct: float = 1.25
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature_moves: int = 20      # use temp=1 for first N moves, then 0
    temperature: float = 1.0

    # Self-play — default to half the logical CPUs so the demo and training
    # threads always have cores to run on; minimum 1, maximum 8.
    num_self_play_workers: int = field(
        default_factory=lambda: max(1, min(8, (os.cpu_count() or 2) // 2))
    )
    games_per_iteration: int = 100

    # Training
    batch_size: int = 256
    learning_rate: float = 1e-3
    # Cosine LR schedule: decays from learning_rate to lr_min over lr_cosine_steps
    # gradient steps, then holds at lr_min until the next curriculum advance, at
    # which point it resets to learning_rate so the network can adapt quickly to
    # the new board size.  Default = min_iters_per_size * train_steps_per_iteration
    # so the LR reaches its floor right around when promotion first becomes possible.
    lr_cosine_steps: int = 1_000   # = 5 iters × 200 steps
    lr_min: float = 1e-4
    weight_decay: float = 1e-4
    value_loss_weight: float = 1.0
    train_steps_per_iteration: int = 200

    # Replay buffer
    replay_buffer_capacity: int = 100_000

    # Arena
    arena_games: int = 40
    arena_win_threshold: float = 0.55

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    keep_last_n_checkpoints: int = 5

    # Curriculum: advance to the next board size when BOTH conditions hold:
    #   1. At least min_iters_per_size iterations completed on the current size.
    #   2. Policy loss has plateaued: relative improvement over the last
    #      min_iters_per_size iterations is below loss_plateau_threshold.
    # This avoids the binary 0%/100% arena win-rate which is not a meaningful
    # mastery signal when MCTS simulations are low.
    loss_plateau_threshold: float = 0.03   # < 3% relative improvement = plateau
    min_iters_per_size: int = 5

    # Pie rule (swap rule): after BLACK's first move WHITE may swap colours.
    # Disable for very early training runs before the net has learned to play.
    use_pie_rule: bool = True
