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
    lr_decay_steps: int = 100_000
    lr_decay_gamma: float = 0.1
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

    # Curriculum: when arena win rate on current size exceeds this, unlock next size
    curriculum_threshold: float = 0.60

    # Pie rule (swap rule): after BLACK's first move WHITE may swap colours.
    # Disable for very early training runs before the net has learned to play.
    use_pie_rule: bool = True
