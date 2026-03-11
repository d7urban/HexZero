"""
Qt signal carriers for cross-thread communication.
Backend threads hold a reference and emit; widgets connect and receive.
"""

from PyQt6.QtCore import QObject, pyqtSignal


class TrainingSignals(QObject):
    # Emitted after each training step with a metrics dict
    metrics_updated = pyqtSignal(dict)

    # Emitted after each MCTS move during a live game display
    # args: board numpy array (H,W), policy array (H*W,), mcts info dict
    game_step = pyqtSignal(object, object, dict)

    # Emitted when a self-play game completes: (winner: int, move_count: int)
    game_finished = pyqtSignal(int, int)

    # Emitted after arena: (cand_wins, champ_wins, draws)
    arena_result = pyqtSignal(int, int, int)

    # Emitted after each arena game: (games_done, cand_wins, total_games)
    arena_progress = pyqtSignal(int, int, int)

    # Emitted after a checkpoint is saved: path string
    checkpoint_saved = pyqtSignal(str)

    # Emitted to update the status bar
    status_message = pyqtSignal(str)

    # Emitted when iteration starts/ends
    iteration_started = pyqtSignal(int)   # iteration number
    iteration_finished = pyqtSignal(int)

    # Self-play progress: (games_done, games_total)
    self_play_progress = pyqtSignal(int, int)

    # Buffer fill: total samples currently stored
    buffer_updated = pyqtSignal(int)

    # Curriculum: emitted when training advances to a larger board size
    board_size_advanced = pyqtSignal(int)

    # Curriculum iteration progress: (iters_on_size, min_iters_per_size)
    curriculum_progress = pyqtSignal(int, int)

    # Pie rule swap rate after self-play: (swap_games, total_games)
    swap_rate_updated = pyqtSignal(int, int)

    # Loss-plateau progress: (improvement_pct, threshold_pct, has_data)
    # improvement_pct = relative loss drop over last window (%), lower = more plateaued
    # has_data = False until enough iterations have accumulated
    plateau_updated = pyqtSignal(float, float, bool)
