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
