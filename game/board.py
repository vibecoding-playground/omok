#%%
"""
Omok (Gomoku) 9x9 Board Implementation
AlphaZero-style state representation for neural network input.
"""
import numpy as np
from enum import IntEnum
from typing import Optional, Tuple, List
from copy import deepcopy


class Player(IntEnum):
    """Stone colors. BLACK plays first."""
    BLACK = 1
    WHITE = -1
    EMPTY = 0


class Board:
    """
    9x9 Omok board with AlphaZero-compatible state representation.

    State encoding: (2, 9, 9) tensor
      - Channel 0: Current player's stones (1 where stone exists)
      - Channel 1: Opponent's stones (1 where stone exists)
    """
    SIZE = 9
    WIN_LENGTH = 5

    def __init__(self):
        self.reset()

    def reset(self) -> np.ndarray:
        """Reset board to initial empty state. Returns initial observation."""
        # Internal board: 0=empty, 1=black, -1=white
        self._board = np.zeros((self.SIZE, self.SIZE), dtype=np.int8)
        self._current_player = Player.BLACK
        self._last_move: Optional[Tuple[int, int]] = None
        self._move_count = 0
        self._winner: Optional[Player] = None
        self._game_over = False
        return self.get_state()

    def get_state(self) -> np.ndarray:
        """
        Get board state as (2, 9, 9) float32 tensor for neural network.
        Perspective is always from current player's view.
        """
        state = np.zeros((2, self.SIZE, self.SIZE), dtype=np.float32)

        # Channel 0: Current player's stones
        state[0] = (self._board == self._current_player).astype(np.float32)
        # Channel 1: Opponent's stones
        state[1] = (self._board == -self._current_player).astype(np.float32)

        return state

    def get_valid_moves(self) -> np.ndarray:
        """Return flat (81,) boolean mask of valid moves."""
        return (self._board == Player.EMPTY).flatten()

    def get_valid_moves_2d(self) -> np.ndarray:
        """Return (9, 9) boolean mask of valid moves."""
        return self._board == Player.EMPTY

    @property
    def current_player(self) -> Player:
        return self._current_player

    @property
    def last_move(self) -> Optional[Tuple[int, int]]:
        return self._last_move

    @property
    def is_game_over(self) -> bool:
        return self._game_over

    @property
    def winner(self) -> Optional[Player]:
        return self._winner

    @property
    def move_count(self) -> int:
        return self._move_count

    def play(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        Play a move. Action is flattened index [0, 80].

        Returns:
            state: New board state (2, 9, 9)
            reward: 1.0 if current player wins, -1.0 if loses, 0.0 otherwise
            done: Whether game is over
        """
        if self._game_over:
            raise ValueError("Game is already over")

        row, col = divmod(action, self.SIZE)

        if not self._is_valid_position(row, col):
            raise ValueError(f"Invalid position: ({row}, {col})")

        if self._board[row, col] != Player.EMPTY:
            raise ValueError(f"Position ({row}, {col}) is already occupied")

        # Place stone
        self._board[row, col] = self._current_player
        self._last_move = (row, col)
        self._move_count += 1

        # Check for win
        if self._check_win(row, col):
            self._winner = self._current_player
            self._game_over = True
            return self.get_state(), 1.0, True

        # Check for draw (board full)
        if self._move_count >= self.SIZE * self.SIZE:
            self._game_over = True
            return self.get_state(), 0.0, True

        # Switch player
        self._current_player = Player(-self._current_player)

        return self.get_state(), 0.0, False

    def play_xy(self, row: int, col: int) -> Tuple[np.ndarray, float, bool]:
        """Play move using (row, col) coordinates."""
        return self.play(row * self.SIZE + col)

    def _is_valid_position(self, row: int, col: int) -> bool:
        return 0 <= row < self.SIZE and 0 <= col < self.SIZE

    def _check_win(self, row: int, col: int) -> bool:
        """Check if the last move at (row, col) creates a winning line."""
        player = self._board[row, col]

        # 4 directions: horizontal, vertical, diagonal, anti-diagonal
        directions = [
            (0, 1),   # horizontal
            (1, 0),   # vertical
            (1, 1),   # diagonal
            (1, -1),  # anti-diagonal
        ]

        for dr, dc in directions:
            count = 1  # Count the placed stone

            # Count in positive direction
            r, c = row + dr, col + dc
            while self._is_valid_position(r, c) and self._board[r, c] == player:
                count += 1
                r, c = r + dr, c + dc

            # Count in negative direction
            r, c = row - dr, col - dc
            while self._is_valid_position(r, c) and self._board[r, c] == player:
                count += 1
                r, c = r - dr, c - dc

            if count >= self.WIN_LENGTH:
                return True

        return False

    def copy(self) -> "Board":
        """Create a deep copy of the board."""
        new_board = Board.__new__(Board)
        new_board._board = self._board.copy()
        new_board._current_player = self._current_player
        new_board._last_move = self._last_move
        new_board._move_count = self._move_count
        new_board._winner = self._winner
        new_board._game_over = self._game_over
        return new_board

    def __repr__(self) -> str:
        """ASCII representation of the board."""
        symbols = {Player.EMPTY: ".", Player.BLACK: "●", Player.WHITE: "○"}
        lines = []

        # Column headers
        lines.append("  " + " ".join(str(i) for i in range(self.SIZE)))

        for row in range(self.SIZE):
            row_str = f"{row} "
            for col in range(self.SIZE):
                stone = self._board[row, col]
                symbol = symbols[Player(stone)]
                # Mark last move
                if self._last_move == (row, col):
                    symbol = "◉" if stone == Player.BLACK else "◎"
                row_str += symbol + " "
            lines.append(row_str)

        status = f"Turn: {'Black ●' if self._current_player == Player.BLACK else 'White ○'}"
        if self._game_over:
            if self._winner:
                status = f"Winner: {'Black ●' if self._winner == Player.BLACK else 'White ○'}"
            else:
                status = "Draw!"
        lines.append(status)

        return "\n".join(lines)


#%%
if __name__ == "__main__":
    # Quick test
    board = Board()
    print(board)
    print(f"\nState shape: {board.get_state().shape}")
    print(f"Valid moves: {board.get_valid_moves().sum()}")

    # Play a few moves
    board.play_xy(4, 4)  # Black center
    board.play_xy(4, 5)  # White
    board.play_xy(3, 3)  # Black
    print("\n" + str(board))
