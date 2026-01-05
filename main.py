#%%
"""
Omok (Gomoku) AI - Main Entry Point

Usage:
    python main.py play [--model PATH]   # Play against AI
    python main.py train [--iterations N]  # Train the model
"""
import argparse
import curses
import os
import sys
from pathlib import Path

import torch
import numpy as np

from game.board import Board, Player
from model.network import PolicyValueNet, get_device
from mcts.search import MCTS, MCTSConfig
from training.trainer import Trainer, TrainingConfig


class OmokCLI:
    """
    Curses-based CLI for playing Omok against AI.

    Controls:
        Arrow keys: Move cursor
        Enter/Space: Place stone
        q: Quit
    """

    # Board display characters
    EMPTY = "·"
    BLACK = "●"
    WHITE = "○"
    CURSOR = "□"

    # Colors (curses color pair indices)
    COLOR_NORMAL = 1
    COLOR_BLACK = 2
    COLOR_WHITE = 3
    COLOR_CURSOR = 4
    COLOR_LAST_MOVE = 5
    COLOR_STATUS = 6

    def __init__(self, model: PolicyValueNet, device: torch.device, mcts_sims: int = 400):
        self.model = model
        self.device = device
        self.mcts_config = MCTSConfig(num_simulations=mcts_sims, temperature=0.1)
        self.board = Board()
        self.cursor_row = 4
        self.cursor_col = 4
        self.player_color = Player.BLACK  # Human plays black
        self.message = ""

    def init_colors(self):
        """Initialize curses color pairs."""
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(self.COLOR_NORMAL, curses.COLOR_WHITE, -1)
        curses.init_pair(self.COLOR_BLACK, curses.COLOR_WHITE, -1)
        curses.init_pair(self.COLOR_WHITE, curses.COLOR_CYAN, -1)
        curses.init_pair(self.COLOR_CURSOR, curses.COLOR_YELLOW, -1)
        curses.init_pair(self.COLOR_LAST_MOVE, curses.COLOR_GREEN, -1)
        curses.init_pair(self.COLOR_STATUS, curses.COLOR_MAGENTA, -1)

    def draw_board(self, stdscr):
        """Draw the game board."""
        stdscr.clear()

        # Title
        title = "=== OMOK (9x9) ==="
        stdscr.addstr(0, 0, title, curses.A_BOLD)

        # Column headers
        col_header = "  " + " ".join(str(i) for i in range(Board.SIZE))
        stdscr.addstr(2, 0, col_header)

        # Board rows
        for row in range(Board.SIZE):
            row_str = f"{row} "
            stdscr.addstr(3 + row, 0, row_str)

            for col in range(Board.SIZE):
                x = 2 + col * 2
                y = 3 + row

                stone = self.board._board[row, col]
                is_cursor = (row == self.cursor_row and col == self.cursor_col)
                is_last_move = self.board.last_move == (row, col)

                if is_cursor and stone == Player.EMPTY:
                    char = self.CURSOR
                    color = curses.color_pair(self.COLOR_CURSOR) | curses.A_BOLD
                elif stone == Player.BLACK:
                    char = self.BLACK
                    if is_last_move:
                        color = curses.color_pair(self.COLOR_LAST_MOVE) | curses.A_BOLD
                    else:
                        color = curses.color_pair(self.COLOR_BLACK) | curses.A_BOLD
                elif stone == Player.WHITE:
                    char = self.WHITE
                    if is_last_move:
                        color = curses.color_pair(self.COLOR_LAST_MOVE) | curses.A_BOLD
                    else:
                        color = curses.color_pair(self.COLOR_WHITE)
                else:
                    char = self.EMPTY
                    color = curses.color_pair(self.COLOR_NORMAL)

                stdscr.addstr(y, x, char, color)

        # Status line
        status_y = 3 + Board.SIZE + 1
        if self.board.is_game_over:
            if self.board.winner == self.player_color:
                status = "You WIN!"
            elif self.board.winner is None:
                status = "DRAW!"
            else:
                status = "AI WINS!"
            stdscr.addstr(status_y, 0, status, curses.color_pair(self.COLOR_STATUS) | curses.A_BOLD)
        else:
            turn = "Your turn (●)" if self.board.current_player == self.player_color else "AI thinking..."
            stdscr.addstr(status_y, 0, turn)

        # Message
        if self.message:
            stdscr.addstr(status_y + 1, 0, self.message)

        # Controls help
        help_y = status_y + 3
        stdscr.addstr(help_y, 0, "Controls: ↑↓←→ Move | Enter Place | q Quit")

        stdscr.refresh()

    def handle_input(self, key) -> bool:
        """
        Handle keyboard input.
        Returns False if should quit.
        """
        self.message = ""

        if key == ord("q") or key == ord("Q"):
            return False

        if self.board.is_game_over:
            return True

        # Only allow input during player's turn
        if self.board.current_player != self.player_color:
            return True

        # Arrow keys
        if key == curses.KEY_UP:
            self.cursor_row = max(0, self.cursor_row - 1)
        elif key == curses.KEY_DOWN:
            self.cursor_row = min(Board.SIZE - 1, self.cursor_row + 1)
        elif key == curses.KEY_LEFT:
            self.cursor_col = max(0, self.cursor_col - 1)
        elif key == curses.KEY_RIGHT:
            self.cursor_col = min(Board.SIZE - 1, self.cursor_col + 1)
        elif key in (curses.KEY_ENTER, 10, 13, ord(" ")):
            # Try to place stone
            if self.board._board[self.cursor_row, self.cursor_col] == Player.EMPTY:
                self.board.play_xy(self.cursor_row, self.cursor_col)
            else:
                self.message = "Position occupied!"

        return True

    def ai_move(self, stdscr):
        """AI makes a move using MCTS."""
        self.model.eval()
        mcts = MCTS(self.model, self.mcts_config, self.device)

        # Run MCTS search
        probs = mcts.search(self.board, add_noise=False)
        action = mcts.select_action(probs, deterministic=True)

        # Play the move
        self.board.play(action)

    def run(self, stdscr):
        """Main game loop."""
        curses.curs_set(0)  # Hide cursor
        stdscr.keypad(True)
        self.init_colors()

        running = True
        while running:
            self.draw_board(stdscr)

            if self.board.is_game_over:
                # Wait for quit
                key = stdscr.getch()
                if key == ord("q") or key == ord("Q"):
                    break
                continue

            if self.board.current_player == self.player_color:
                # Player's turn
                key = stdscr.getch()
                running = self.handle_input(key)
            else:
                # AI's turn
                stdscr.nodelay(False)
                self.ai_move(stdscr)


def play_game(model_path: str = None, mcts_sims: int = 400):
    """Start an interactive game against AI."""
    device = get_device()
    print(f"Using device: {device}")

    model = PolicyValueNet().to(device)

    if model_path and os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded model from: {model_path}")
    else:
        print("Using untrained model (random play)")

    cli = OmokCLI(model, device, mcts_sims)

    # Run with curses wrapper
    curses.wrapper(cli.run)


def train_model(iterations: int = 50, checkpoint: str = None):
    """Train the Omok AI."""
    device = get_device()
    print(f"Using device: {device}")

    config = TrainingConfig(
        num_games_per_iteration=50,
        mcts_simulations=200,
        epochs_per_iteration=5,
        checkpoint_dir="checkpoints",
    )

    trainer = Trainer(config, device)

    if checkpoint and os.path.exists(checkpoint):
        trainer.load_checkpoint(checkpoint)

    print(f"\nStarting training for {iterations} iterations...")
    trainer.train(iterations, verbose=True)

    # Save final model
    trainer.save_checkpoint()
    print("\nTraining complete!")


def main():
    parser = argparse.ArgumentParser(description="Omok AI - AlphaZero style")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Play command
    play_parser = subparsers.add_parser("play", help="Play against AI")
    play_parser.add_argument(
        "--model", "-m",
        type=str,
        default="checkpoints/model_latest.pt",
        help="Path to model checkpoint"
    )
    play_parser.add_argument(
        "--sims", "-s",
        type=int,
        default=400,
        help="MCTS simulations per move"
    )

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument(
        "--iterations", "-i",
        type=int,
        default=50,
        help="Number of training iterations"
    )
    train_parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        default=None,
        help="Resume from checkpoint"
    )

    args = parser.parse_args()

    if args.command == "play":
        play_game(args.model, args.sims)
    elif args.command == "train":
        train_model(args.iterations, args.checkpoint)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
