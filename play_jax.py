#!/usr/bin/env python3
"""
Omok AI - JAX Model Local Play

JAX/Flax로 학습된 모델과 대전하는 CLI 인터페이스입니다.

Usage:
    python play_jax.py --model checkpoint_jax.pkl
    python play_jax.py --model omok_model_jax.pkl
"""
import argparse
import curses
import math
import pickle
import os
from enum import IntEnum
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# JAX imports
import jax
import jax.numpy as jnp
from jax import random, jit
import flax.linen as nn


# ============================================================================
# Game Logic
# ============================================================================

class Player(IntEnum):
    BLACK = 1
    WHITE = -1
    EMPTY = 0


class Board:
    SIZE = 9
    WIN_LENGTH = 5

    def __init__(self):
        self.reset()

    def reset(self) -> np.ndarray:
        self._board = np.zeros((self.SIZE, self.SIZE), dtype=np.int8)
        self._current_player = Player.BLACK
        self._last_move: Optional[Tuple[int, int]] = None
        self._move_count = 0
        self._winner: Optional[Player] = None
        self._game_over = False
        return self.get_state()

    def get_state(self) -> np.ndarray:
        state = np.zeros((2, self.SIZE, self.SIZE), dtype=np.float32)
        state[0] = (self._board == self._current_player).astype(np.float32)
        state[1] = (self._board == -self._current_player).astype(np.float32)
        return state

    def get_valid_moves(self) -> np.ndarray:
        return (self._board == Player.EMPTY).flatten()

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
        if self._game_over:
            raise ValueError("Game is already over")

        row, col = divmod(action, self.SIZE)
        if not (0 <= row < self.SIZE and 0 <= col < self.SIZE):
            raise ValueError(f"Invalid position")
        if self._board[row, col] != Player.EMPTY:
            raise ValueError(f"Position occupied")

        self._board[row, col] = self._current_player
        self._last_move = (row, col)
        self._move_count += 1

        if self._check_win(row, col):
            self._winner = self._current_player
            self._game_over = True
            return self.get_state(), 1.0, True

        if self._move_count >= self.SIZE * self.SIZE:
            self._game_over = True
            return self.get_state(), 0.0, True

        self._current_player = Player(-self._current_player)
        return self.get_state(), 0.0, False

    def play_xy(self, row: int, col: int) -> Tuple[np.ndarray, float, bool]:
        return self.play(row * self.SIZE + col)

    def _check_win(self, row: int, col: int) -> bool:
        player = self._board[row, col]
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            for sign in [1, -1]:
                r, c = row + sign * dr, col + sign * dc
                while 0 <= r < self.SIZE and 0 <= c < self.SIZE and self._board[r, c] == player:
                    count += 1
                    r, c = r + sign * dr, c + sign * dc
            if count >= self.WIN_LENGTH:
                return True
        return False

    def copy(self) -> "Board":
        new_board = Board.__new__(Board)
        new_board._board = self._board.copy()
        new_board._current_player = self._current_player
        new_board._last_move = self._last_move
        new_board._move_count = self._move_count
        new_board._winner = self._winner
        new_board._game_over = self._game_over
        return new_board


# ============================================================================
# Neural Network (Flax)
# ============================================================================

class ResBlock(nn.Module):
    channels: int

    @nn.compact
    def __call__(self, x, train: bool = True):
        residual = x
        x = nn.Conv(self.channels, (3, 3), padding='SAME', use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        x = nn.Conv(self.channels, (3, 3), padding='SAME', use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        return nn.relu(x + residual)


class PolicyValueNet(nn.Module):
    num_filters: int = 128
    num_res_blocks: int = 4
    board_size: int = 9

    @nn.compact
    def __call__(self, x, train: bool = True):
        action_size = self.board_size * self.board_size

        x = nn.Conv(self.num_filters, (3, 3), padding='SAME', use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)

        for _ in range(self.num_res_blocks):
            x = ResBlock(self.num_filters)(x, train=train)

        policy = nn.Conv(2, (1, 1), use_bias=False)(x)
        policy = nn.BatchNorm(use_running_average=not train)(policy)
        policy = nn.relu(policy)
        policy = policy.reshape((policy.shape[0], -1))
        policy = nn.Dense(action_size)(policy)
        policy = nn.log_softmax(policy)

        value = nn.Conv(1, (1, 1), use_bias=False)(x)
        value = nn.BatchNorm(use_running_average=not train)(value)
        value = nn.relu(value)
        value = value.reshape((value.shape[0], -1))
        value = nn.Dense(64)(value)
        value = nn.relu(value)
        value = nn.Dense(1)(value)
        value = nn.tanh(value)

        return policy, value


# ============================================================================
# MCTS
# ============================================================================

@dataclass
class MCTSConfig:
    num_simulations: int = 400
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature: float = 0.1


class Node:
    def __init__(self, prior: float):
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: Dict[int, "Node"] = {}
        self.is_expanded = False

    @property
    def value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0

    def ucb_score(self, parent_visit_count: int, c_puct: float) -> float:
        exploration = c_puct * self.prior * math.sqrt(parent_visit_count) / (1 + self.visit_count)
        return self.value + exploration

    def select_child(self, c_puct: float) -> Tuple[int, "Node"]:
        best_score, best_action, best_child = -float("inf"), -1, None
        for action, child in self.children.items():
            score = child.ucb_score(self.visit_count, c_puct)
            if score > best_score:
                best_score, best_action, best_child = score, action, child
        return best_action, best_child


class MCTS:
    def __init__(self, apply_fn, params, batch_stats, config: MCTSConfig):
        self.apply_fn = apply_fn
        self.params = params
        self.batch_stats = batch_stats
        self.config = config
        self.root: Optional[Node] = None

        @jit
        def predict_fn(params, batch_stats, state):
            state = state.transpose(1, 2, 0)[None, ...]
            variables = {'params': params, 'batch_stats': batch_stats}
            (log_policy, value), _ = apply_fn(
                variables, state, train=False, mutable=['batch_stats']
            )
            return jax.nn.softmax(log_policy[0]), value[0, 0]

        self.predict_fn = predict_fn

    def search(self, board: Board, add_noise: bool = False) -> np.ndarray:
        self.root = Node(prior=0.0)
        self._expand(self.root, board)

        if add_noise:
            self._add_dirichlet_noise(self.root, board.get_valid_moves())

        for _ in range(self.config.num_simulations):
            node = self.root
            scratch_board = board.copy()
            search_path = [node]

            while node.is_expanded and not scratch_board.is_game_over:
                action, node = node.select_child(self.config.c_puct)
                scratch_board.play(action)
                search_path.append(node)

            if scratch_board.is_game_over:
                value = 0.0 if scratch_board.winner is None else -1.0
            else:
                value = self._expand(node, scratch_board)

            self._backup(search_path, value)

        return self._get_action_probs(board.get_valid_moves())

    def _expand(self, node: Node, board: Board) -> float:
        state = jnp.array(board.get_state())
        valid_moves = board.get_valid_moves()

        policy, value = self.predict_fn(self.params, self.batch_stats, state)
        policy = np.array(policy)

        policy = policy * valid_moves
        policy_sum = policy.sum()
        if policy_sum > 0:
            policy = policy / policy_sum
        else:
            policy = valid_moves / valid_moves.sum()

        for action in range(len(policy)):
            if valid_moves[action]:
                node.children[action] = Node(prior=policy[action])
        node.is_expanded = True
        return float(value)

    def _add_dirichlet_noise(self, node: Node, valid_moves: np.ndarray):
        valid_actions = np.where(valid_moves)[0]
        noise = np.random.dirichlet([self.config.dirichlet_alpha] * len(valid_actions))
        for i, action in enumerate(valid_actions):
            if action in node.children:
                node.children[action].prior = (
                    (1 - self.config.dirichlet_epsilon) * node.children[action].prior
                    + self.config.dirichlet_epsilon * noise[i]
                )

    def _backup(self, search_path: List[Node], value: float):
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            value = -value

    def _get_action_probs(self, valid_moves: np.ndarray) -> np.ndarray:
        visit_counts = np.zeros(81, dtype=np.float32)
        for action, child in self.root.children.items():
            visit_counts[action] = child.visit_count

        if self.config.temperature == 0:
            probs = np.zeros_like(visit_counts)
            probs[np.argmax(visit_counts)] = 1.0
        else:
            visit_counts = visit_counts ** (1 / self.config.temperature)
            probs = visit_counts / (visit_counts.sum() + 1e-8)
        return probs

    def select_action(self, probs: np.ndarray, deterministic: bool = True) -> int:
        return int(np.argmax(probs)) if deterministic else int(np.random.choice(len(probs), p=probs))


# ============================================================================
# CLI Interface
# ============================================================================

class OmokCLI:
    EMPTY = "·"
    BLACK = "●"
    WHITE = "○"
    CURSOR = "□"

    COLOR_NORMAL = 1
    COLOR_BLACK = 2
    COLOR_WHITE = 3
    COLOR_CURSOR = 4
    COLOR_LAST_MOVE = 5
    COLOR_STATUS = 6

    def __init__(self, params, batch_stats, mcts_sims: int = 400):
        self.params = params
        self.batch_stats = batch_stats
        self.mcts_config = MCTSConfig(num_simulations=mcts_sims, temperature=0.1)
        self.board = Board()
        self.cursor_row = 4
        self.cursor_col = 4
        self.player_color = Player.BLACK
        self.message = ""
        self.model = PolicyValueNet()

    def init_colors(self):
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(self.COLOR_NORMAL, curses.COLOR_WHITE, -1)
        curses.init_pair(self.COLOR_BLACK, curses.COLOR_WHITE, -1)
        curses.init_pair(self.COLOR_WHITE, curses.COLOR_CYAN, -1)
        curses.init_pair(self.COLOR_CURSOR, curses.COLOR_YELLOW, -1)
        curses.init_pair(self.COLOR_LAST_MOVE, curses.COLOR_GREEN, -1)
        curses.init_pair(self.COLOR_STATUS, curses.COLOR_MAGENTA, -1)

    def draw_board(self, stdscr):
        stdscr.clear()

        title = "=== OMOK (9x9) - JAX Model ==="
        stdscr.addstr(0, 0, title, curses.A_BOLD)

        col_header = "  " + " ".join(str(i) for i in range(Board.SIZE))
        stdscr.addstr(2, 0, col_header)

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

        if self.message:
            stdscr.addstr(status_y + 1, 0, self.message)

        help_y = status_y + 3
        stdscr.addstr(help_y, 0, "Controls: ↑↓←→ Move | Enter Place | q Quit")

        stdscr.refresh()

    def handle_input(self, key) -> bool:
        self.message = ""

        if key == ord("q") or key == ord("Q"):
            return False

        if self.board.is_game_over:
            return True

        if self.board.current_player != self.player_color:
            return True

        if key == curses.KEY_UP:
            self.cursor_row = max(0, self.cursor_row - 1)
        elif key == curses.KEY_DOWN:
            self.cursor_row = min(Board.SIZE - 1, self.cursor_row + 1)
        elif key == curses.KEY_LEFT:
            self.cursor_col = max(0, self.cursor_col - 1)
        elif key == curses.KEY_RIGHT:
            self.cursor_col = min(Board.SIZE - 1, self.cursor_col + 1)
        elif key in (curses.KEY_ENTER, 10, 13, ord(" ")):
            if self.board._board[self.cursor_row, self.cursor_col] == Player.EMPTY:
                self.board.play_xy(self.cursor_row, self.cursor_col)
            else:
                self.message = "Position occupied!"

        return True

    def ai_move(self, stdscr):
        mcts = MCTS(self.model.apply, self.params, self.batch_stats, self.mcts_config)
        probs = mcts.search(self.board, add_noise=False)
        action = mcts.select_action(probs, deterministic=True)
        self.board.play(action)

    def run(self, stdscr):
        curses.curs_set(0)
        stdscr.keypad(True)
        self.init_colors()

        running = True
        while running:
            self.draw_board(stdscr)

            if self.board.is_game_over:
                key = stdscr.getch()
                if key == ord("q") or key == ord("Q"):
                    break
                continue

            if self.board.current_player == self.player_color:
                key = stdscr.getch()
                running = self.handle_input(key)
            else:
                stdscr.nodelay(False)
                self.ai_move(stdscr)


# ============================================================================
# Main
# ============================================================================

def load_jax_model(model_path: str):
    """Load JAX model from checkpoint."""
    print(f"Loading model from: {model_path}")

    with open(model_path, "rb") as f:
        data = pickle.load(f)

    # Handle both checkpoint format and final model format
    if "params" in data:
        params = data["params"]
        batch_stats = data["batch_stats"]
    else:
        raise ValueError("Unknown checkpoint format")

    print("Model loaded successfully!")
    return params, batch_stats


def main():
    parser = argparse.ArgumentParser(description="Omok AI - JAX Model Play")
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Path to JAX model (checkpoint_jax.pkl or omok_model_jax.pkl)"
    )
    parser.add_argument(
        "--sims", "-s",
        type=int,
        default=400,
        help="MCTS simulations per move (default: 400)"
    )

    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return

    params, batch_stats = load_jax_model(args.model)
    cli = OmokCLI(params, batch_stats, args.sims)

    print("Starting game...")
    print("Controls: Arrow keys to move, Enter to place, q to quit")

    curses.wrapper(cli.run)


if __name__ == "__main__":
    main()
