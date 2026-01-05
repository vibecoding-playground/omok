#%% [markdown]
# # Omok (오목) AI - JAX/TPU Training
#
# AlphaZero 스타일 강화학습으로 9x9 오목 AI를 TPU에서 학습합니다.
#
# **사용법:**
# 1. 런타임 > 런타임 유형 변경 > **TPU** 선택
# 2. 모든 셀 실행
# 3. 학습 완료 후 모델 다운로드

#%%
# Setup - Install JAX with TPU support
import subprocess
subprocess.run(["pip", "install", "-q", "jax[tpu]", "-f", "https://storage.googleapis.com/jax-releases/libtpu_releases.html"])
subprocess.run(["pip", "install", "-q", "flax", "optax", "numpy", "tqdm", "matplotlib"])

#%%
import os
import math
import functools
from enum import IntEnum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Sequence

import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import jax
import jax.numpy as jnp
from jax import random, jit, vmap, pmap
import flax.linen as nn
from flax.training import train_state
import optax

# Check devices
print(f"JAX devices: {jax.devices()}")
print(f"Device count: {jax.device_count()}")
print(f"Local device count: {jax.local_device_count()}")

#%% [markdown]
# ## 1. Game Logic

#%%
class Player(IntEnum):
    BLACK = 1
    WHITE = -1
    EMPTY = 0


class Board:
    """9x9 Omok board - pure NumPy for compatibility."""
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
        """Get board state as (2, 9, 9) array."""
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


print(f"Board size: {Board.SIZE}x{Board.SIZE}")

#%% [markdown]
# ## 2. Neural Network (Flax)

#%%
class ResBlock(nn.Module):
    """Residual block with BatchNorm."""
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
    """AlphaZero-style Policy-Value Network in Flax."""
    num_filters: int = 128
    num_res_blocks: int = 4
    board_size: int = 9

    @nn.compact
    def __call__(self, x, train: bool = True):
        action_size = self.board_size * self.board_size

        # Initial conv block
        x = nn.Conv(self.num_filters, (3, 3), padding='SAME', use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)

        # Residual tower
        for _ in range(self.num_res_blocks):
            x = ResBlock(self.num_filters)(x, train=train)

        # Policy head
        policy = nn.Conv(2, (1, 1), use_bias=False)(x)
        policy = nn.BatchNorm(use_running_average=not train)(policy)
        policy = nn.relu(policy)
        policy = policy.reshape((policy.shape[0], -1))  # Flatten
        policy = nn.Dense(action_size)(policy)
        policy = nn.log_softmax(policy)

        # Value head
        value = nn.Conv(1, (1, 1), use_bias=False)(x)
        value = nn.BatchNorm(use_running_average=not train)(value)
        value = nn.relu(value)
        value = value.reshape((value.shape[0], -1))  # Flatten
        value = nn.Dense(64)(value)
        value = nn.relu(value)
        value = nn.Dense(1)(value)
        value = nn.tanh(value)

        return policy, value


# Initialize model
model = PolicyValueNet()
rng = random.PRNGKey(42)

# Create dummy input (batch, height, width, channels) - NHWC format for JAX
dummy_input = jnp.ones((1, 9, 9, 2))
variables = model.init(rng, dummy_input, train=False)

param_count = sum(x.size for x in jax.tree_util.tree_leaves(variables['params']))
print(f"Model parameters: {param_count:,}")

#%% [markdown]
# ## 3. MCTS

#%%
@dataclass
class MCTSConfig:
    num_simulations: int = 200
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature: float = 1.0


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

        # JIT compile the predict function
        @jit
        def predict_fn(params, batch_stats, state):
            # state: (2, 9, 9) -> (1, 9, 9, 2) NHWC
            state = state.transpose(1, 2, 0)[None, ...]
            variables = {'params': params, 'batch_stats': batch_stats}
            (log_policy, value), _ = apply_fn(
                variables, state, train=False, mutable=['batch_stats']
            )
            return jax.nn.softmax(log_policy[0]), value[0, 0]

        self.predict_fn = predict_fn

    def search(self, board: Board, add_noise: bool = True) -> np.ndarray:
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

        # Mask invalid moves
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

    def select_action(self, probs: np.ndarray, deterministic: bool = False) -> int:
        return int(np.argmax(probs)) if deterministic else int(np.random.choice(len(probs), p=probs))


print("MCTS ready")

#%% [markdown]
# ## 4. Self-Play & Training

#%%
@dataclass
class GameRecord:
    states: List[np.ndarray] = field(default_factory=list)
    policies: List[np.ndarray] = field(default_factory=list)
    players: List[Player] = field(default_factory=list)
    winner: Player = None

    def add_move(self, state: np.ndarray, policy: np.ndarray, player: Player):
        self.states.append(state)
        self.policies.append(policy)
        self.players.append(player)

    def get_training_samples(self) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        samples = []
        for state, policy, player in zip(self.states, self.policies, self.players):
            value = 0.0 if self.winner is None else (1.0 if self.winner == player else -1.0)
            samples.append((state, policy, value))
        return samples


class SelfPlayWorker:
    def __init__(self, apply_fn, params, batch_stats, mcts_config: MCTSConfig):
        self.apply_fn = apply_fn
        self.params = params
        self.batch_stats = batch_stats
        self.mcts_config = mcts_config

    def play_game(self, temperature_threshold: int = 15) -> GameRecord:
        board = Board()
        record = GameRecord()
        mcts = MCTS(self.apply_fn, self.params, self.batch_stats, self.mcts_config)

        while not board.is_game_over:
            mcts.config.temperature = 1.0 if board.move_count < temperature_threshold else 0.1
            state = board.get_state()
            policy = mcts.search(board, add_noise=True)
            record.add_move(state, policy, board.current_player)
            action = mcts.select_action(policy, deterministic=False)
            board.play(action)

        record.winner = board.winner
        return record


class ReplayBuffer:
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.states, self.policies, self.values = [], [], []

    def add(self, states: np.ndarray, policies: np.ndarray, values: np.ndarray):
        for s, p, v in zip(states, policies, values):
            if len(self.states) >= self.max_size:
                self.states.pop(0)
                self.policies.pop(0)
                self.values.pop(0)
            self.states.append(s)
            self.policies.append(p)
            self.values.append(v)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        indices = np.random.choice(len(self.states), size=batch_size, replace=False)
        return (
            np.array([self.states[i] for i in indices]),
            np.array([self.policies[i] for i in indices]),
            np.array([self.values[i] for i in indices]),
        )

    def __len__(self) -> int:
        return len(self.states)


print("Self-play system ready")

#%% [markdown]
# ## 5. Training Configuration

#%%
# Hyperparameters - TPU optimized
NUM_ITERATIONS = 30
GAMES_PER_ITERATION = 50
MCTS_SIMULATIONS = 200
BATCH_SIZE = 256  # Larger batch for TPU
EPOCHS_PER_ITERATION = 5
LEARNING_RATE = 1e-3
MAX_BUFFER_SIZE = 50000

print(f"Training config (TPU optimized):")
print(f"  Iterations: {NUM_ITERATIONS}")
print(f"  Games/iter: {GAMES_PER_ITERATION}")
print(f"  Batch size: {BATCH_SIZE}")

#%% [markdown]
# ## 6. Training Loop

#%%
class TrainState(train_state.TrainState):
    batch_stats: Any


def create_train_state(rng, model, learning_rate):
    """Initialize training state with model and optimizer."""
    dummy_input = jnp.ones((1, 9, 9, 2))
    variables = model.init(rng, dummy_input, train=True)

    # Optimizer with gradient clipping
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate, weight_decay=1e-4),
    )

    return TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=tx,
        batch_stats=variables['batch_stats'],
    )


@jit
def train_step(state: TrainState, batch_states, batch_policies, batch_values):
    """Single training step - JIT compiled."""

    def loss_fn(params):
        variables = {'params': params, 'batch_stats': state.batch_stats}
        (log_policy, value), updates = state.apply_fn(
            variables, batch_states, train=True, mutable=['batch_stats']
        )

        # Policy loss: cross-entropy
        policy_loss = -jnp.sum(batch_policies * log_policy) / log_policy.shape[0]

        # Value loss: MSE
        value_loss = jnp.mean((value.squeeze() - batch_values) ** 2)

        total_loss = policy_loss + value_loss
        return total_loss, (policy_loss, value_loss, updates)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (policy_loss, value_loss, updates)), grads = grad_fn(state.params)

    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=updates['batch_stats'])

    return state, policy_loss, value_loss


def train():
    rng = random.PRNGKey(42)
    model = PolicyValueNet()
    state = create_train_state(rng, model, LEARNING_RATE)

    mcts_config = MCTSConfig(num_simulations=MCTS_SIMULATIONS)
    buffer = ReplayBuffer(MAX_BUFFER_SIZE)

    history = {"policy_loss": [], "value_loss": [], "game_length": []}

    for iteration in range(1, NUM_ITERATIONS + 1):
        print(f"\n{'='*50}")
        print(f"Iteration {iteration}/{NUM_ITERATIONS}")
        print(f"{'='*50}")

        # Self-play
        worker = SelfPlayWorker(
            model.apply, state.params, state.batch_stats, mcts_config
        )
        games = []

        for _ in tqdm(range(GAMES_PER_ITERATION), desc="Self-play"):
            game = worker.play_game()
            games.append(game)

        # Collect samples
        all_samples = []
        for game in games:
            all_samples.extend(game.get_training_samples())

        states = np.array([s[0] for s in all_samples], dtype=np.float32)
        policies = np.array([s[1] for s in all_samples], dtype=np.float32)
        values = np.array([s[2] for s in all_samples], dtype=np.float32)
        buffer.add(states, policies, values)

        avg_length = np.mean([len(g.states) for g in games])
        history["game_length"].append(avg_length)
        print(f"Avg game length: {avg_length:.1f}, Buffer: {len(buffer)}")

        # Training
        if len(buffer) < BATCH_SIZE:
            continue

        total_policy_loss, total_value_loss, num_batches = 0.0, 0.0, 0

        for epoch in range(EPOCHS_PER_ITERATION):
            for _ in range(len(buffer) // BATCH_SIZE):
                batch_states, batch_policies, batch_values = buffer.sample(BATCH_SIZE)

                # Convert to JAX arrays (NHWC format)
                batch_states = jnp.array(batch_states.transpose(0, 2, 3, 1))
                batch_policies = jnp.array(batch_policies)
                batch_values = jnp.array(batch_values)

                state, policy_loss, value_loss = train_step(
                    state, batch_states, batch_policies, batch_values
                )

                total_policy_loss += float(policy_loss)
                total_value_loss += float(value_loss)
                num_batches += 1

        avg_policy_loss = total_policy_loss / num_batches
        avg_value_loss = total_value_loss / num_batches
        history["policy_loss"].append(avg_policy_loss)
        history["value_loss"].append(avg_value_loss)

        print(f"Policy loss: {avg_policy_loss:.4f}, Value loss: {avg_value_loss:.4f}")

    return state, history


# Run training
print("Starting training on TPU...")
trained_state, history = train()
print("\nTraining complete!")

#%%
# Plot training curves
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(history["policy_loss"])
axes[0].set_title("Policy Loss")
axes[0].set_xlabel("Iteration")

axes[1].plot(history["value_loss"])
axes[1].set_title("Value Loss")
axes[1].set_xlabel("Iteration")

axes[2].plot(history["game_length"])
axes[2].set_title("Avg Game Length")
axes[2].set_xlabel("Iteration")

plt.tight_layout()
plt.show()

#%% [markdown]
# ## 7. Save & Download Model

#%%
import pickle

# Save as pickle (JAX-native format)
model_data = {
    "params": trained_state.params,
    "batch_stats": trained_state.batch_stats,
    "history": history,
}

with open("omok_model_jax.pkl", "wb") as f:
    pickle.dump(model_data, f)

print("Model saved to omok_model_jax.pkl")

# Download
try:
    from google.colab import files
    files.download("omok_model_jax.pkl")
    print("Download started!")
except ImportError:
    print("Not on Colab - model saved locally")

#%% [markdown]
# ## 8. Test the Model

#%%
def test_model(state):
    """Play a test game with the trained model."""
    board = Board()
    mcts_config = MCTSConfig(num_simulations=100, temperature=0.1)
    mcts = MCTS(PolicyValueNet().apply, state.params, state.batch_stats, mcts_config)

    while not board.is_game_over:
        probs = mcts.search(board, add_noise=False)
        action = mcts.select_action(probs, deterministic=True)
        board.play(action)

    # Display
    symbols = {Player.EMPTY: "·", Player.BLACK: "●", Player.WHITE: "○"}
    print("\nTest game result:")
    print("  " + " ".join(str(i) for i in range(9)))
    for r in range(9):
        row_str = f"{r} "
        for c in range(9):
            row_str += symbols[Player(board._board[r, c])] + " "
        print(row_str)

    if board.winner:
        print(f"\nWinner: {'Black' if board.winner == Player.BLACK else 'White'}")
    else:
        print("\nDraw!")

test_model(trained_state)
