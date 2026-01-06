#%% [markdown]
# # Omok (오목) AI - Colab Training (GPU)
#
# AlphaZero 스타일 강화학습으로 9x9 오목 AI를 학습합니다.
#
# **특징:**
# - Google Drive에 체크포인트 자동 저장
# - 세션 종료 후에도 이어서 학습 가능
#
# **사용법:**
# 1. 런타임 > 런타임 유형 변경 > **GPU** 선택
# 2. 모든 셀 실행 (기존 체크포인트가 있으면 자동으로 이어서 학습)

#%% [markdown]
# ## 0. Google Drive 마운트

#%%
from google.colab import drive
drive.mount('/content/drive')

import os
SAVE_DIR = '/content/drive/MyDrive/omok'
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"Save directory: {SAVE_DIR}")

#%%
# Setup
import subprocess
subprocess.run(["pip", "install", "-q", "torch", "numpy", "tqdm", "matplotlib"])

#%%
import math
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from enum import IntEnum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from copy import deepcopy
from tqdm.auto import tqdm

# Device 설정 - Colab GPU 사용
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

#%% [markdown]
# ## 1. Game Logic

#%%
class Player(IntEnum):
    BLACK = 1
    WHITE = -1
    EMPTY = 0


class Board:
    """9x9 Omok board with AlphaZero-compatible state representation."""
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
        """Get board state as (2, 9, 9) tensor for neural network."""
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
            raise ValueError(f"Invalid position: ({row}, {col})")
        if self._board[row, col] != Player.EMPTY:
            raise ValueError(f"Position ({row}, {col}) is already occupied")

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


# Quick test
board = Board()
print(f"Board size: {Board.SIZE}x{Board.SIZE}")
print(f"State shape: {board.get_state().shape}")

#%% [markdown]
# ## 2. Neural Network

#%%
class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class PolicyValueNet(nn.Module):
    """AlphaZero-style Policy-Value Network."""

    def __init__(self, board_size: int = 9, in_channels: int = 2,
                 num_filters: int = 128, num_res_blocks: int = 4):
        super().__init__()
        self.board_size = board_size
        self.action_size = board_size * board_size

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, num_filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
        )
        self.res_blocks = nn.Sequential(*[ResBlock(num_filters) for _ in range(num_res_blocks)])

        self.policy_head = nn.Sequential(
            nn.Conv2d(num_filters, 2, 1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * board_size * board_size, self.action_size),
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(num_filters, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(board_size * board_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv_block(x)
        x = self.res_blocks(x)
        policy = F.log_softmax(self.policy_head(x), dim=1)
        value = self.value_head(x)
        return policy, value

    def predict(self, state: torch.Tensor, valid_moves: torch.Tensor) -> Tuple[torch.Tensor, float]:
        self.eval()
        with torch.no_grad():
            state = state.unsqueeze(0)
            log_policy, value = self(state)
            policy = torch.exp(log_policy).squeeze(0)
            policy = policy * valid_moves.float()
            policy_sum = policy.sum()
            if policy_sum > 0:
                policy = policy / policy_sum
            else:
                policy = valid_moves.float() / valid_moves.sum()
            return policy, value.item()


model = PolicyValueNet().to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

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
    def __init__(self, model: PolicyValueNet, config: MCTSConfig, device: torch.device):
        self.model = model
        self.config = config
        self.device = device
        self.root: Optional[Node] = None

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
        state = torch.from_numpy(board.get_state()).to(self.device)
        valid_moves = torch.from_numpy(board.get_valid_moves()).to(self.device)
        policy, value = self.model.predict(state, valid_moves)
        policy = policy.cpu().numpy()

        for action in range(len(policy)):
            if valid_moves[action]:
                node.children[action] = Node(prior=policy[action])
        node.is_expanded = True
        return value

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


print("MCTS initialized")

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
            if self.winner is None:
                value = 0.0
            elif self.winner == player:
                value = 1.0
            else:
                value = -1.0
            samples.append((state, policy, value))
        return samples


class SelfPlayWorker:
    def __init__(self, model: PolicyValueNet, mcts_config: MCTSConfig, device: torch.device):
        self.model = model
        self.mcts_config = mcts_config
        self.device = device

    def play_game(self, temperature_threshold: int = 15) -> GameRecord:
        board = Board()
        record = GameRecord()
        mcts = MCTS(self.model, self.mcts_config, self.device)

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

    def get_all(self) -> Tuple[List, List, List]:
        return self.states, self.policies, self.values

    def load(self, states, policies, values):
        self.states = list(states)
        self.policies = list(policies)
        self.values = list(values)

    def __len__(self) -> int:
        return len(self.states)


print("Self-play system ready")

#%% [markdown]
# ## 5. Training Configuration

#%%
# Training hyperparameters - GPU 최적화
NUM_ITERATIONS = 50          # 학습 반복 횟수
GAMES_PER_ITERATION = 50     # 이터레이션당 자가대국 수
MCTS_SIMULATIONS = 200       # MCTS 시뮬레이션 횟수
BATCH_SIZE = 128             # 배치 사이즈 (GPU 메모리 활용)
EPOCHS_PER_ITERATION = 5     # 이터레이션당 학습 에폭
LEARNING_RATE = 1e-3
MAX_BUFFER_SIZE = 50000

# Checkpoint paths
CHECKPOINT_PATH = os.path.join(SAVE_DIR, "checkpoint.pt")
BUFFER_PATH = os.path.join(SAVE_DIR, "replay_buffer.pkl")
HISTORY_PATH = os.path.join(SAVE_DIR, "history.pkl")

print(f"Training config:")
print(f"  Iterations: {NUM_ITERATIONS}")
print(f"  Games/iter: {GAMES_PER_ITERATION}")
print(f"  MCTS sims: {MCTS_SIMULATIONS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Checkpoint: {CHECKPOINT_PATH}")

#%% [markdown]
# ## 6. Checkpoint Functions

#%%
def save_checkpoint(model, optimizer, scheduler, iteration, history, buffer):
    """Save training state to Google Drive."""
    # Save model & optimizer
    torch.save({
        "iteration": iteration,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }, CHECKPOINT_PATH)

    # Save replay buffer
    states, policies, values = buffer.get_all()
    with open(BUFFER_PATH, "wb") as f:
        pickle.dump({"states": states, "policies": policies, "values": values}, f)

    # Save history
    with open(HISTORY_PATH, "wb") as f:
        pickle.dump(history, f)

    print(f"  💾 Checkpoint saved (iteration {iteration})")


def load_checkpoint(model, optimizer, scheduler, buffer):
    """Load training state from Google Drive if exists."""
    if not os.path.exists(CHECKPOINT_PATH):
        print("No checkpoint found. Starting fresh.")
        return 0, {"policy_loss": [], "value_loss": [], "game_length": []}

    # Load model & optimizer
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    iteration = checkpoint["iteration"]

    # Load replay buffer
    if os.path.exists(BUFFER_PATH):
        with open(BUFFER_PATH, "rb") as f:
            buf_data = pickle.load(f)
            buffer.load(buf_data["states"], buf_data["policies"], buf_data["values"])

    # Load history
    history = {"policy_loss": [], "value_loss": [], "game_length": []}
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "rb") as f:
            history = pickle.load(f)

    print(f"✅ Checkpoint loaded! Resuming from iteration {iteration}")
    print(f"   Buffer size: {len(buffer)}")
    return iteration, history

#%% [markdown]
# ## 7. Train!

#%%
def train():
    model = PolicyValueNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    buffer = ReplayBuffer(MAX_BUFFER_SIZE)
    mcts_config = MCTSConfig(num_simulations=MCTS_SIMULATIONS)

    # Load checkpoint if exists
    start_iteration, history = load_checkpoint(model, optimizer, scheduler, buffer)

    if start_iteration >= NUM_ITERATIONS:
        print(f"Already completed {NUM_ITERATIONS} iterations!")
        return model, history

    for iteration in range(start_iteration + 1, NUM_ITERATIONS + 1):
        print(f"\n{'='*50}")
        print(f"Iteration {iteration}/{NUM_ITERATIONS}")
        print(f"{'='*50}")

        # Self-play
        model.eval()
        worker = SelfPlayWorker(model, mcts_config, device)
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
            save_checkpoint(model, optimizer, scheduler, iteration, history, buffer)
            continue

        model.train()
        total_policy_loss, total_value_loss, num_batches = 0.0, 0.0, 0

        for epoch in range(EPOCHS_PER_ITERATION):
            for _ in range(len(buffer) // BATCH_SIZE):
                batch_states, batch_policies, batch_values = buffer.sample(BATCH_SIZE)

                states_t = torch.from_numpy(batch_states).to(device)
                policies_t = torch.from_numpy(batch_policies).to(device)
                values_t = torch.from_numpy(batch_values).to(device).unsqueeze(1)

                log_policy, value = model(states_t)
                policy_loss = -torch.sum(policies_t * log_policy) / log_policy.shape[0]
                value_loss = F.mse_loss(value, values_t)
                loss = policy_loss + value_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                num_batches += 1

        scheduler.step()

        avg_policy_loss = total_policy_loss / num_batches
        avg_value_loss = total_value_loss / num_batches
        history["policy_loss"].append(avg_policy_loss)
        history["value_loss"].append(avg_value_loss)

        print(f"Policy loss: {avg_policy_loss:.4f}, Value loss: {avg_value_loss:.4f}")

        # Save checkpoint after every iteration
        save_checkpoint(model, optimizer, scheduler, iteration, history, buffer)

    return model, history


# Run training
print("Starting training...")
trained_model, history = train()
print("\n🎉 Training complete!")

#%%
# Plot training curves
if len(history["policy_loss"]) > 0:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history["policy_loss"])
    axes[0].set_title("Policy Loss")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Loss")

    axes[1].plot(history["value_loss"])
    axes[1].set_title("Value Loss")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Loss")

    axes[2].plot(history["game_length"])
    axes[2].set_title("Average Game Length")
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("Moves")

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "training_curves.png"))
    plt.show()
    print(f"Training curves saved to {SAVE_DIR}/training_curves.png")

#%% [markdown]
# ## 8. Export Final Model

#%%
# Save final model for local use
FINAL_MODEL_PATH = os.path.join(SAVE_DIR, "omok_model.pt")
torch.save({
    "model_state_dict": trained_model.state_dict(),
    "history": history,
}, FINAL_MODEL_PATH)
print(f"Final model saved to {FINAL_MODEL_PATH}")

# Download option
try:
    from google.colab import files
    files.download(FINAL_MODEL_PATH)
except:
    pass

#%% [markdown]
# ## 9. Test the Model

#%%
def test_model(model):
    """Play a test game and show the result."""
    model.eval()
    board = Board()
    mcts_config = MCTSConfig(num_simulations=100, temperature=0.1)
    mcts = MCTS(model, mcts_config, device)

    moves = []
    while not board.is_game_over:
        probs = mcts.search(board, add_noise=False)
        action = mcts.select_action(probs, deterministic=True)
        row, col = divmod(action, 9)
        moves.append((board.current_player, row, col))
        board.play(action)

    # Display final board
    symbols = {Player.EMPTY: "·", Player.BLACK: "●", Player.WHITE: "○"}
    print("\nTest game result:")
    print("  " + " ".join(str(i) for i in range(9)))
    for r in range(9):
        row_str = f"{r} "
        for c in range(9):
            row_str += symbols[Player(board._board[r, c])] + " "
        print(row_str)

    if board.winner:
        winner = "Black ●" if board.winner == Player.BLACK else "White ○"
        print(f"\nWinner: {winner} ({len(moves)} moves)")
    else:
        print(f"\nDraw! ({len(moves)} moves)")

test_model(trained_model)
