#%%
"""
Training loop for AlphaZero-style Omok AI.

The training process alternates between:
1. Self-play: Generate games using current model
2. Training: Update model on collected data
3. Evaluation: (Optional) Compare new model vs old
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional
from tqdm import tqdm

from game.board import Board
from model.network import PolicyValueNet, get_device
from mcts.search import MCTS, MCTSConfig
from training.self_play import SelfPlayWorker, collect_training_data


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    # Self-play
    num_games_per_iteration: int = 100
    mcts_simulations: int = 200
    temperature_threshold: int = 15

    # Training
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs_per_iteration: int = 5

    # Replay buffer
    max_buffer_size: int = 50000

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_interval: int = 5  # Save every N iterations


class ReplayBuffer:
    """
    Circular buffer for storing training samples.
    Older samples are discarded when buffer is full.
    """
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.states = []
        self.policies = []
        self.values = []

    def add(
        self,
        states: np.ndarray,
        policies: np.ndarray,
        values: np.ndarray,
    ):
        """Add new samples to buffer."""
        for s, p, v in zip(states, policies, values):
            if len(self.states) >= self.max_size:
                # Remove oldest samples
                self.states.pop(0)
                self.policies.pop(0)
                self.values.pop(0)
            self.states.append(s)
            self.policies.append(p)
            self.values.append(v)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample a random batch from buffer."""
        indices = np.random.choice(len(self.states), size=batch_size, replace=False)
        return (
            np.array([self.states[i] for i in indices]),
            np.array([self.policies[i] for i in indices]),
            np.array([self.values[i] for i in indices]),
        )

    def __len__(self) -> int:
        return len(self.states)


class Trainer:
    """
    Main training orchestrator for AlphaZero-style learning.

    Each iteration:
    1. Generate self-play games
    2. Add samples to replay buffer
    3. Train network on samples from buffer
    4. Save checkpoint periodically
    """
    def __init__(
        self,
        config: TrainingConfig = None,
        device: torch.device = None,
    ):
        self.config = config or TrainingConfig()
        self.device = device or get_device()

        # Initialize model
        self.model = PolicyValueNet().to(self.device)

        # Optimizer: AdamW provides better regularization than Adam
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )

        # Replay buffer
        self.buffer = ReplayBuffer(self.config.max_buffer_size)

        # MCTS config for self-play
        self.mcts_config = MCTSConfig(
            num_simulations=self.config.mcts_simulations,
        )

        # Training stats
        self.iteration = 0

        # Ensure checkpoint directory exists
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def train_iteration(self, verbose: bool = True) -> dict:
        """
        Run one training iteration: self-play + training.

        Returns dict with training statistics.
        """
        self.iteration += 1
        stats = {"iteration": self.iteration}

        # 1. Self-play phase
        if verbose:
            print(f"\n=== Iteration {self.iteration} ===")
            print("Generating self-play games...")

        self.model.eval()
        worker = SelfPlayWorker(self.model, self.mcts_config, self.device)
        games = worker.generate_games(
            self.config.num_games_per_iteration,
            self.config.temperature_threshold,
            verbose=verbose,
        )

        # Collect stats
        game_lengths = [len(g.states) for g in games]
        stats["avg_game_length"] = np.mean(game_lengths)
        stats["games_generated"] = len(games)

        # 2. Add to replay buffer
        states, policies, values = collect_training_data(games)
        self.buffer.add(states, policies, values)
        stats["buffer_size"] = len(self.buffer)

        # 3. Training phase
        if len(self.buffer) < self.config.batch_size:
            if verbose:
                print("Not enough samples for training yet.")
            return stats

        if verbose:
            print("Training network...")

        self.model.train()
        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_batches = 0

        for epoch in range(self.config.epochs_per_iteration):
            # Number of batches per epoch
            batches_per_epoch = len(self.buffer) // self.config.batch_size

            for _ in range(batches_per_epoch):
                batch_states, batch_policies, batch_values = self.buffer.sample(
                    self.config.batch_size
                )

                # Convert to tensors
                states_t = torch.from_numpy(batch_states).to(self.device)
                policies_t = torch.from_numpy(batch_policies).to(self.device)
                values_t = torch.from_numpy(batch_values).to(self.device).unsqueeze(1)

                # Forward pass
                log_policy, value = self.model(states_t)

                # Policy loss: cross-entropy with MCTS policy
                # We use KL divergence: -sum(target * log(pred))
                policy_loss = -torch.sum(policies_t * log_policy) / log_policy.shape[0]

                # Value loss: MSE between predicted and actual value
                value_loss = nn.functional.mse_loss(value, values_t)

                # Combined loss
                loss = policy_loss + value_loss

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                num_batches += 1

        self.scheduler.step()

        stats["policy_loss"] = total_policy_loss / num_batches
        stats["value_loss"] = total_value_loss / num_batches
        stats["learning_rate"] = self.optimizer.param_groups[0]["lr"]

        if verbose:
            print(f"Policy loss: {stats['policy_loss']:.4f}")
            print(f"Value loss: {stats['value_loss']:.4f}")

        # 4. Save checkpoint
        if self.iteration % self.config.save_interval == 0:
            self.save_checkpoint()

        return stats

    def train(self, num_iterations: int, verbose: bool = True):
        """Run multiple training iterations."""
        for _ in range(num_iterations):
            self.train_iteration(verbose=verbose)

    def save_checkpoint(self, path: Optional[str] = None):
        """Save model checkpoint."""
        if path is None:
            path = os.path.join(
                self.config.checkpoint_dir,
                f"model_iter_{self.iteration}.pt"
            )

        torch.save({
            "iteration": self.iteration,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }, path)
        print(f"Checkpoint saved: {path}")

        # Also save as "latest"
        latest_path = os.path.join(self.config.checkpoint_dir, "model_latest.pt")
        torch.save({
            "iteration": self.iteration,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }, latest_path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.iteration = checkpoint["iteration"]
        print(f"Loaded checkpoint from iteration {self.iteration}")


#%%
if __name__ == "__main__":
    # Quick training test
    device = get_device()
    print(f"Device: {device}")

    config = TrainingConfig(
        num_games_per_iteration=5,  # Very few for quick test
        mcts_simulations=50,
        epochs_per_iteration=2,
    )

    trainer = Trainer(config, device)
    stats = trainer.train_iteration(verbose=True)

    print("\nTraining stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
