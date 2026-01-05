#%%
"""
Self-Play data generation for AlphaZero-style training.

Each self-play game produces training samples:
- state: board position
- policy: MCTS visit count distribution
- value: final game outcome from that position's perspective
"""
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List, Tuple
from tqdm import tqdm

from game.board import Board, Player
from model.network import PolicyValueNet
from mcts.search import MCTS, MCTSConfig


@dataclass
class GameRecord:
    """Record of a single self-play game."""
    states: List[np.ndarray] = field(default_factory=list)
    policies: List[np.ndarray] = field(default_factory=list)
    players: List[Player] = field(default_factory=list)  # Who played each move
    winner: Player = None

    def add_move(self, state: np.ndarray, policy: np.ndarray, player: Player):
        self.states.append(state)
        self.policies.append(policy)
        self.players.append(player)

    def get_training_samples(self) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """
        Convert game record to training samples.

        Returns list of (state, policy, value) tuples.
        Value is +1 if the player who played this move won, -1 if lost, 0 if draw.
        """
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
    """
    Generates self-play games using MCTS.

    The worker plays games against itself, recording:
    - Board states
    - MCTS policy distributions
    - Game outcomes

    These are used as training data for the neural network.
    """
    def __init__(
        self,
        model: PolicyValueNet,
        mcts_config: MCTSConfig = None,
        device: torch.device = None,
    ):
        self.model = model
        self.mcts_config = mcts_config or MCTSConfig()
        self.device = device or torch.device("cpu")

    def play_game(self, temperature_threshold: int = 15) -> GameRecord:
        """
        Play a single self-play game.

        Args:
            temperature_threshold: Use temperature=1 for first N moves,
                                   then temperature=0 (deterministic).
                                   This adds diversity in openings.

        Returns:
            GameRecord containing all moves and outcome.
        """
        board = Board()
        record = GameRecord()
        mcts = MCTS(self.model, self.mcts_config, self.device)

        while not board.is_game_over:
            # Use high temperature early game for diversity
            if board.move_count < temperature_threshold:
                mcts.config.temperature = 1.0
            else:
                mcts.config.temperature = 0.1  # Nearly deterministic

            # Run MCTS search
            state = board.get_state()
            policy = mcts.search(board, add_noise=True)

            # Record move
            record.add_move(state, policy, board.current_player)

            # Select and play action
            action = mcts.select_action(policy, deterministic=False)
            board.play(action)

        record.winner = board.winner
        return record

    def generate_games(
        self,
        num_games: int,
        temperature_threshold: int = 15,
        verbose: bool = True,
    ) -> List[GameRecord]:
        """Generate multiple self-play games."""
        self.model.eval()
        games = []

        iterator = range(num_games)
        if verbose:
            iterator = tqdm(iterator, desc="Self-play")

        for _ in iterator:
            game = self.play_game(temperature_threshold)
            games.append(game)

        return games


def collect_training_data(
    games: List[GameRecord],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Collect training samples from multiple games.

    Returns:
        states: (N, 2, 9, 9) board states
        policies: (N, 81) MCTS policies
        values: (N,) game outcomes
    """
    all_samples = []
    for game in games:
        all_samples.extend(game.get_training_samples())

    states = np.array([s[0] for s in all_samples], dtype=np.float32)
    policies = np.array([s[1] for s in all_samples], dtype=np.float32)
    values = np.array([s[2] for s in all_samples], dtype=np.float32)

    return states, policies, values


#%%
if __name__ == "__main__":
    from model.network import get_device

    device = get_device()
    print(f"Device: {device}")

    model = PolicyValueNet().to(device)
    config = MCTSConfig(num_simulations=50)  # Fewer sims for quick test

    worker = SelfPlayWorker(model, config, device)

    print("Playing test game...")
    game = worker.play_game()

    print(f"Game length: {len(game.states)} moves")
    print(f"Winner: {game.winner}")

    samples = game.get_training_samples()
    print(f"Training samples: {len(samples)}")
    print(f"Sample state shape: {samples[0][0].shape}")
    print(f"Sample policy shape: {samples[0][1].shape}")
