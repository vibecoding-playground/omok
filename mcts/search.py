#%%
"""
Monte Carlo Tree Search (MCTS) with AlphaZero-style neural network guidance.

Key differences from vanilla MCTS:
- Uses neural network policy as prior for action selection
- Uses neural network value instead of random rollouts
- PUCT formula for exploration-exploitation balance
"""
import math
import numpy as np
import torch
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

from game.board import Board, Player
from model.network import PolicyValueNet


@dataclass
class MCTSConfig:
    """MCTS hyperparameters."""
    num_simulations: int = 200    # Number of MCTS simulations per move
    c_puct: float = 1.5           # Exploration constant in PUCT formula
    dirichlet_alpha: float = 0.3  # Dirichlet noise for root exploration
    dirichlet_epsilon: float = 0.25  # Weight of Dirichlet noise at root
    temperature: float = 1.0      # Temperature for action selection


class Node:
    """
    MCTS tree node storing statistics for PUCT computation.

    Each node represents a game state after taking an action from parent.
    """
    def __init__(self, prior: float):
        self.prior = prior        # P(s, a) from neural network
        self.visit_count = 0      # N(s, a)
        self.value_sum = 0.0      # W(s, a) = sum of values from this node
        self.children: Dict[int, "Node"] = {}  # action -> child node
        self.is_expanded = False

    @property
    def value(self) -> float:
        """Q(s, a) = W(s, a) / N(s, a)"""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, parent_visit_count: int, c_puct: float) -> float:
        """
        PUCT score = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))

        Balances exploitation (Q) with exploration (prior * sqrt term).
        Higher prior means we explore that action more initially.
        """
        exploration = c_puct * self.prior * math.sqrt(parent_visit_count) / (1 + self.visit_count)
        return self.value + exploration

    def select_child(self, c_puct: float) -> Tuple[int, "Node"]:
        """Select child with highest UCB score."""
        best_score = -float("inf")
        best_action = -1
        best_child = None

        for action, child in self.children.items():
            score = child.ucb_score(self.visit_count, c_puct)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child


class MCTS:
    """
    AlphaZero-style MCTS using neural network for evaluation.

    The search process:
    1. Selection: Traverse tree using PUCT until reaching unexpanded node
    2. Expansion: Use neural network policy to initialize child priors
    3. Evaluation: Use neural network value to evaluate leaf
    4. Backup: Propagate value up the tree (negated at each level)
    """
    def __init__(
        self,
        model: PolicyValueNet,
        config: MCTSConfig = None,
        device: torch.device = None,
    ):
        self.model = model
        self.config = config or MCTSConfig()
        self.device = device or torch.device("cpu")
        self.root: Optional[Node] = None

    def search(self, board: Board, add_noise: bool = True) -> np.ndarray:
        """
        Run MCTS from current position and return action probabilities.

        Args:
            board: Current game state
            add_noise: Whether to add Dirichlet noise at root (for training)

        Returns:
            Action probabilities based on visit counts (81,)
        """
        # Initialize root
        self.root = Node(prior=0.0)
        self._expand(self.root, board)

        # Add Dirichlet noise at root for exploration during training
        if add_noise:
            self._add_dirichlet_noise(self.root, board.get_valid_moves())

        # Run simulations
        for _ in range(self.config.num_simulations):
            node = self.root
            scratch_board = board.copy()
            search_path = [node]

            # Selection: traverse tree until we find unexpanded node
            while node.is_expanded and not scratch_board.is_game_over:
                action, node = node.select_child(self.config.c_puct)
                scratch_board.play(action)
                search_path.append(node)

            # Get value for leaf node
            if scratch_board.is_game_over:
                # Terminal node: use actual game outcome
                if scratch_board.winner is None:
                    value = 0.0  # Draw
                else:
                    # Winner is from perspective of player who just moved
                    # We need value from perspective of player to move
                    value = -1.0  # Current player lost
            else:
                # Non-terminal: expand and use neural network value
                value = self._expand(node, scratch_board)

            # Backup: propagate value up the tree
            # Value alternates sign because players alternate
            self._backup(search_path, value)

        return self._get_action_probs(board.get_valid_moves())

    def _expand(self, node: Node, board: Board) -> float:
        """
        Expand node using neural network.

        Returns the value estimate from the neural network.
        """
        state = torch.from_numpy(board.get_state()).to(self.device)
        valid_moves = torch.from_numpy(board.get_valid_moves()).to(self.device)

        policy, value = self.model.predict(state, valid_moves)
        policy = policy.cpu().numpy()

        # Create children for valid moves
        for action in range(len(policy)):
            if valid_moves[action]:
                node.children[action] = Node(prior=policy[action])

        node.is_expanded = True
        return value

    def _add_dirichlet_noise(self, node: Node, valid_moves: np.ndarray):
        """Add Dirichlet noise to root node priors for exploration."""
        valid_actions = np.where(valid_moves)[0]
        noise = np.random.dirichlet([self.config.dirichlet_alpha] * len(valid_actions))

        for i, action in enumerate(valid_actions):
            if action in node.children:
                node.children[action].prior = (
                    (1 - self.config.dirichlet_epsilon) * node.children[action].prior
                    + self.config.dirichlet_epsilon * noise[i]
                )

    def _backup(self, search_path: List[Node], value: float):
        """
        Backup value through the search path.
        Value is negated at each step because players alternate.
        """
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            value = -value  # Flip for opponent's perspective

    def _get_action_probs(self, valid_moves: np.ndarray) -> np.ndarray:
        """
        Get action probabilities from visit counts.
        With temperature=1, probs are proportional to visit counts.
        With temperature->0, we select the most visited action.
        """
        visit_counts = np.zeros(81, dtype=np.float32)

        for action, child in self.root.children.items():
            visit_counts[action] = child.visit_count

        if self.config.temperature == 0:
            # Deterministic: pick most visited
            probs = np.zeros_like(visit_counts)
            probs[np.argmax(visit_counts)] = 1.0
        else:
            # Stochastic: proportional to visit counts ^ (1/temp)
            visit_counts = visit_counts ** (1 / self.config.temperature)
            probs = visit_counts / visit_counts.sum()

        return probs

    def select_action(self, probs: np.ndarray, deterministic: bool = False) -> int:
        """Select action from probability distribution."""
        if deterministic:
            return int(np.argmax(probs))
        return int(np.random.choice(len(probs), p=probs))


#%%
if __name__ == "__main__":
    from model.network import get_device

    device = get_device()
    print(f"Device: {device}")

    model = PolicyValueNet().to(device)
    model.eval()

    config = MCTSConfig(num_simulations=50)
    mcts = MCTS(model, config, device)

    board = Board()
    print(board)

    # Run search
    probs = mcts.search(board, add_noise=False)
    action = mcts.select_action(probs, deterministic=True)

    row, col = divmod(action, 9)
    print(f"\nBest move: ({row}, {col})")
    print(f"Visit distribution (top 5):")
    top_actions = np.argsort(probs)[-5:][::-1]
    for a in top_actions:
        r, c = divmod(a, 9)
        print(f"  ({r}, {c}): {probs[a]:.3f}")
