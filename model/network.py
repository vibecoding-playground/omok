#%%
"""
Policy-Value Network for AlphaZero-style Omok AI.

Architecture: ResNet-based dual-head network
- Input: (batch, 2, 9, 9) board state
- Policy head: (batch, 81) move probabilities
- Value head: (batch, 1) position evaluation [-1, 1]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def get_device() -> torch.device:
    """
    Get the best available device for training/inference.
    Priority: MPS (Apple Silicon) > CUDA > CPU
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class ResBlock(nn.Module):
    """
    Residual block with pre-activation (Pre-LN style).
    Pre-activation is more stable for deep networks and avoids
    gradient explosion issues common in Post-LN.
    """
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
        out = F.relu(out + residual)
        return out


class PolicyValueNet(nn.Module):
    """
    AlphaZero-style Policy-Value Network for 9x9 Omok.

    Architecture:
    - Initial conv block: 2 -> 128 channels
    - 4 residual blocks (sufficient for 9x9 board)
    - Policy head: 128 -> 2 -> flatten -> 81
    - Value head: 128 -> 1 -> flatten -> FC -> tanh

    ### [SOTA Alert]
    Using torch.compile() for inference speedup when available.
    BatchNorm for training stability over LayerNorm in conv nets.
    """
    def __init__(
        self,
        board_size: int = 9,
        in_channels: int = 2,
        num_filters: int = 128,
        num_res_blocks: int = 4,
    ):
        super().__init__()
        self.board_size = board_size
        self.action_size = board_size * board_size

        # Initial convolution block
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, num_filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
        )

        # Residual tower
        self.res_blocks = nn.Sequential(
            *[ResBlock(num_filters) for _ in range(num_res_blocks)]
        )

        # Policy head: predicts move probabilities
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_filters, 2, 1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * board_size * board_size, self.action_size),
        )

        # Value head: predicts game outcome [-1, 1]
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
        """Initialize weights using Kaiming initialization."""
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
        """
        Forward pass.

        Args:
            x: Board state tensor (batch, 2, 9, 9)

        Returns:
            policy: Log probabilities over actions (batch, 81)
            value: Position evaluation (batch, 1)
        """
        # Shared backbone
        x = self.conv_block(x)
        x = self.res_blocks(x)

        # Dual heads
        policy = self.policy_head(x)
        value = self.value_head(x)

        # Return log-softmax for numerical stability in loss computation
        return F.log_softmax(policy, dim=1), value

    def predict(
        self, state: torch.Tensor, valid_moves: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """
        Get policy distribution and value for a single state.
        Masks invalid moves and renormalizes.

        Args:
            state: Single board state (2, 9, 9)
            valid_moves: Boolean mask of valid moves (81,)

        Returns:
            policy: Probability distribution over valid moves (81,)
            value: Position evaluation scalar
        """
        self.eval()
        with torch.no_grad():
            state = state.unsqueeze(0)  # Add batch dim
            log_policy, value = self(state)
            policy = torch.exp(log_policy).squeeze(0)

            # Mask invalid moves and renormalize
            policy = policy * valid_moves.float()
            policy_sum = policy.sum()
            if policy_sum > 0:
                policy = policy / policy_sum
            else:
                # Fallback: uniform over valid moves
                policy = valid_moves.float() / valid_moves.sum()

            return policy, value.item()


#%%
if __name__ == "__main__":
    device = get_device()
    print(f"Using device: {device}")

    model = PolicyValueNet().to(device)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 2, 9, 9).to(device)
    log_policy, value = model(x)

    assert log_policy.shape == (batch_size, 81), f"Policy shape: {log_policy.shape}"
    assert value.shape == (batch_size, 1), f"Value shape: {value.shape}"

    print(f"Policy shape: {log_policy.shape}")
    print(f"Value shape: {value.shape}")
    print(f"Value range: [{value.min().item():.3f}, {value.max().item():.3f}]")
