import torch
from torch.nn import LeakyReLU, Linear, Sequential

from typing import Tuple

from ..utility_constants import BOARD_SIZE

NUM_ACTIONS = BOARD_SIZE[1]

class ActorCritic(torch.nn.Module):
    """Combined actor-critic network"""
    def __init__(self, num_hidden_units: int):
        super().__init__()

        self.common = Sequential(
            Linear(388, num_hidden_units),
            LeakyReLU(),
        )
        self.actor = Linear(num_hidden_units, NUM_ACTIONS)
        self.critic = Linear(num_hidden_units, 1)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.common(inputs)
        return self.actor(x), self.critic(x)
