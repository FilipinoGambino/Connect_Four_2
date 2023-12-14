import torch
from torch.nn import LeakyReLU, Linear, Sequential

from typing import Tuple

from ..utility_constants import BOARD_SIZE

NUM_ACTIONS = BOARD_SIZE[1]

class ActorCritic(torch.nn.Module):
    """Combined actor-critic network"""
    def __init__(self, num_input_units):
        super().__init__()

        num_hidden_units = 128
        self.common = Sequential(
            Linear(num_input_units, num_hidden_units),
            LeakyReLU(),
        )
        self.actor = Linear(num_hidden_units, NUM_ACTIONS)
        self.critic = Linear(num_hidden_units, 1)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.common(inputs)
        actor_logits = self.actor(x)
        actor_actions = ActorCritic.logits_to_actions(actor_logits)
        critic_value = self.critic(x)
        return actor_logits, actor_actions, critic_value

    @staticmethod
    @torch.no_grad()
    def logits_to_actions(logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(dim=0)
