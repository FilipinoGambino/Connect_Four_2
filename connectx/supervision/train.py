import torch
from typing import Tuple

from sl_agent import SLAgent

def run_episode(
        initial_state: torch.Tensor,
        model: torch.nn.Module,
        max_steps: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Runs a single episode to collect training data."""

    policy_logits = [None] * max_steps
    values = [None] * max_steps
    rewards = [None] * max_steps

    env_output = initial_state

    for step in range(max_steps):
        # Run the model and to get action probabilities and critic value
        agent_output = model(env_output)

        # Store critic values
        values[step] = agent_output['baseline']

        # Store log probability of the action chosen
        policy_logits[step] = agent_output['policy_logits']

        # Store reward
        rewards[step] = reward

        # Store log probability of the action chosen
        action = info['action']
        action_probs_step = info['masked_logits']
        action_prob = action_probs_step[0, action]
        action_probs.append(action_prob)

    action_probs = torch.stack(action_probs)
    values = torch.stack(values)
    rewards = torch.stack(rewards)

    return action_probs, values, rewards