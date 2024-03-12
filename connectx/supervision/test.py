import collections
import torch
from typing import Tuple

from sl_agent import SLAgent


def run_episode(
        env_steps: torch.Tensor,
        model: torch.nn.Module,
        max_steps: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Runs a single episode to collect training data."""

    policy_logits = [None] * max_steps
    values = [None] * max_steps
    rewards = [None] * max_steps
    action_probs = [None] * max_steps


    for step, env_output in enumerate(env_steps):
        # Run the model and to get action probabilities and critic value
        agent_output = model(env_output['obs'])

        # Store critic values
        values[step] = agent_output['baseline']

        # Store log probability of the action chosen
        policy_logits[step] = agent_output['policy_logits']

        # Store reward
        rewards[step] = env_output['action'] == agent_output['actions']

        # Store log probability of the action chosen
        action = agent_output['actions']
        action_prob = agent_output['policy_logits'][0, action]
        action_probs[step] = action_prob

    action_probs = torch.stack(action_probs)
    values = torch.stack(values)
    rewards = torch.stack(rewards)

    return action_probs, values, rewards

def get_expected_return(
    rewards: torch.Tensor,
    gamma: float,
    standardize: bool = True) -> torch.Tensor:
    """Compute expected returns per timestep."""
    eps = 1e-6
    returns = torch.zeros_like(rewards, dtype=torch.float32)

    rewards = rewards.to(dtype=torch.float32)
    discounted_sum = 0.
    for i,reward in enumerate(rewards):
        discounted_sum = reward + gamma * discounted_sum
        returns[i] = discounted_sum
    returns = torch.stack(returns)

    if standardize:
        returns = (returns - torch.mean(returns)) / (torch.std(returns) + eps)

    return returns

def compute_loss(
        action_probs: torch.Tensor,
        values: torch.Tensor,
        returns: torch.Tensor) -> torch.Tensor:
    """Computes the combined Actor-Critic loss."""

    advantage = returns - values

    action_log_probs = torch.math.log(action_probs)
    actor_loss = -torch.sum(action_log_probs * advantage)

    critic_loss = huber_loss(values, returns)

    return actor_loss + critic_loss

def train_step(
        env_steps: torch.Tensor,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        gamma: float,
        max_steps_per_episode: int,
) -> torch.Tensor:
    """Runs a model training step."""

    # print(f"\ntrain_step initial_state:\n{initial_state}")
    # Run the model for one episode to collect training data
    action_probs, values, rewards = run_episode(env_steps, model, max_steps_per_episode)

    # Calculate the expected returns
    returns = get_expected_return(rewards, gamma)

    # Calculate the loss values to update our network
    loss = compute_loss(action_probs, values, returns)

    # compute gradients (grad)
    loss.backward()

    # Apply the gradients to the model's parameters
    optimizer.step()
    optimizer.zero_grad()

    episode_reward = torch.sum(rewards)

    return episode_reward

min_episodes_criterion = 100
max_episodes = 10000
max_steps_per_episode = 500
learning_rate = 0.001

# consecutive trials
reward_threshold = 1
running_reward = 0

huber_loss = torch.nn.HuberLoss(reduction='mean', delta=1.0)

# The discount factor for future rewards
gamma = 0.99

# Keep the last episodes reward
episodes_reward: collections.deque = collections.deque(maxlen=min_episodes_criterion)
print(__file__)

if __name__=='__main__':
    train_step()