import numpy as np
import tensorflow as tf
import gym
import collections
import statistics
import tqdm
import os
import yaml
from types import SimpleNamespace

from connectx.connectx_gym import create_env
from connectx.nns import create_model
from connectx.utils import flags_to_namespace

from typing import Tuple

import yaml
import torch


# path = '.\\connectx\\base_replays\\'
# for dir, folders, files in os.walk(path):
#     fnames = files
#
# for fname in fnames:
#     with open(f"{path}{fname}", 'r') as data:
#         match = json.load(data)
#     break
#
# for key,val in match.items():
#     print(key)
#     for key_,val_ in val.items():
#         if key_ == "board":
#             print(np.array(val_).reshape((6,7)))
#         else:
#             print(val_)
#     print()

# Wrap Gym's `env.step` call as an operation in a TensorFlow function.
# This would allow it to be included in a callable TensorFlow graph.

# Create the environment

# env = gym.make('connectx_env:connectx_env/Connect_Four-v0')

def load_object(dct):
    return SimpleNamespace(**dct)

with open("connectx/agent/model_config.yaml", 'r') as file:
    flags = flags_to_namespace(yaml.safe_load(file))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
env = create_env(flags, device)

# Set seed for experiment reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

# Small epsilon value for stabilizing division operations
eps = np.finfo(np.float32).eps.item()

model = create_model(flags, device)

# def env_step(action_tensors: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
#     """Returns state, reward, done, info flag given an action."""
#
#     state, reward, done, info = env.step(action_tensors)
#     print(f"\nstate:\n{state}")
#     return (state.astype(np.float32),
#             np.array(reward, np.int32),
#             np.array(done, np.int32),
#             info)

def run_episode(
        initial_state: torch.Tensor,
        model: torch.nn.Module,
        max_steps: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Runs a single episode to collect training data."""

    action_probs = []
    values = []
    rewards = []

    # initial_state_shape = initial_state.shape
    # print(f"\nrun_episode initial state:\n{initial_state}")
    state = initial_state

    for step in range(max_steps):
        # Convert state into a batched tensor (batch size = 1)
        # state = tf.expand_dims(state, 0)

        # Run the model and to get action probabilities and critic value
        action_logits, actions, value = model(state)
        # print(f"model outputs:\n{action_logits_t}\n{value}")
        # Sample next action from the action probability distribution
        # action = tf.random.categorical(action_logits_t, 1)[0, 0]
        # action_probs_t = tf.nn.softmax(action_logits_t)

        # Store critic values
        print(f"\nvalue:\n{value}")
        print(f"\naction logits step:\n{action_logits}")
        values.append(value)

        # Store log probability of the action chosen
        # action_probs = action_probs.write(t, action_probs_t[0, action])

        # Apply action to the environment to get next state and reward
        state, reward, done, info = env.step(actions)
        # state.set_shape(initial_state_shape)

        # Store reward
        rewards.append(reward)

        # Store log probability of the action chosen
        action = info['action']
        action_probs_step = info['masked_logits']
        action_prob = action_probs_step[0, action]
        action_probs.append(action_prob)

    action_probs = torch.stack(action_probs)
    values = torch.stack(values)
    rewards = torch.stack(rewards)

    return action_probs, values, rewards

def get_expected_return(
    rewards: tf.Tensor,
    gamma: float,
    standardize: bool = True) -> tf.Tensor:
    """Compute expected returns per timestep."""

    n = tf.shape(rewards)[0]
    returns = tf.TensorArray(dtype=tf.float32, size=n)

    # Start from the end of `rewards` and accumulate reward sums
    # into the `returns` array
    rewards = tf.cast(rewards[::-1], dtype=tf.float32)
    discounted_sum = tf.constant(0.0)
    discounted_sum_shape = discounted_sum.shape
    for i in tf.range(n):
        reward = rewards[i]
        discounted_sum = reward + gamma * discounted_sum
        discounted_sum.set_shape(discounted_sum_shape)
        returns = returns.write(i, discounted_sum)
    returns = returns.stack()[::-1]

    if standardize:
        returns = ((returns - tf.math.reduce_mean(returns)) /
                   (tf.math.reduce_std(returns) + eps))

    return returns

huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

def compute_loss(
        action_probs: torch.Tensor,
        values: torch.Tensor,
        returns: torch.Tensor) -> tf.Tensor:
    """Computes the combined Actor-Critic loss."""

    advantage = returns - values

    action_log_probs = torch.math.log(action_probs)
    actor_loss = -torch.sum(action_log_probs * advantage)

    critic_loss = huber_loss(values, returns)

    return actor_loss + critic_loss

optimizer = torch.optim.Adam(model.parameters())


def train_step(
        initial_state: torch.Tensor,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        gamma: float,
        max_steps_per_episode: int,
) -> torch.Tensor:
    """Runs a model training step."""

    # print(f"\ntrain_step initial_state:\n{initial_state}")
    # Run the model for one episode to collect training data
    action_probs, values, rewards = run_episode(initial_state, model, max_steps_per_episode)

    # Calculate the expected returns
    returns = get_expected_return(rewards, gamma)

    # Convert training data to appropriate TF tensor shapes
    # action_probs, values, returns = [tf.expand_dims(x, 1) for x in [action_probs, values, returns]]

    # Calculate the loss values to update our network
    loss = compute_loss(action_probs, values, returns)

    # compute gradients (grad)
    loss.backward()
    '''
    https://stackoverflow.com/questions/64856195/what-is-tape-based-autograd-in-pytorch
    with torch.no_grad():
        weights -= weights.grad * learning_rate
        biases -= biases.grad * learning_rate
        weights.grad.zero_()
        biases.grad.zero_()
    '''
    optimizer.zero_grad()

    # Apply the gradients to the model's parameters
    optimizer.step()
    optimizer.zero_grad()

    episode_reward = torch.sum(rewards)

    return episode_reward

def reshape(obs):
    stacked_obs = []
    if isinstance(obs, dict):
        for layer in obs.values():
            stacked_obs += layer.flatten()
        return torch.stack(stacked_obs, 0)
    else:
        print(f"Type {type(obs)} not implemented.")
        raise NotImplementedError

min_episodes_criterion = 100
max_episodes = 10000
max_steps_per_episode = 500
learning_rate = 0.001

# consecutive trials
reward_threshold = 1
running_reward = 0

# The discount factor for future rewards
gamma = 0.99

# Keep the last episodes reward
episodes_reward: collections.deque = collections.deque(maxlen=min_episodes_criterion)
print(__file__)


if __name__=='__main__':
    pass