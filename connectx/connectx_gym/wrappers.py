import copy
import gym
from logging import getLogger
import numpy as np
import torch
from typing import Dict, List, Union, Tuple

from .reward_spaces import BaseRewardSpace

logger = getLogger(__name__)

class LoggingEnv(gym.Wrapper):
    def __init__(self, env: gym.Env, reward_space: BaseRewardSpace):
        super(LoggingEnv, self).__init__(env)
        self.reward_space = reward_space
        self.vals_peak = {}
        # self.p1_cumulative_reward = 0.
        # self.p2_cumulative_reward = 0.

    def info(self, info: Dict[str, np.ndarray], rewards: List[int]) -> Dict[str, np.ndarray]:
        info = copy.copy(info)
        logs = dict()

        # self.p1_cumulative_reward += rewards[0]
        # self.p2_cumulative_reward += rewards[1]
        # logs["total_p1_rewards"] = self.p1_cumulative_reward
        # logs["total_p2_rewards"] = self.p2_cumulative_reward

        info.update({f"LOGGING_{key}": np.array(val, dtype=np.float32) for key, val in logs.items()})
        # Add any additional info from the reward space
        info.update(self.reward_space.get_info())
        return info

    def reset(self, **kwargs):
        obs, rewards, done, info = super(LoggingEnv, self).reset(**kwargs)
        # self.p1_cumulative_reward = 0.
        # self.p2_cumulative_reward = 0.
        return obs, rewards, done, self.info(info, rewards)

    def step(self, action: Dict[str, np.ndarray]):
        obs, rewards, done, info = super(LoggingEnv, self).step(action)
        return obs, rewards, done, self.info(info, rewards)


class RewardSpaceWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, reward_space: BaseRewardSpace):
        super(RewardSpaceWrapper, self).__init__(env)
        self.env = env
        self.reward_space = reward_space

    def _get_rewards_and_done(self, game_state) -> Tuple[Tuple[float, float], bool]:
        rewards, done = self.reward_space.compute_rewards(game_state)
        return rewards, done

    def reset(self, **kwargs):
        obs, _, _, info = super(RewardSpaceWrapper, self).reset(**kwargs)
        return obs, *self._get_rewards_and_done(obs), info

    def step(self, action):
        obs, _, _, info = super(RewardSpaceWrapper, self).step(action)
        return obs, *self._get_rewards_and_done(obs), info

class VecEnv(gym.Env):
    def __init__(self, envs: List[gym.Env]):
        self.envs = envs
        self.last_outs = [() for _ in range(len(self.envs))]

    @staticmethod
    def _stack_dict(x: List[Union[Dict, np.ndarray]]) -> Union[Dict, np.ndarray]:
        if isinstance(x[0], dict):
            return {key: VecEnv._stack_dict([i[key] for i in x]) for key in x[0].keys()}
        else:
            return np.stack([arr for arr in x], axis=0)

    @staticmethod
    def _vectorize_env_outs(env_outs: List[Tuple]) -> Tuple:
        obs_list, reward_list, done_list, info_list = zip(*env_outs)
        obs_stacked = VecEnv._stack_dict(obs_list)
        reward_stacked = np.array(reward_list)
        done_stacked = np.array(done_list)
        info_stacked = VecEnv._stack_dict(info_list)
        return obs_stacked, reward_stacked, done_stacked, info_stacked


    def reset(self, force: bool = True, **kwargs):
        if force:
            # noinspection PyArgumentList
            self.last_outs = [env.reset(**kwargs) for env in self.envs]
            return VecEnv._vectorize_env_outs(self.last_outs)
        for i, env in enumerate(self.envs):
            # Check if env finished
            if self.last_outs[i][2]:
                # noinspection PyArgumentList
                self.last_outs[i] = env.reset()
        return VecEnv._vectorize_env_outs(self.last_outs)

    def step(self, actions: List[int]):
        self.last_outs = [env.step(a) for env, a in zip(self.envs, actions)]
        return VecEnv._vectorize_env_outs(self.last_outs)

    def render(self, mode: str = "human", **kwargs):
        # noinspection PyArgumentList
        return self.envs[0].render(**kwargs)

    def close(self):
        return [env.close() for env in self.envs]

    @property
    def unwrapped(self) -> List[gym.Env]:
        return [env.unwrapped for env in self.envs]

    @property
    def action_space(self) -> List[gym.spaces.Dict]:
        return [env.action_space for env in self.envs]

    @property
    def action_space_n(self):
        action_spaces = self.action_space
        return [act.n for act in action_spaces]

    @property
    def observation_space(self) -> List[gym.spaces.Dict]:
        return [env.observation_space for env in self.envs]

    @property
    def metadata(self) -> List[Dict]:
        return [env.metadata for env in self.envs]


class PytorchEnv(gym.Wrapper):
    def __init__(self, env: Union[gym.Env, VecEnv], device: torch.device = torch.device("cpu")):
        super(PytorchEnv, self).__init__(env)
        self.device = device

    def reset(self, **kwargs) -> Tuple[Dict, List, bool, List]:
        return tuple([self._to_tensor(out) for out in super(PytorchEnv, self).reset(**kwargs)])

    def step(self, actions: List[torch.Tensor]):
        action = [int(act) for act in actions]
        return tuple([self._to_tensor(out) for out in super(PytorchEnv, self).step(action)])

    def _to_tensor(self, x: Union[Dict, np.ndarray]) -> Dict[str, Union[Dict, torch.Tensor]]:
        if isinstance(x, dict):
            return {key: self._to_tensor(val) for key, val in x.items()}
        elif isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(device=self.device)


class DictEnv(gym.Wrapper):
    @staticmethod
    def _dict_env_out(env_out: tuple) -> dict:
        obs, reward, done, info = env_out
        return dict(
            obs=obs,
            reward=reward,
            done=done,
            info=info
        )

    def reset(self, **kwargs):
        return DictEnv._dict_env_out(super(DictEnv, self).reset(**kwargs))

    def step(self, action):
        return DictEnv._dict_env_out(super(DictEnv, self).step(action))