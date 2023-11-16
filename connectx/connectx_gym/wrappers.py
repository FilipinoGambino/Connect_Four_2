import gym
import torch
import numpy as np

from kaggle_environments.core import Environment
from typing import Dict, List, NoReturn, Optional, Union, Tuple

from . import ConnectFour
from .reward_spaces import BaseRewardSpace

class KaggleToGymWrapper(gym.Env):
    def __init__(self, env: Environment, act_space, obs_space):
        # super(KaggleToGymWrapper, self).__init__(env)
        self.env = env
        self.action_space = act_space
        self.observation_space = obs_space

    def step(self, action: int) -> Tuple[dict, Tuple[float, float], bool, bool, dict]:
        player1, player2 = self.env.step(action)

        rows = self.env.configuration['rows']
        cols = self.env.configuration['columns']
        obs = {
            "board": np.array(player1['observation']['board']).reshape((rows, cols)),
            "turn": player1['turn']
        }

        reward = (player1['reward'], player2['reward'])
        done = True if player1['status'] == 'DONE' else False
        info = {}

        return obs, _, done, info

    def reset(self, **kwargs):
        self.env.reset()

    def render(self, mode='human'):
        self.env.render(mode=mode)


class LoggingEnv(gym.Wrapper):
    def __init__(self, env: gym.Env, reward_space: BaseRewardSpace):
        super(LoggingEnv, self).__init__(env)
        self.reward_space = reward_space
        self.vals_peak = {}
        self.reward_sums = [0., 0.]
        self.actions_distributions = {

        }

    def info(self, info: Dict[str, np.ndarray], rewards: List[int]) -> Dict[str, np.ndarray]:
        return info

    def reset(self, **kwargs):
        obs, reward, done, info = super(LoggingEnv, self).reset(**kwargs)
        # self._reset_peak_vals()
        self.reward_sums = [0., 0.]
        self.actions_distributions = {
            key: 0. for key in self.actions_distributions.keys()
        }
        return obs, reward, done, self.info(info, reward)

    def step(self, action: Dict[str, np.ndarray]):
        obs, reward, done, truncate, info = super(LoggingEnv, self).step(action)
        return obs, reward, done, self.info(info, reward)


class RewardSpaceWrapper(gym.Wrapper):
    def __init__(self, env: ConnectFour, reward_space: BaseRewardSpace):
        super(RewardSpaceWrapper, self).__init__(env)
        self.reward_space = reward_space

    def _get_rewards_and_done(self) -> Tuple[Tuple[float, float], bool]:
        rewards, done = self.reward_space.compute_rewards_and_done(self.unwrapped.game_state, self.done)
        if self.unwrapped.done and not done:
            raise RuntimeError("Reward space did not return done, but the connectx engine is done.")
        self.unwrapped.done = done
        return rewards, done

    def reset(self, **kwargs):
        obs, _, _, info = super(RewardSpaceWrapper, self).reset(**kwargs)
        return (obs, *self._get_rewards_and_done(), info)

    def step(self, action):
        obs, _, _, info = super(RewardSpaceWrapper, self).step(action)
        return (obs, *self._get_rewards_and_done(), info)

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

    def reset(self, force: bool = False, **kwargs):
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

    def step(self, action: Dict[str, np.ndarray]):
        actions = [
            {key: val[i] for key, val in action.items()} for i in range(len(self.envs))
        ]
        self.last_outs = [env.step(a) for env, a in zip(self.envs, actions)]
        return VecEnv._vectorize_env_outs(self.last_outs)

    def render(self, idx: int, mode: str = "human", **kwargs):
        # noinspection PyArgumentList
        return self.envs[idx].render(mode, **kwargs)

    def close(self):
        return [env.close() for env in self.envs]

    def seed(self, seed: Optional[int] = None) -> list:
        if seed is not None:
            return [env.seed(seed + i) for i, env in enumerate(self.envs)]
        else:
            return [env.seed(seed) for i, env in enumerate(self.envs)]

    @property
    def unwrapped(self) -> List[gym.Env]:
        return [env.unwrapped for env in self.envs]

    @property
    def action_space(self) -> List[gym.spaces.Dict]:
        return [env.action_space for env in self.envs]

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

    def reset(self, **kwargs):
        return tuple([self._to_tensor(out) for out in super(PytorchEnv, self).reset(**kwargs)])

    def step(self, action: Dict[str, torch.Tensor]):
        action = {
            key: val.cpu().numpy() for key, val in action.items()
        }
        return tuple([self._to_tensor(out) for out in super(PytorchEnv, self).step(action)])

    def _to_tensor(self, x: Union[Dict, np.ndarray]) -> Dict[str, Union[Dict, torch.Tensor]]:
        if isinstance(x, dict):
            return {key: self._to_tensor(val) for key, val in x.items()}
        else:
            return torch.from_numpy(x).to(self.device, non_blocking=True)