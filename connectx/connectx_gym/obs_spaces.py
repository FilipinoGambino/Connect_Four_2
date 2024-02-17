from abc import ABC, abstractmethod
import gym
import numpy as np
from typing import Dict, List, Tuple

from ..connectx_game.game import Game
from ..utility_constants import BOARD_SIZE

class BaseObsSpace(ABC):
    # NB: Avoid using Discrete() space, as it returns a shape of ()
    # NB: "_COUNT" keys indicate that the value is used to scale the embedding of another value
    @abstractmethod
    def get_obs_spec(
            self,
            board_dims: Tuple[int, int] = BOARD_SIZE
    ) -> gym.spaces.Dict:
        pass

    @abstractmethod
    def wrap_env(self, env) -> gym.Wrapper:
        pass

class MultiObs(BaseObsSpace):
    def __init__(self, named_obs_spaces: Dict[str, BaseObsSpace], *args, **kwargs):
        super(MultiObs, self).__init__(*args, **kwargs)
        self.named_obs_spaces = named_obs_spaces

    def get_obs_spec(
            self,
            board_dims: Tuple[int, int] = BOARD_SIZE
    ) -> gym.spaces.Dict:
        return gym.spaces.Dict({
            name + key: val
            for name, obs_space in self.named_obs_spaces.items()
            for key, val in obs_space.get_obs_spec(board_dims).spaces.items()
        })

    def wrap_env(self, env) -> gym.Wrapper:
        return _MultiObsWrapper(env, self.named_obs_spaces)


class _MultiObsWrapper(gym.Wrapper):
    def __init__(self, env, named_obs_spaces: Dict[str, BaseObsSpace]):
        super(_MultiObsWrapper, self).__init__(env)
        self.named_obs_space_wrappers = {key: val.wrap_env(env) for key, val in named_obs_spaces.items()}

    def reset(self, **kwargs):
        observation, reward, done, info = self.env.reset(**kwargs)
        return self.observation(observation), reward, done, info

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return self.observation(observation), reward, done, info

    def observation(self, observation: Game) -> Dict[str, np.ndarray]:
        return {
            name + key: val
            for name, obs_space in self.named_obs_space_wrappers.items()
            for key, val in obs_space.observation(observation).items()
        }

class BasicObsSpace(BaseObsSpace, ABC):
    def get_obs_spec(
            self,
            board_dims: Tuple[int, int] = BOARD_SIZE
    ) -> gym.spaces.Dict:
        x = board_dims[0]
        y = board_dims[1]
        return gym.spaces.Dict({
            "filled_cells": gym.spaces.MultiBinary((x, y)),
            "empty_cells": gym.spaces.MultiBinary((x, y)),
            "p1_cells": gym.spaces.MultiBinary((x, y)),
            "p2_cells": gym.spaces.MultiBinary((x, y)),
            "turn": gym.spaces.Box(low=0, high=1, shape=[1, 1]),
        })

    def wrap_env(self, env) -> gym.Wrapper:
        return _BasicObsSpaceWrapper(env)


class _BasicObsSpaceWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super(_BasicObsSpaceWrapper, self).__init__(env)
        self._empty_obs = {}
        for key, spec in BasicObsSpace().get_obs_spec().spaces.items():
            if isinstance(spec, gym.spaces.MultiBinary):
                self._empty_obs[key] = np.zeros(spec.shape, dtype=np.int64)
            elif isinstance(spec, gym.spaces.Box):
                self._empty_obs[key] = np.zeros(spec.shape, dtype=np.float32) + spec.low
            else:
                raise NotImplementedError(f"{type(spec)} is not an accepted observation space.")

    def reset(self, **kwargs):
        observation, reward, done, info = self.env.reset(**kwargs)
        return self.observation(observation), reward, done, info

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return self.observation(observation), reward, done, info

    def observation(self, observation: Game) -> Dict[str, np.ndarray]:
        board = observation.board
        p1 = observation.active_player
        p2 = observation.inactive_player
        norm_turn = observation.turn / observation.board.size
        obs = {
            "filled_cells": np.where(board != 0, 1, 0),
            "empty_cells": np.where(board == 0, 1, 0),
            "p1_cells": np.where(board == p1.mark, 1, 0),
            "p2_cells": np.where(board == p2.mark, 1, 0),
            "turn": np.full(shape=(1,1), fill_value=norm_turn, dtype=np.float32),
        }
        return obs