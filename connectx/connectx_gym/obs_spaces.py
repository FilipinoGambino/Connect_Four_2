from abc import ABC, abstractmethod
import gym
import numpy as np
from typing import Dict, List, Tuple

from ..connectx_game.game_constants import Constants

BOARD_SIZE = Constants.BOARD_SIZE

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

# class FixedShapeObs(BaseObsSpace, ABC):
#     pass

class FixedShapeContinuousObs(BaseObsSpace, ABC):
    def get_obs_spec(
            self,
            board_dims: Tuple[int, int] = BOARD_SIZE
    ) -> gym.spaces.Dict:
        x = board_dims[0]
        y = board_dims[1]
        return gym.spaces.Dict({
            "board": gym.spaces.MultiBinary(x * y),
            "empties": gym.spaces.MultiBinary(x * y),
            "p1_cells": gym.spaces.MultiBinary(x * y),
            "p2_cells": gym.spaces.MultiBinary(x * y),
            "turn": gym.spaces.Box(low=0, high=1),
        })

    def wrap_env(self, env) -> gym.Wrapper:
        return _FixedShapeContinuousObsWrapper(env)


class _FixedShapeContinuousObsWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super(_FixedShapeContinuousObsWrapper, self).__init__(env)
        self._empty_obs = {}
        for key, spec in FixedShapeContinuousObs().get_obs_spec().spaces.items():
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

    def observation(self, observation: np.ndarray) -> Dict[str, np.ndarray]:
        rows, cols = BOARD_SIZE
        board = np.array(observation['board']).reshape((rows, cols))
        p1_mark = observation['mark']
        p2_mark = (p1_mark + 1) % 2

        obs = {
            "board": board,
            "empty_cells": np.where(board == 0, 1, 0),
            "p1_cells": np.where(board == p1_mark, 1, 0),
            "p2_cells": np.where(board == p2_mark, 1, 0),
            "turn": observation['step'],
        }
        return obs