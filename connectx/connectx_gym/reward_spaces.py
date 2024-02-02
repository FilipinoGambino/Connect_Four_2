from typing import NamedTuple, Tuple, Dict
from abc import ABC, abstractmethod
import logging
import math

from kaggle_environments.core import Environment
import numpy as np

from .connectx_env import ConnectFour
from ..utility_constants import BOARD_SIZE

class RewardSpec(NamedTuple):
    reward_min: float
    reward_max: float
    zero_sum: bool
    only_once: bool


class BaseRewardSpace(ABC):
    """
    A class used for defining a reward space and/or done state for either the full game or a sub-task
    """
    def __init__(self, **kwargs):
        if kwargs:
            logging.warning(f"RewardSpace received unexpected kwargs: {kwargs}")

    @staticmethod
    @abstractmethod
    def get_reward_spec() -> RewardSpec:
        pass

    @abstractmethod
    def compute_rewards(self, game_state: Environment) -> Tuple[Tuple[float, float], bool]:
        pass

    def get_info(self) -> Dict[str, np.ndarray]:
        return {}

# Full game reward spaces defined below

class FullGameRewardSpace(BaseRewardSpace):
    """
    A class used for defining a reward space for the full game.
    """
    def compute_rewards(self, game_state: Environment) -> Tuple[Tuple[float, float], bool]:
        pass

    @abstractmethod
    def _compute_rewards(self, game_state: dict) -> Tuple[float, float]:
        pass


class GameResultReward(FullGameRewardSpace):
    @staticmethod
    def get_reward_spec() -> RewardSpec:
        return RewardSpec(
            reward_min=-1.,
            reward_max=1.,
            zero_sum=True,
            only_once=True
        )

    def __init__(self, early_stop: bool = False, **kwargs):
        super(GameResultReward, self).__init__(**kwargs)
        self.early_stop = early_stop

    def compute_rewards(self, game_state: ConnectFour) -> Tuple[float, bool]:
        if self.early_stop:
            raise NotImplementedError  # done = done or should_early_stop(game_state)
        return self._compute_rewards(game_state), game_state.done

    def _compute_rewards(self, game_state: ConnectFour) -> float:
        if not game_state.done:
            return 0.
        return game_state.info['reward']

class LongGameReward(BaseRewardSpace):
    @staticmethod
    def get_reward_spec() -> RewardSpec:
        return RewardSpec(
            reward_min=-1.,
            reward_max=1.,
            zero_sum=False,
            only_once=False
        )
    def __init__(self, early_stop: bool = False, **kwargs):
        super(LongGameReward, self).__init__(**kwargs)
        self.early_stop = early_stop

    def compute_rewards(self, game_state: ConnectFour) -> Tuple[float, bool]:
        if self.early_stop:
            raise NotImplementedError  # done = done or should_early_stop(game_state)
        return self._compute_rewards(game_state), game_state.done

    def _compute_rewards(self, game_state: ConnectFour) -> float:
        return game_state.turn / math.prod(BOARD_SIZE)