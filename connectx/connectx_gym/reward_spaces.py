from typing import NamedTuple, Tuple, Dict
from abc import ABC, abstractmethod
import logging

from kaggle_environments.core import Environment
import numpy as np

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
    def _compute_rewards(self, game_state: dict, done: bool) -> Tuple[float, float]:
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

    def compute_rewards(self, game_state: Environment) -> Tuple[Tuple[float, float], bool]:
        if self.early_stop:
            raise NotImplementedError  # done = done or should_early_stop(game_state)
        done = game_state.done
        return self._compute_rewards(game_state, done), done

    def _compute_rewards(self, game_state: Environment, done: bool) -> Tuple[float, float]:
        if not done:
            return 0., 0.
        rewards = (game_state.state[0]['reward'], game_state.state[1]['reward'])
        return rewards
