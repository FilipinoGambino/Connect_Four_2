from typing import NamedTuple, Tuple, Dict
from abc import ABC, abstractmethod
import logging

import numpy as np

from kaggle_environments.core import Environment
# from .connectx_env.game_objects import Player

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
    def compute_rewards_and_done(self, game_state: Environment, done: bool) -> Tuple[Tuple[float, float], bool]:
        pass

    def get_info(self) -> Dict[str, np.ndarray]:
        return {}

# Full game reward spaces defined below

class FullGameRewardSpace(BaseRewardSpace):
    """
    A class used for defining a reward space for the full game.
    """
    def compute_rewards_and_done(self, game_state: Environment, done: bool) -> Tuple[Tuple[float, float], bool]:
        return self.compute_rewards(game_state), done

    @abstractmethod
    def compute_rewards(self, game_state: Environment) -> Tuple[float, float]:
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

    def compute_rewards_and_done(self, game_state: Environment, done: bool) -> Tuple[Tuple[float, float], bool]:
        # if self.early_stop:
        #     done = done or should_early_stop(game_state)
        return self.compute_rewards(game_state), done

    def compute_rewards(self, game_state: Environment) -> Tuple[float, float]:
        if not game_state.done:
            return 0., 0.
        rewards = [player['reward'] for player in game_state]
        # rewards = [int(GameResultReward.compute_player_reward(p)) for p in game_state.players]
        return tuple(rewards)

    # @staticmethod
    # def compute_player_reward(player: Player):
    #     if player.victory:
    #         return GameResultReward.get_reward_spec().reward_max
    #     else:
    #         return GameResultReward.get_reward_spec().reward_min