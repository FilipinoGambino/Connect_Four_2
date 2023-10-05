from typing import NamedTuple, Tuple, Dict
from abc import ABC, abstractmethod
import logging

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
    def compute_rewards_and_done(self, game_state: GameState, done: bool) -> Tuple[Tuple[float, float], bool]:
        pass

    def get_info(self) -> Dict[str, np.ndarray]:
        return {}

# Full game reward spaces defined below

class FullGameRewardSpace(BaseRewardSpace):
    """
    A class used for defining a reward space for the full game.
    """
    def compute_rewards_and_done(self, game_state: GameState, done: bool) -> Tuple[Tuple[float, float], bool]:
        return self.compute_rewards(game_state, done), done

    @abstractmethod
    def compute_rewards(self, game_state: GameState, done: bool) -> Tuple[float, float]:
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

    def compute_rewards_and_done(self, game_state: GameState, done: bool) -> Tuple[Tuple[float, float], bool]:
        if self.early_stop:
            done = done or should_early_stop(game_state)
        return self.compute_rewards(game_state, done), done

    def compute_rewards(self, game_state: GameState, done: bool) -> Tuple[float, float]:
        if not done:
            return 0., 0.

        # reward here is defined as the total lichen per player with robot count as a tie-breaking mechanism
        rewards = [int(GameResultReward.compute_player_reward(p)) for p in game_state.players]
        # i.e. reward = (1st, 2nd) => * 2 = (2, 4) => -3 = (-1, 1) or for ties reward = (2nd, 2nd) => (1, 1)
        rewards = (rankdata(rewards)) * 2. - 3.
        return tuple(rewards)

    @staticmethod
    def compute_player_reward(player: Player):
        lichen_count = player.lichen_count
        robot_count = player.robot_count
        return lichen_count * 10_000 + robot_count