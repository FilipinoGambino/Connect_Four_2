from abc import ABC, abstractmethod
import logging
import math
import numpy as np
from scipy.signal import convolve2d
from typing import NamedTuple, Tuple, Dict

from ..connectx_game.game import Game
from ..utility_constants import IN_A_ROW, PLAYER_MARKS, VICTORY_KERNELS


logging.basicConfig(
    format=(
        "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
    ),
    level=0,
)


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
    def compute_rewards(self, game_state: Game) -> Tuple[Tuple[float, float], bool]:
        pass

    def get_info(self) -> Dict[str, np.ndarray]:
        return {}

# Full game reward spaces defined below

class FullGameRewardSpace(BaseRewardSpace):
    """
    A class used for defining a reward space for the full game.
    """

    @abstractmethod
    def compute_rewards(self, game_state: Game) -> Tuple[Tuple[float, float], bool]:
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

    def __init__(self, **kwargs):
        super(GameResultReward, self).__init__(**kwargs)

    def compute_rewards(self, game_state: Game) -> Tuple[float, bool]:
        '''
        The inactive player (p1) is the one that just completed an action so we need their reward
        :param game_state:
        :return: reward for the completed action, whether or not the game state is done
        '''
        rewards = [0,0]
        done = False
        for player_mark in PLAYER_MARKS:
            for kernel_name, kernel in VICTORY_KERNELS.items():
                conv = convolve2d(game_state.board == player_mark, kernel, mode="valid")
                if np.any(conv == IN_A_ROW):
                    rewards = [-1, -1]
                    rewards[player_mark - 1] = 1
                    done = True
        if game_state.turn == game_state.max_turns:
            done = True
        reward = rewards[game_state.last_player_mark - 1]
        return [reward], done