from abc import ABC, abstractmethod
import logging
import math
import numpy as np
from scipy.signal import convolve2d
from typing import NamedTuple, Tuple, Dict

from ..connectx_game.game import Game
from ..utility_constants import BOARD_SIZE, IN_A_ROW, GAME_STATUS

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
    def compute_rewards(self, game_state: Game) -> Tuple[Tuple[float, float], bool]:
        pass

    @abstractmethod
    def _compute_rewards(self, game_state: Game) -> Tuple[float, float]:
        pass


class GameResultReward(FullGameRewardSpace):
    @staticmethod
    def get_reward_spec() -> RewardSpec:
        return RewardSpec(
            reward_min=-1.,
            reward_max=1.,
            zero_sum=False,
            only_once=True
        )

    def __init__(self, **kwargs):
        super(GameResultReward, self).__init__(**kwargs)

    def compute_rewards(self, game_state: Game) -> Tuple[float, bool]:
        p1,p2 = game_state.players
        if p1.active(game_state.turn - 1):
            result = game_state.should_win(p1,p2)
        elif p2.active(game_state.turn - 1):
            result = game_state.should_win(p1, p2)

        if result == GAME_STATUS['ACTIVE_PLAYER_WINS']:
            reward = 1.
            done = True
        elif result == GAME_STATUS['UNDEFENDED_POSITION']:
            reward = -1.
            done = False
        elif result == GAME_STATUS['NO_WINNING_MOVE']:
            reward = 1/42
            done = False
        return reward, done

    def _compute_rewards(self, game_state: Game) -> float:
        raise NotImplementedError

class LongGameReward(BaseRewardSpace):
    @staticmethod
    def get_reward_spec() -> RewardSpec:
        return RewardSpec(
            reward_min=-1.,
            reward_max=1.,
            zero_sum=False,
            only_once=False
        )
    def __init__(self, **kwargs):
        super(LongGameReward, self).__init__(**kwargs)
        self.board_size = math.prod(BOARD_SIZE)

    def compute_rewards(self, game_state: Game) -> Tuple[float, bool]:
        done = game_state.game_end() != "No Winner"
        return self._compute_rewards(game_state), done

    def _compute_rewards(self, game_state: Game) -> float:
        return game_state.turn / self.board_size


class MoreInARowReward(BaseRewardSpace):
    @staticmethod
    def get_reward_spec() -> RewardSpec:
        return RewardSpec(
            reward_min=-1.,
            reward_max=1.,
            zero_sum=False,
            only_once=False
        )
    def __init__(self, **kwargs):
        super(MoreInARowReward, self).__init__(**kwargs)
        self.search_length = IN_A_ROW - 1

        horizontal_kernel = np.ones([1, self.search_length], dtype=np.uint8)
        vertical_kernel = np.transpose(horizontal_kernel)
        diag1_kernel = np.eye(self.search_length, dtype=np.uint8)
        diag2_kernel = np.fliplr(diag1_kernel)

        self.reward_kernels = [
            horizontal_kernel,
            vertical_kernel,
            diag1_kernel,
            diag2_kernel,
        ]

        self.base_reward = -1/math.prod(BOARD_SIZE)

    def compute_rewards(self, game_state: Game) -> Tuple[float, bool]:
        return self._compute_rewards(game_state), game_state.done

    def _compute_rewards(self, game_state: Game) -> float:
        player = game_state.active_player
        count = 1
        for kernel in self.reward_kernels:
            conv = convolve2d(game_state.board == player.mark, kernel, mode="valid")
            count += np.count_nonzero(conv == self.search_length)
        return math.log10(count)