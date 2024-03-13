from abc import ABC, abstractmethod
import logging
import math
import numpy as np
from scipy.signal import convolve2d
from typing import NamedTuple, Tuple, Dict

from ..connectx_game.game import Game
from ..utility_constants import BOARD_SIZE, IN_A_ROW, GAME_STATUS, VICTORY_KERNELS

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
        if game_state.turn > IN_A_ROW * 2:
            p1 = game_state.inactive_player
            p2 = game_state.active_player
            for idx,kernel in enumerate(VICTORY_KERNELS.values()):
                convolutions = convolve2d(game_state.board == p1.mark, kernel, mode="valid")
                if np.max(convolutions) == IN_A_ROW:
                    reward = 1.
                    done = True
                    return reward, done

            # Check every next move to see if p2 can win
            valid_columns = [col for col in range(BOARD_SIZE[1]) if not game_state.board[:, col].all()]
            for col in valid_columns:
                board = game_state.board.copy()
                row = game_state.get_lowest_available_row(col)
                board[row,col] = p2.mark
                for kernel in VICTORY_KERNELS.values():
                    convolutions = convolve2d(board == p2.mark, kernel, mode="valid")
                    if np.max(convolutions) == IN_A_ROW:
                        reward = -1.
                        done = False
                        return reward, done

        reward = 0
        done = False
        return reward, done


class DiagonalEmphasisReward(BaseRewardSpace):
    @staticmethod
    def get_reward_spec() -> RewardSpec:
        return RewardSpec(
            reward_min=-1.,
            reward_max=1.,
            zero_sum=False,
            only_once=False
        )
    def __init__(self, **kwargs):
        super(DiagonalEmphasisReward, self).__init__(**kwargs)
        self.board_size = math.prod(BOARD_SIZE)

    def compute_rewards(self, game_state: Game) -> Tuple[float, bool]:
        p1 = game_state.inactive_player # The player that just performed an action
        p2 = game_state.active_player

        reward, done = self.compute_player_reward(game_state, p1.mark, p2.mark)
        p1.reward += reward

        # This is double dipping
        # reward, done = self.compute_player_reward(game_state, p2.mark, p1.mark)
        # p2.reward += reward

        return p1.reward, done

    def compute_player_reward(self, game_state: Game, mark1: int, mark2: int):
        cells_to_check = dict(horizontal=[(1,0),(2,0),(3,0)],
                              vertical=[(0,1),(0,2),(0,3)],
                              diagonal_identity=[(1,1),(2,2),(3,3)],
                              diagonal_flipped=[(-1,1),(-2,2),(-3,3)])
        max_row = BOARD_SIZE[0]
        max_col = BOARD_SIZE[1]

        reward = 0
        done = False
        for mark_row, mark_col in np.argwhere(game_state.board == mark1):
            for key, pairs in cells_to_check.items():
                count = 1
                for check_row, check_col in pairs:
                    row, col = check_row + mark_row, check_col + mark_col
                    if not 0 <= row < max_row or not 0 <= col < max_col or game_state.board[row][col] == mark2:
                        break

                    if game_state.board[row][col] == mark1:
                        count += 1
                    elif game_state.board[row][col] == 0:
                        continue

                    if key.startswith('diagonal'):
                        if count >= 4:
                            r = 1.
                            done = True
                        elif count == 3:
                            r = .5
                        elif count == 2:
                            r = 1/42
                        else:
                            r = 0
                    else:
                        if count >= 4:
                            r = -1.
                            done = True
                        elif count == 3:
                            r = -.3
                        elif count == 2:
                            r = 1/42
                        else:
                            r = 0
                    reward = max(r, reward)

        return reward, done

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