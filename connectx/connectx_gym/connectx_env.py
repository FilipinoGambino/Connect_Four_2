from typing import Any, Dict, List, Optional, Tuple
import gym
import math
import numpy as np
from scipy.signal import convolve2d

from .act_spaces import BaseActSpace
from .obs_spaces import BaseObsSpace
from .reward_spaces import GameResultReward
from ..connectx_game.game import Game

from ..utility_constants import BOARD_SIZE, IN_A_ROW, PLAYER_MARKS, VICTORY_KERNELS

ROWS, COLUMNS = BOARD_SIZE


# class ConnectFour(gym.Env):
#     metadata = {'render_modes': ['human']}
#     spec = None
#
#     def __init__(
#             self,
#             act_space: BaseActSpace,
#             obs_space: BaseObsSpace,
#             configuration: Optional[Dict[str, Any]] = None,
#             autoplay: bool = True
#     ):
#         super(ConnectFour, self).__init__()
#         self.action_space = act_space
#         self.obs_space = obs_space
#         self.default_reward_space = GameResultReward()
#
#         if configuration is not None:
#             self.configuration = configuration
#         else:
#             self.configuration = dict(rows=6, columns=7, inarow=4)
#
#         self.autoplay = autoplay
#
#         self.game_state = Game(self.configuration)
#
#     def reset(self, **kwargs):
#         self.game_state = Game(self.configuration)
#         return self.get_obs_reward_done_info()
#
#     def step(self, action):
#         if self.autoplay:
#             self.game_state.step(action)
#         return self.get_obs_reward_done_info()
#
#     def manual_step(self, obs):
#         self.game_state.update(obs)
#
#     def info(self, rewards):
#         return dict(
#             available_actions_mask=self.action_space.get_available_actions_mask(self.board),
#             rewards=rewards
#         )
#
#     def get_obs_reward_done_info(self):
#         rewards, done = self.default_reward_space.compute_rewards(game_state=self.game_state)
#         return self.game_state, rewards, done, self.info(rewards)
#
#     def render(self, **kwargs):
#         raise NotImplementedError
#
#     @property
#     def board(self):
#         return self.game_state.board
#
#     @property
#     def turn(self):
#         return self.game_state.turn

class ConnectFour(gym.Env):
    metadata = {'render_modes': ['human']}
    spec = None

    def __init__(
            self,
            act_space: BaseActSpace,
            obs_space: BaseObsSpace,
            configuration: Optional[Dict[str, Any]] = None,
    ):
        super(ConnectFour, self).__init__()

        self.act_space = act_space
        self.obs_space = obs_space

        if configuration is not None:
            self.configuration = configuration
        else:
            self.configuration = dict(rows=ROWS, columns=COLUMNS, inarow=IN_A_ROW)

        self.game_state = Game(self.configuration)

        self.rewards = [0, 0]
        self.done = False

    def reset(self, **kwargs):
        self.game_state = Game(self.configuration)
        self.rewards = [0, 0]
        self.done = False
        return self.get_obs_reward_done_info()

    def step(self, action):
        self.game_state.step(action)
        return self.get_obs_reward_done_info()

    def check_game_over(self):
        for player_mark in PLAYER_MARKS:
            for kernel_name, kernel in VICTORY_KERNELS.items():
                conv = convolve2d(self.board == player_mark, kernel, mode="valid")
                if np.any(conv == IN_A_ROW):
                    self.rewards = [-1, -1]
                    self.rewards[player_mark - 1] = 1
                    self.done = True
        if self.game_state.turn == self.game_state.max_turns:
            self.done = True

    def info(self):
        return dict(
            available_actions_mask=self.act_space.get_available_actions_mask(self.board),
            rewards=self.rewards,
            turn=self.game_state.turn
        )

    def get_obs_reward_done_info(self):
        self.check_game_over()
        return self.game_state, self.rewards, self.done, self.info()

    def render(self, **kwargs):
        raise NotImplementedError

    @property
    def board(self):
        return self.game_state.board

    @property
    def winner(self):
        if max(self.rewards) == 0:
            return None
        else:
            return np.argmax(self.rewards) + 1