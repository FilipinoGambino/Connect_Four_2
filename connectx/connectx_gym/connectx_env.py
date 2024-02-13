from kaggle_environments import make
from typing import Any, Dict, List, Optional, Tuple
import gym
import math
import numpy as np

from .act_spaces import BaseActSpace
from .obs_spaces import BaseObsSpace
from .reward_spaces import GameResultReward
from ..connectx_game.game import Game

import logging

logging.basicConfig(
    format=(
        "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
    ),
    level=0,
)


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
        self.action_space = act_space
        self.obs_space = obs_space
        self.default_reward_space = GameResultReward()

        if configuration is not None:
            self.configuration = configuration
        else:
            self.configuration = make("connectx").configuration

        self.game_state = Game()

    def reset(self, **kwargs):
        assert kwargs.get('configuration', None) == self.configuration
        self.game_state.update(kwargs['observation'])
        return self.get_obs_reward_done_info()

    def step(self, action):
        self.game_state.step(action)
        return self.get_obs_reward_done_info()

    def info(self, rewards):
        return dict(
            available_actions_mask=self.action_space.get_available_actions_mask(self.board),
            rewards=rewards
        )

    def get_obs_reward_done_info(self):
        rewards, done = self.default_reward_space.compute_rewards(
            game_state=self.game_state,
            player=self.active_player-1
        )
        return self.game_state, rewards, done, self.info(rewards)

    def render(self, **kwargs):
        raise NotImplementedError

    @property
    def board(self):
        return self.game_state.board

    @property
    def max_turns(self):
        if isinstance(self.board, np.ndarray):
            return self.board.size
        elif isinstance(self.board, list):
            return len(self.board)
        else:
            raise NotImplementedError

    @property
    def turn(self):
        return self.game_state.turn

    @property
    def active_player(self):
        return self.game_state.current_player

    @property
    def inactive_player(self):
        return max(1, (self.active_player + 1) % 3)