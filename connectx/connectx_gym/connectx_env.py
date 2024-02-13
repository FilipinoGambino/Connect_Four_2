from kaggle_environments import make
from typing import Any, Dict, List, Optional, Tuple
import gym
import math
import numpy as np

from .act_spaces import BaseActSpace
from .obs_spaces import BaseObsSpace
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

        if configuration is not None:
            self.configuration = configuration
        else:
            self.configuration = make("connectx").configuration

        self.game_state = Game(self.configuration)
        self.reward = 0.
        self.done = False
        self.info = dict()

    def reset(self, **kwargs):
        logging.info('Env Resetting')
        config = kwargs.get('configuration', self.configuration)
        self.game_state = Game(config)
        self.done = False
        self.info['available_actions_mask'] = self.action_space.get_available_actions_mask(self.board)
        return self.game_state, self.reward, self.done, self.info

    def step(self, action):
        logging.info('Env Stepping')
        self.game_state.step(action)
        return self.game_state, self.reward, self.done, self.info

    def update(self, observation):
        logging.info('Env Updating')
        self.game_state.update(observation)
        self.info['available_actions_mask'] = self.action_space.get_available_actions_mask(self.board)

    def render(self, **kwargs):
        raise NotImplementedError

    @property
    def turn(self):
        return self.game_state.turn

    @property
    def board(self):
        return self.game_state.board

    @property
    def mark(self):
        return self.game_state.mark

    @property
    def max_turns(self):
        return self.game_state.max_turns