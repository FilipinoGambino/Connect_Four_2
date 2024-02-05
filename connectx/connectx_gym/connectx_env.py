from kaggle_environments import make
from typing import Dict, List, Optional, Tuple
import gym
import math
import numpy as np

from .act_spaces import BaseActSpace
from .obs_spaces import BaseObsSpace

from ..utility_constants import BOARD_SIZE


class ConnectFour(gym.Env):
    metadata = {'render_modes': ['human']}
    spec = None

    def __init__(
            self,
            act_space: BaseActSpace,
            obs_space: BaseObsSpace,
            player_id: int,
            adversary: str,
    ):
        super(ConnectFour, self).__init__()
        self.env = make("connectx", debug=True)
        self.player_id = player_id
        self.mark = player_id + 1
        players = [adversary, adversary]
        players[player_id] = None
        self.trainer = self.env.train(players)

        self.rows = self.env.configuration.rows
        self.columns = self.env.configuration.columns

        self.game_reward = 0.
        self.action_space = act_space
        self.obs_space = obs_space
        self.info = dict()

    def reset(self, **kwargs):
        obs = self.trainer.reset()
        self.game_reward = 0.
        done = False
        self._update(obs)

        return obs, self.game_reward, done, self.info

    def step(self, action):
        obs, self.game_reward, done, _ = self.trainer.step(action)
        self._update(obs, action)
        return obs, self.game_reward, done, self.info

    def _update(self, obs, action=-1):
        obs_array = np.array(obs['board']).reshape((1,*BOARD_SIZE))

        self.info = dict(
            action=action,
            reward=self.game_reward,
            available_actions_mask=self.action_space.get_available_actions_mask(obs_array),
        )

    def render(self, **kwargs):
        self.env.render(**kwargs)


    @property
    def turn(self):
        return self.env.state[0]['observation']['step']

    @property
    def done(self):
        return self.env.done

    @property
    def board(self):
        return np.array(self.env.state[0]['observation']['board']).reshape(BOARD_SIZE)

    @property
    def configuration(self):
        return self.env.configuration

    @property
    def steps(self):
        return self.env.steps