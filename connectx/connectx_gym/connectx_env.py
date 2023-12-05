from kaggle_environments import make
from typing import Dict, List, Optional, Tuple
import gym
import numpy as np

from .act_spaces import BaseActSpace
from .obs_spaces import BaseObsSpace
from .reward_spaces import GameResultReward

from ..utility_constants import BOARD_SIZE

class ConnectFour(gym.Env):
    metadata = {'render_modes': ['human']}
    spec = None

    def __init__(
            self,
            act_space: BaseActSpace,
            obs_space: BaseObsSpace,
            seed: Optional[int] = 42,
    ):
        super(ConnectFour, self).__init__()
        self.env = make("connectx", debug=True)
        self.trainer = self.env.train([None, "negamax"])

        self.rows = self.env.configuration.rows
        self.columns = self.env.configuration.columns

        self.action_space = act_space
        self.obs_space = obs_space
        self.default_reward_space = GameResultReward()
        self.info = {}

    def reset(self, **kwargs):
        print('resetting')
        obs = self.trainer.reset()
        obs = np.array(obs['board']).reshape([self.rows, self.columns])
        self.info = []
        return obs

    def step(self, logits):
        action = self.process_actions(logits)
        obs, reward, done, _ = self.trainer.step(action)

        return obs, reward, done, self.info

    def process_actions(self, logits: np.ndarray) -> Tuple[List[List[str]], Dict[str, np.ndarray]]:
        step = self.env.state[0]['step']
        board = self.env.state[0]['observation']['board']
        obs = np.array(board).reshape(BOARD_SIZE)
        valid_actions = self.action_space.process_actions(
            logits,
            obs,
        )
        actions = int(np.argmax(valid_actions))

        self.info[step].append(dict(
            logits=logits,
            masked_actions=valid_actions,
            actions=actions
        ))
        return actions

    def render(self, **kwargs):
        self.env.render(**kwargs)
