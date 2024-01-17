from kaggle_environments import make
from typing import Dict, List, Optional, Tuple
import gym
import numpy as np
from scipy.special import softmax

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
            player_id: int,
            seed: Optional[int] = 42,
    ):
        super(ConnectFour, self).__init__()
        self.env = make("connectx", debug=True)
        players = ["negamax", "negamax"]
        players[player_id-1] = None
        self.trainer = self.env.train(players)

        self.rows = self.env.configuration.rows
        self.columns = self.env.configuration.columns

        self.action_space = act_space
        self.obs_space = obs_space
        self.default_reward_space = GameResultReward()
        self.info = []

    def reset(self, **kwargs):
        obs = self.trainer.reset()
        self.info = []
        reward = 0
        done = False
        return obs, reward, done, self.info

    def step(self, action):
        obs, reward, done, _ = self.trainer.step(action)
        return obs, reward, done, self.info

    def process_actions(self, logits: np.ndarray) -> Tuple[List[List[str]], Dict[str, np.ndarray]]:
        step = self.env.state[0]['observation']['step']
        board = self.env.state[0]['observation']['board']
        obs = np.array(board).reshape(BOARD_SIZE)
        print(f"\naction logits:\n{logits}")
        valid_action_logits = self.action_space.process_actions(
            logits,
            obs,
        )
        print(f"\nvalid actions:\n{valid_action_logits}")
        valid_action_probs = softmax(valid_action_logits)
        action = np.random.choice(BOARD_SIZE[1], p=valid_action_probs)

        self.info.append(
            dict(
                logits=logits,
                masked_logits=valid_action_logits,
                masked_probs=valid_action_probs,
                action=action,
                step=step,
            )
        )
        return action

    def render(self, **kwargs):
        self.env.render(**kwargs)
