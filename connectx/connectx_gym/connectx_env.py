from kaggle_environments import make
import gym
from gym import spaces

from connectx.connectx_game.game import Game
from .reward_spaces import GameResultReward

class ConnectFour(gym.Env):
    metadata = {"render.modes": []}

    def __init__(self):
        super(ConnectFour, self).__init__()
        n_cols = 7
        n_rows = 5
        self.obs_space = spaces.Box(0, 2, (n_cols, n_rows), dtype=int)
        self.action_space = spaces.Discrete(n_rows)
        self.default_reward_space = GameResultReward()

        self.game_state = Game()
        configuration, seed = None, None
        if configuration is not None:
            self.configuration = configuration
        else:
            self.configuration = make("connectx").configuration
            # 2: warnings, 1: errors, 0: none
            self.configuration["loglevel"] = 0
        if seed is not None:
            self.seed(seed)
        elif "seed" not in self.configuration:
            self.seed()
        self.done = False
        self.info = {}

    def step(self, action):
        self.game_state.step(action)

    def reset(self, seed=None, options=None):
        self.game_state.reset()

    def render(self, **kwargs):
        print(self.game_state.board)

    def close(self):
        raise NotImplemented
