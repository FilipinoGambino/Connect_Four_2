import gym
from gym import spaces
from kaggle_environments import make
from connectx.connectx_gym import GameResultReward
# from connectx.connectx_game.game import Game


class ConnectFour(gym.Env):
    metadata = {"render.modes": []}

    def __init__(self, configuration = None, seed = None):
        super(ConnectFour, self).__init__()
        n_cols = 7
        n_rows = 5
        self.obs_space = spaces.Box(0, 2, (n_cols, n_rows), dtype=int)
        self.action_space = spaces.Discrete(n_rows)
        self.reward_space = GameResultReward()

        # self.game_state = Game()
        self.game_state = make('connectx')

        if configuration is not None:
            self.configuration = configuration
        else:
            self.configuration = self.game_state.configuration
            # 2: warnings, 1: errors, 0: none
            self.configuration["loglevel"] = 0
        if seed is not None:
            self.seed = seed
        else:
            self.seed = 42
        self.done = False
        self.info = {}

    def step(self, action):
        self.game_state.step(action)
        rewards = self.reward_space.compute_rewards(self.game_state)
        return (
            self.game_state,
            rewards,
            self.game_state.done,
            False,
            self.game_state.info
        )

    def reset(self, seed=None, options=None):
        self.game_state = make('connectx')

    def render(self, **kwargs):
        print(self.game_state.state)

    def close(self):
        raise NotImplemented
