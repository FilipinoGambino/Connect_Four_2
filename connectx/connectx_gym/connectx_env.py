from kaggle_environments import make, Environment
import gym

from .reward_spaces import GameResultReward

class ConnectFour(gym.Env, Environment):
    metadata = {"render.modes": []}

    def __init__(
            self,
            act_space,
            obs_space,
            seed = None
    ):
        super(ConnectFour, self).__init__()
        self.action_space = act_space
        self.obs_space = obs_space
        self.reward_space = GameResultReward()

        self.game = make('connectx')
        self.seed = seed if seed else 42

    def step(self, action):
        self.game.step(action)
        rewards = self.reward_space.compute_rewards(self.game)

        return (
            self.game.state,
            rewards,
            self.game.done,
            False,
            self.game.info
        )

    def reset(self, seed=None, options=None):
        self.game.reset()

    def render(self, **kwargs):
        print(self.game.render(mode='human'))

    def close(self):
        pass
