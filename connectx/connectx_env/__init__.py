from connectx.connectx_gym.reward_spaces import GameResultReward
from gym.envs.registration import register

register(
    id='connectx_env/Connect_Four-v0',
    entry_point='connectx_env.envs:ConnectFourEnv',
    max_episode_steps=100,
)