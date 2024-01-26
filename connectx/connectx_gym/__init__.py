import torch
from typing import Optional

from . import act_spaces, obs_spaces, reward_spaces
from .connectx_env import ConnectFour
from .wrappers import DictEnv, LoggingEnv, PytorchEnv, RewardSpaceWrapper, TensorflowEnv, VecEnv

ACT_SPACES_DICT = {
    key: val for key, val in act_spaces.__dict__.items()
    if isinstance(val, type) and issubclass(val, act_spaces.BaseActSpace)
}
OBS_SPACES_DICT = {
    key: val for key, val in obs_spaces.__dict__.items()
    if isinstance(val, type) and issubclass(val, obs_spaces.BaseObsSpace)
}
REWARD_SPACES_DICT = {
    key: val for key, val in reward_spaces.__dict__.items()
    if isinstance(val, type) and issubclass(val, reward_spaces.BaseRewardSpace)
}

def create_flexible_obs_space(flags, teacher_flags: Optional) -> obs_spaces.BaseObsSpace:
    return flags.obs_space(**flags.obs_space_kwargs)


def create_env(flags, device: torch.device, teacher_flags: Optional = None, seed: Optional[int] = None) -> PytorchEnv:
    if seed is None:
        seed = flags.seed
    envs = []
    for i in range(flags.n_actor_envs):
        env = ConnectFour(
            act_space=flags.act_space(),
            obs_space=create_flexible_obs_space(flags, teacher_flags),
            player_id=flags.player_id,
            seed=seed
        )
        reward_space = create_reward_space(flags)
        env = RewardSpaceWrapper(env, reward_space)
        env = env.obs_space.wrap_env(env)
        env = LoggingEnv(env, reward_space)
        envs.append(env)
    env = VecEnv(envs)
    env = PytorchEnv(env, device)
    # env = TensorflowEnv(env, device)
    env = DictEnv(env)
    return env


def create_reward_space(flags) -> reward_spaces.BaseRewardSpace:
    return flags.reward_space(**flags.reward_space_kwargs)