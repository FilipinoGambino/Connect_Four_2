import torch
from torch import nn
from typing import Optional

from ..connectx_gym import create_flexible_obs_space, obs_spaces

from ..connectx_game.game_constants import Constants

height, width = Constants.BOARD_SIZE

def create_model(
        flags,
        device: torch.device,
        teacher_model_flags: Optional = None,
        is_teacher_model: bool = False
) -> nn.Module:
    obs_space = create_flexible_obs_space(flags, teacher_model_flags)

    return _create_model(
        teacher_model_flags if is_teacher_model else flags,
        device,
        obs_space,
    )


def _create_model(
        flags,
        device: torch.device,
        obs_space: obs_spaces.BaseObsSpace,
):
    act_space = flags.act_space()
    if flags.model_arch == "conv_model":
        base_model = nn.Sequential(
            conv_embedding_input_layer,
            *[ResidualBlock(
                in_channels=flags.hidden_dim,
                out_channels=flags.hidden_dim,
                height=height,
                width=width,
                kernel_size=flags.kernel_size,
                normalize=flags.normalize,
                activation=nn.LeakyReLU,
                rescale_se_input=flags.rescale_se_input,
            ) for _ in range(flags.n_blocks)]
        )
    if flags.model_arch == "linear_model":
        base_model = nn.Sequential(
            *[(nn.Linear(flags.in_dim, flags.hidden_dim),
            nn.LeakyReLU(),
            ) for _ in range(flags.n_blocks)]
        )

    model = BasicActorCriticNetwork(
        base_model=base_model,
        base_out_channels=flags.hidden_dim,
        action_space=act_space.get_action_space(),
        reward_space=flags.reward_space.get_reward_spec(),
        n_value_heads=1,
        rescale_value_input=flags.rescale_value_input
    )
    return model.to(device=device)