from pathlib import Path
from types import SimpleNamespace
import yaml
import torch

from connectx.connectx_gym import ConnectFour
from connectx.actor_critic.policy import ActorCritic
from connectx.utils import flags_to_namespace
from connectx.connectx_gym import create_env

MODEL_CONFIG_PATH = Path(__file__).parent / "model_config.yaml"
RL_AGENT_CONFIG_PATH = Path(__file__).parent / "rl_agent_config.yaml"
ENV_CONFIG_PATH = Path(__file__).parent.parent.parent / "env_config.yaml"
# CHECKPOINT_PATH, = list(Path(__file__).parent.glob('*.pt'))


class Actor:
    def __init__(self, player_id):
        # with open(MODEL_CONFIG_PATH, 'r') as file:
        #     self.model_flags = flags_to_namespace(yaml.safe_load(file))
        with open(RL_AGENT_CONFIG_PATH, 'r') as f:
            self.agent_flags = SimpleNamespace(**yaml.safe_load(f))
        with open(ENV_CONFIG_PATH, 'r') as f:
            self.env_flags = flags_to_namespace(yaml.safe_load(f))

        if torch.cuda.is_available():
            if self.agent_flags.device == "player_id":
                device_id = f"cuda:{min(player_id, torch.cuda.device_count() - 1)}"
            else:
                device_id = self.agent_flags.device
        else:
            device_id = "cpu"

        self.device = torch.device(device_id)

        self.env = create_env(self.env_flags, self.device)
        obs = self.env.unwrapped[0].obs_space.get_obs_spec().sample()

        num_inputs = 0
        for key in obs:
            num_inputs += obs[key].shape[0]
        num_inputs *= self.env_flags.n_actor_envs

        self.model = ActorCritic(num_inputs)

    def __call__(self, *args, **kwargs):
        logits = self.step

    def step(self, action):
        self.env.step(action)

    @property
    def unwrapped_env(self) -> ConnectFour:
        return self.env.unwrapped[0]

    @property
    def game_state(self):
        return self.unwrapped_env.env.state

if __name__=="__main__":
    agent = Actor(1)
    print(agent.model)