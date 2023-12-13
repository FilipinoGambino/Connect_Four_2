from pathlib import Path
from types import SimpleNamespace
import yaml
import torch

from connectx.actor_critic.policy import ActorCritic
from connectx.utils import flags_to_namespace
from connectx.connectx_gym import create_env

MODEL_CONFIG_PATH = Path(__file__).parent / "config.yaml"
RL_AGENT_CONFIG_PATH = Path(__file__).parent / "rl_agent_config.yaml"
ENV_CONFIG_PATH = Path(__file__).parent / "env_config.yaml"
# CHECKPOINT_PATH, = list(Path(__file__).parent.glob('*.pt'))


class RLAgent:
    def __init__(self, player_id):
        with open(MODEL_CONFIG_PATH, 'r') as file:
            self.model_flags = flags_to_namespace(yaml.safe_load(file))
        with open(RL_AGENT_CONFIG_PATH, 'r') as f:
            self.agent_flags = SimpleNamespace(**yaml.safe_load(f))
        with open(ENV_CONFIG_PATH, 'r') as f:
            self.env_flags = SimpleNamespace(**yaml.safe_load(f))

        if torch.cuda.is_available():
            if self.agent_flags.device == "player_id":
                device_id = f"cuda:{min(player_id, torch.cuda.device_count() - 1)}"
            else:
                device_id = self.agent_flags.device
        else:
            device_id = "cpu"

        self.device = torch.device(device_id)

        self.env = create_env(self.env_flags, self.device)
        print(self.env.observation_space.sample())
        self.model = ActorCritic()

    def __call__(self, *args, **kwargs):
        pass

    def step(self, action):
        self.env.step(action)