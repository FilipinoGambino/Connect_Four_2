from pathlib import Path
import torch
from types import SimpleNamespace
import yaml

from connectx.utils import Stopwatch
from connectx.nns import create_model
from connectx.utils import flags_to_namespace
from connectx.connectx_gym import ConnectFour, create_reward_space, wrappers

MODEL_CONFIG_PATH = Path(__file__).parent / "model_config.yaml"
RL_AGENT_CONFIG_PATH = Path(__file__).parent / "rl_agent_config.yaml"
CHECKPOINT_PATH, _ = list(Path(__file__).parent.glob('*.pt'))


class SLAgent:
    def __init__(self, player_id):
        with open(MODEL_CONFIG_PATH, 'r') as file:
            self.model_flags = flags_to_namespace(yaml.safe_load(file))
        with open(RL_AGENT_CONFIG_PATH, 'r') as f:
            self.agent_flags = SimpleNamespace(**yaml.safe_load(f))

        if torch.cuda.is_available():
            if self.agent_flags.device == "player_id":
                device_id = f"cuda:{min(player_id, torch.cuda.device_count() - 1)}"
            else:
                device_id = self.agent_flags.device
        else:
            device_id = "cpu"

        self.device = torch.device(device_id)

        env = ConnectFour(
            act_space=self.model_flags.act_space(),
            obs_space=self.model_flags.obs_space(),
            autoplay=False
        )
        reward_space = create_reward_space(self.model_flags)
        env = wrappers.RewardSpaceWrapper(env, reward_space)
        env = env.obs_space.wrap_env(env)
        env = wrappers.LoggingEnv(env, reward_space)
        env = wrappers.VecEnv([env])
        env = wrappers.PytorchEnv(env, device_id)
        self.env = wrappers.DictEnv(env)

        self.model = create_model(self.model_flags, self.device)
        checkpoint_states = torch.load(CHECKPOINT_PATH, map_location=self.device)
        self.model.load_state_dict(checkpoint_states["model_state_dict"])
        self.model.eval()

        self.stopwatch = Stopwatch()

    def __call__(self, obs, conf):
        self.stopwatch.reset()

        self.stopwatch.start("Observation processing")
        env_output = self.env.reset(configuration=conf, observation=obs)

        self.stopwatch.stop().start("Model prediction")
        outputs = self.model(env_output)
        self.stopwatch.stop()

        return outputs