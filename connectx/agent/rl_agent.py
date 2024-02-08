import os
from pathlib import Path
from types import SimpleNamespace
from typing import Dict
import yaml
import torch

from connectx.utils import Stopwatch
from connectx.connectx_gym import ConnectFour
from connectx.nns import create_model, models
from connectx.utils import flags_to_namespace
from connectx.connectx_gym import create_env

MODEL_CONFIG_PATH = Path(__file__).parent / "model_config.yaml"
RL_AGENT_CONFIG_PATH = Path(__file__).parent / "rl_agent_config.yaml"
CHECKPOINT_PATH,_ = list(Path(__file__).parent.glob('*.pt'))
AGENT = None

os.environ["OMP_NUM_THREADS"] = "1"

class RLAgent:
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
        )
        reward_space = create_reward_space(self.model_flags)
        env = RewardSpaceWrapper(env, reward_space)
        env = env.obs_space.wrap_env(env)
        env = LoggingEnv(env, reward_space)
        env = VecEnv([env])
        env = PytorchEnv(env, device_id)
        self.env = DictEnv(env)

        self.action_placeholder = torch.ones(1)

        self.model = create_model(self.model_flags, self.device)
        checkpoint_states = torch.load(CHECKPOINT_PATH, map_location=self.device)
        self.model.load_state_dict(checkpoint_states["model_state_dict"])
        self.model.eval()

        self.stopwatch = Stopwatch()

    def __call__(self, obs, conf):
        self.stopwatch.reset()

        self.stopwatch.start("Observation processing")
        # self.preprocess(obs, conf)
        # env_output = self.get_env_output()
        # print(f'output: {env_output["obs"]}')
        aam = self.env.unwrapped[0].action_space.get_available_actions_mask(obs)
        relevant_env_output_augmented = {
            "obs": env_output["obs"],
            "info": {
                "available_actions_mask": aam,
            },
        }

        self.stopwatch.stop().start("Model inference")
        with torch.no_grad():
            outputs = self.model.select_best_actions(relevant_env_output_augmented)
            agent_output = {
                "policy_logits": outputs["policy_logits"].cpu(),
                "baseline": outputs["baseline"].cpu()
            }
            agent_output["actions"] = models.DictActor.logits_to_actions(
                torch.flatten(agent_output["policy_logits"], start_dim=0, end_dim=-2),
                sample=False
            ).view(*agent_output["policy_logits"].shape[:-1], -1)


        actions = agent_output["action"]

        self.stopwatch.stop()


        value = agent_output["baseline"].numpy()
        value_msg = f"Turn: {self.game_state.turn} - Predicted value: {value:.2f}"
        timing_msg = f"{str(self.stopwatch)}"
        overage_time_msg = f"Remaining overage time: {obs['remainingOverageTime']:.2f}"

        # actions.append(annotate.sidetext(value_msg))
        print(" - ".join([value_msg, timing_msg, overage_time_msg]))
        return actions

    def get_env_output(self):
        return self.env.step(self.action_placeholder)

    def aggregate_augmented_predictions(self, policy: torch.Tensor) -> torch.Tensor:
        """
        Moves the predictions to the cpu, applies the inverse of all augmentations,
        and then returns the mean prediction for each available action.
        """
        # if len(self.data_augmentations) == 0:
        return policy.cpu()

        # policy_reoriented = [{key: val[0].unsqueeze(0) for key, val in policy.items()}]
        # for i, augmentation in enumerate(self.data_augmentations):
        #     augmented_policy = {key: val[i + 1].unsqueeze(0) for key, val in policy.items()}
        #     policy_reoriented.append(augmentation.apply(augmented_policy, inverse=True, is_policy=True))
        # return {
        #     key: torch.cat([d[key] for d in policy_reoriented], dim=0).mean(dim=0, keepdim=True)
        #     for key in policy.keys()
        # }

    @property
    def unwrapped_env(self) -> ConnectFour:
        return self.env.unwrapped[0]

    @property
    def game_state(self):
        return self.unwrapped_env.env.state

if __name__=="__main__":
    from kaggle_environments import make
    env = make('connectx')
    agent = RLAgent(1)

    env.play([agent, 'random'])
