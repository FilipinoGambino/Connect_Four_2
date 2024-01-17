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
# CHECKPOINT_PATH, = list(Path(__file__).parent.glob('*.pt'))


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

        self.env = create_env(self.model_flags, self.device)
        obs = self.env.unwrapped[0].obs_space.get_obs_spec().sample()
        self.action_placeholder = [torch.ones(1) for _ in range(self.model_flags.n_actor_envs)]

        num_inputs = 0
        for key in obs:
            num_inputs += obs[key].shape[0]
        num_inputs *= self.model_flags.n_actor_envs

        self.model = create_model(self.model_flags, self.device)

        self.stopwatch = Stopwatch()

    def __call__(self, obs, raw_model_output: bool = False):
        self.stopwatch.reset()

        self.stopwatch.start("Observation processing")
        # self.preprocess(obs, conf)
        env_output = self.get_env_output()
        print(f'output: {env_output["obs"]}')
        relevant_env_output_augmented = {
            "obs": env_output["obs"],
            "info": {
                "available_actions_mask": self.env.unwrapped[0].action_space.get_available_actions_mask(env_output['obs']),
            },
        }

        self.stopwatch.stop().start("Model inference")
        with torch.no_grad():
            agent_output_augmented = self.model.select_best_actions(relevant_env_output_augmented)
            agent_output = {
                "policy_logits": self.aggregate_augmented_predictions(agent_output_augmented["policy_logits"]),
                "baseline": agent_output_augmented["baseline"].mean(dim=0, keepdim=True).cpu()
            }
            agent_output["action"] = models.DictActor.logits_to_actions(
                torch.flatten(agent_output["policy_logits"], start_dim=0, end_dim=-2),
                sample=False,
                actions_per_square=None
            ).view(*agent_output["policy_logits"].shape[:-1], -1)

        # Used for debugging and visualization
        if raw_model_output:
            return agent_output

        self.stopwatch.stop().start("Collision detection")
        actions = agent_output["action"]

        self.stopwatch.stop()


        value = agent_output["baseline"].squeeze().numpy()[obs.player]
        value_msg = f"Turn: {self.game_state.turn} - Predicted value: {value:.2f}"
        timing_msg = f"{str(self.stopwatch)}"
        overage_time_msg = f"Remaining overage time: {obs['remainingOverageTime']:.2f}"

        # actions.append(annotate.sidetext(value_msg))
        # DEBUG_MESSAGE(" - ".join([value_msg, timing_msg, overage_time_msg]))
        return actions

    def get_env_output(self):
        print(self.action_placeholder)
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
    agent = RLAgent(1)
    print(agent(agent.env.reset()))