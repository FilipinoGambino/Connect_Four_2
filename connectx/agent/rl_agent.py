import logging
import numpy as np
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Dict
import yaml
import torch

from connectx.utils import Stopwatch
from connectx.connectx_gym import create_reward_space, ConnectFour, wrappers
from connectx.nns import create_model, models
from connectx.utils import flags_to_namespace

MODEL_CONFIG_PATH = Path(__file__).parent / "model_config.yaml"
RL_AGENT_CONFIG_PATH = Path(__file__).parent / "rl_agent_config.yaml"
CHECKPOINT_PATH,_ = list(Path(__file__).parent.glob('*.pt'))
AGENT = None

os.environ["OMP_NUM_THREADS"] = "1"

logging.basicConfig(
    format=(
        "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
    ),
    level=0,
)

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
            autoplay=False
        )
        reward_space = create_reward_space(self.model_flags)
        env = wrappers.RewardSpaceWrapper(env, reward_space)
        env = env.obs_space.wrap_env(env)
        env = wrappers.LoggingEnv(env, reward_space)
        env = wrappers.VecEnv([env])
        env = wrappers.PytorchEnv(env, device_id)
        self.env = wrappers.DictEnv(env)

        self.action_placeholder = torch.ones(1)

        self.model = create_model(self.model_flags, self.device)
        checkpoint_states = torch.load(CHECKPOINT_PATH, map_location=self.device)
        self.model.load_state_dict(checkpoint_states["model_state_dict"])
        self.model.eval()

        self.stopwatch = Stopwatch()

    def __call__(self, obs, conf):
        self.stopwatch.reset()

        self.stopwatch.start("Observation processing")
        self.preprocess(obs)

        env_output = self.get_env_output()

        self.stopwatch.stop().start("Model inference")
        with torch.no_grad():
            outputs = self.model.select_best_actions(env_output)
            # outputs = self.model.sample_actions(env_output)
            # agent_output = {
            #     "policy_logits": outputs["policy_logits"].cpu(),
            #     "baseline": outputs["baseline"].cpu()
            # }
            # agent_output["actions"] = models.DictActor.logits_to_actions(
            #     torch.flatten(agent_output["policy_logits"], start_dim=0, end_dim=-2),
            #     sample=False
            # ).view(*agent_output["policy_logits"].shape[:-1], -1)


        action = outputs["actions"].item()

        self.stopwatch.stop()

        value = outputs["baseline"].numpy().item(0)
        value_msg = f"Turn: {obs['step']} - Predicted value: {value:.2f} | Column:{action} |"
        timing_msg = f"{str(self.stopwatch)}"
        overage_time_msg = f"Remaining overage time: {obs['remainingOverageTime']:.2f}"

        print(" - ".join([value_msg, timing_msg, overage_time_msg]))
        return action

    def get_env_output(self):
        return self.env.step(self.action_placeholder)

    def preprocess(self, obs):
        if obs['step'] == 0:
            self.unwrapped_env.reset()
        else:
            self.unwrapped_env.manual_step(obs)

    def aggregate_augmented_predictions(self, policy: torch.Tensor) -> torch.Tensor:
        """
        Moves the predictions to the cpu, applies the inverse of all augmentations,
        and then returns the mean prediction for each available action.
        """
        return policy.cpu()

    @property
    def unwrapped_env(self) -> ConnectFour:
        return self.env.unwrapped[0]
    #
    # @property
    # def game_state(self):
    #     return self.unwrapped_env.game_state
    #
    # @property
    # def board(self):
    #     return self.unwrapped_env.board

if __name__=="__main__":
    from kaggle_environments import evaluate, make
    env = make('connectx', debug=False)

    env.reset()
    env.run([RLAgent(1), 'negamax'])
    print(env.render(mode='ansi'))
    env.reset()
    # env.run(['random', RLAgent(2)])
    # print(env.render(mode='ansi'))
    # env.reset()
    # env.run([RLAgent(1), 'negamax'])
    # print(env.render(mode='ansi'))
    # # env.play([RLAgent(1), None])


    def mean_reward1(rewards):
        return sum(r[0] for r in rewards) / float(len(rewards))

    def mean_reward2(rewards):
        return sum(r[-1] for r in rewards) / float(len(rewards))


    # Run multiple episodes to estimate its performance.
    # print("My Agent vs Random Agent:", mean_reward1(evaluate("connectx", [RLAgent(1), "random"], num_episodes=100)))
    # print("My Agent vs Random Agent:", mean_reward2(evaluate("connectx", ["random", RLAgent(2)], num_episodes=100)))
    # print("My Agent vs Negamax Agent:", mean_reward1(evaluate("connectx", [RLAgent(1), "negamax"], num_episodes=10)))
    # print("My Agent vs Negamax Agent:", mean_reward2(evaluate("connectx", ["negamax", RLAgent(2)], num_episodes=10)))

    # print(evaluate("connectx", [RLAgent(1), "random"], num_episodes=100))