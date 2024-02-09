from abc import ABC, abstractmethod
from copy import deepcopy
from functools import lru_cache
import numpy as np
import numpy.ma as ma
import gym
import torch
from kaggle_environments.core import Environment
from typing import Dict, List, Optional, Tuple

from connectx.utility_constants import BOARD_SIZE

ROWS,COLUMNS = BOARD_SIZE

class BaseActSpace(ABC):
    @abstractmethod
    def get_action_space(self) -> gym.spaces.Dict:
        pass

    @abstractmethod
    def process_actions(
            self,
            action_tensors_dict: np.ndarray,
            game_state: Environment,
    ) -> List[str]:
        pass

    @staticmethod
    def get_available_actions_mask(game_state: Environment) -> Dict[str, np.ndarray]:
        pass


class BasicActionSpace(BaseActSpace):
    def get_action_space(self) -> gym.spaces.Dict:
        columns = BOARD_SIZE[1]
        return gym.spaces.Discrete(columns)

    def process_actions(
            self,
            action_logits: np.ndarray,
            game_state: np.ndarray,
    ) -> Tuple[List[List[str]], Dict[str, np.ndarray]]:
        mask = BasicActionSpace.get_available_actions_mask(game_state)
        valid_actions = ma.masked_array(action_logits, mask=mask)
        return valid_actions

    @staticmethod
    def get_available_actions_mask(game_state: np.ndarray) -> Dict[str, np.ndarray]:
        available_actions_mask = np.array(game_state.all(axis=0), dtype=bool)
        return available_actions_mask


class ImitationActionSpace(BaseActSpace):
    def get_action_space(self) -> gym.spaces.Dict:
        columns = BOARD_SIZE[1]
        return gym.spaces.Discrete(columns)

    def process_actions(
            self,
            action_logits: np.ndarray,
            game_state: np.ndarray,
    ) -> Tuple[List[List[str]], Dict[str, np.ndarray]]:
        mask = BasicActionSpace.get_available_actions_mask(game_state)
        valid_actions = ma.masked_array(action_logits, mask=mask)
        return valid_actions

    @staticmethod
    def get_available_actions_mask(game_state) -> Dict[str, np.ndarray]:
        mask_env = deepcopy(game_state)
        mask_env.run(['negamax', 'negamax'])
        available_actions_mask = np.array(game_state.all(axis=1), dtype=bool).reshape([1,COLUMNS])
        return available_actions_mask


if __name__ == "__main__":
    act = BasicActionSpace()
    space = act.get_action_space()
    print(space.contains(6))