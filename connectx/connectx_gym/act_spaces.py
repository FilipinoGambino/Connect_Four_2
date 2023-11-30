from abc import ABC, abstractmethod
from functools import lru_cache
import numpy as np
import numpy.ma as ma
import gym
from kaggle_environments.core import Environment

from typing import Dict, List, Optional, Tuple
from ..utility_constants import BOARD_SIZE

ROWS,COLUMNS = BOARD_SIZE

class BaseActSpace(ABC):
    @abstractmethod
    def get_action_space(self, board_dims: Tuple[int, int] = BOARD_SIZE) -> gym.spaces.Dict:
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
    def __init__(self, default_board_dims: Optional[Tuple[int, int]] = None):
        self.default_board_dims = BOARD_SIZE if default_board_dims is None else default_board_dims

    @lru_cache(maxsize=None)
    def get_action_space(self, board_dims: Optional[Tuple[int, int]] = None) -> gym.spaces.Dict:
        if board_dims is None:
            board_dims = self.default_board_dims
        columns = board_dims[1]
        return gym.spaces.Discrete(columns)

    # @lru_cache(maxsize=None)
    def process_actions(
            self,
            action_tensors: np.ndarray,
            game_state: np.ndarray,
    ) -> Tuple[List[List[str]], Dict[str, np.ndarray]]:
        mask = BasicActionSpace.get_available_actions_mask(game_state)
        valid_actions = ma.masked_array(action_tensors, mask=mask)
        return valid_actions

    @staticmethod
    def get_available_actions_mask(game_state: Environment) -> Dict[str, np.ndarray]:
        available_actions_mask = np.array(game_state.all(axis=0)).reshape([1,COLUMNS])
        return available_actions_mask