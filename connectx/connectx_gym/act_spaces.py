from abc import ABC, abstractmethod
from functools import lru_cache
import numpy as np
import gym
from kaggle_environments.core import Environment

from typing import Dict, List, Optional, Tuple

class BaseActSpace(ABC):
    @abstractmethod
    def get_action_space(self, board_dims: Tuple[int, int] = MAX_BOARD_SIZE) -> gym.spaces.Dict:
        pass

    @abstractmethod
    def process_actions(
            self,
            action_tensors_dict: Dict[str, np.ndarray],
            game_state: Environment,
    ) -> List[str]:
        pass

    @abstractmethod
    def get_available_actions_mask(
            self,
            game_state: Environment,
    ) -> Dict[str, np.ndarray]:
        pass

    @staticmethod
    @abstractmethod
    def actions_taken_to_distributions(actions_taken: Dict[str, np.ndarray]) -> Dict[str, Dict[str, int]]:
        pass


class BasicActionSpace(BaseActSpace):
    def __init__(self, default_board_dims: Optional[Tuple[int, int]] = None):
        self.default_board_dims = MAX_BOARD_SIZE if default_board_dims is None else default_board_dims

    @lru_cache(maxsize=None)
    def get_action_space(self, board_dims: Optional[Tuple[int, int]] = None) -> gym.spaces.Dict:
        if board_dims is None:
            board_dims = self.default_board_dims
        x = board_dims[0]
        y = board_dims[1]
        # Player count
        p = 2
        return gym.spaces.Dict({
            "worker": gym.spaces.MultiDiscrete(np.zeros((1, p, x, y), dtype=int) + len(ACTION_MEANINGS["worker"])),
            "cart": gym.spaces.MultiDiscrete(np.zeros((1, p, x, y), dtype=int) + len(ACTION_MEANINGS["cart"])),
            "city_tile": gym.spaces.MultiDiscrete(
                np.zeros((1, p, x, y), dtype=int) + len(ACTION_MEANINGS["city_tile"])
            ),
        })

    @lru_cache(maxsize=None)
    def get_action_space_expanded_shape(self, *args, **kwargs) -> Dict[str, Tuple[int, ...]]:
        action_space = self.get_action_space(*args, **kwargs)
        action_space_expanded = {}
        for key, val in action_space.spaces.items():
            action_space_expanded[key] = val.shape + (len(ACTION_MEANINGS[key]),)
        return action_space_expanded

    def process_actions(
            self,
            action_tensors_dict: Dict[str, np.ndarray],
            game_state: Environment,
            board_dims: Tuple[int, int],
            pos_to_unit_dict: Dict[Tuple, Optional[Unit]]
    ) -> Tuple[List[List[str]], Dict[str, np.ndarray]]:
        action_strs = [[], []]
        actions_taken = {
            key: np.zeros(space, dtype=bool) for key, space in self.get_action_space_expanded_shape(board_dims).items()
        }
        return action_strs, actions_taken

    def get_available_actions_mask(
            self,
            game_state: Environment,
            board_dims: Tuple[int, int],
            pos_to_unit_dict: Dict[Tuple, Optional[Unit]],
            pos_to_city_tile_dict: Dict[Tuple, Optional[CityTile]]
    ) -> Dict[str, np.ndarray]:

        return available_actions_mask

    @staticmethod
    def actions_taken_to_distributions(actions_taken: Dict[str, np.ndarray]) -> Dict[str, Dict[str, int]]:
        out = {}
        for space, actions in actions_taken.items():
            out[space] = {
                ACTION_MEANINGS[space][i]: actions[..., i].sum()
                for i in range(actions.shape[-1])
            }