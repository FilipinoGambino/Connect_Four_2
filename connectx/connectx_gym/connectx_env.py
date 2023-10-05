from typing import Any, Optional, Dict

from kaggle_environments import make
import gym

class LuxEnv(gym.Env):
    metadata = {"render.modes": []}

    def __init__(
            self,
            act_space: BaseActSpace,
            obs_space: BaseObsSpace,
            configuration: Optional[Dict[str, Any]] = None,
            seed: Optional[int] = None,
            run_game_automatically: bool = True,
            restart_subproc_after_n_resets: int = 100
    ):
        super(LuxEnv, self).__init__()
        self.obs_space = obs_space
        self.action_space = act_space
        self.default_reward_space = GameResultReward()
        self.observation_space = self.obs_space.get_obs_spec()
        self.board_dims = MAX_BOARD_SIZE
        self.run_game_automatically = run_game_automatically
        self.restart_subproc_after_n_resets = restart_subproc_after_n_resets

        self.game_state = Game()
        if configuration is not None:
            self.configuration = configuration
        else:
            self.configuration = make("connectx").configuration
            # 2: warnings, 1: errors, 0: none
            self.configuration["loglevel"] = 0
        if seed is not None:
            self.seed(seed)
        elif "seed" not in self.configuration:
            self.seed()
        self.done = False
        self.info = {}
        self.pos_to_unit_dict = dict()
        self.pos_to_city_tile_dict = dict()
        self.reset_count = 0

        self._dimension_process = None
        self._q = None
        self._t = None
        self._restart_dimension_process()