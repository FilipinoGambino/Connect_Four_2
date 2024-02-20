import numpy as np
from scipy.signal import convolve2d
import logging

from ..connectx_game.game_objects import Player
from ..utility_constants import BOARD_SIZE, IN_A_ROW, VICTORY_KERNELS, GAME_STATUS

logging.basicConfig(
    format=(
        "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
    ),
    level=0,
)

class Game:
    def __init__(self):
        self.board = np.zeros(shape=BOARD_SIZE, dtype=np.uint8)
        self.turn = 0
        self.players = [Player(1), Player(2)]

    def update(self, obs):
        self.board = np.array(obs['board'], dtype=np.uint8).reshape(BOARD_SIZE)
        self.turn = obs['step']

    def step(self, action):
        row = self.get_lowest_available_row(action)
        self.board[row, action] = self.players[self.turn % 2].mark
        self.turn += 1

    def get_lowest_available_row(self, column):
        for row in range(BOARD_SIZE[0]-1,-1,-1):
            if self.board[row,column] == 0:
                return row
        raise StopIteration(f"Column {column} is full. {self.board[:,column]}")

    def should_win(self, active, inactive):
        for kernel in VICTORY_KERNELS:
            convolutions1 = convolve2d(self.board == active.mark, kernel, mode="valid")
            convolutions2 = convolve2d(self.board == inactive.mark, kernel, mode="valid")
            enemy_can_win = (convolutions1 == 0) * convolutions2
            me_can_win = (convolutions2 == 0) * convolutions1

            if np.max(convolutions1) == IN_A_ROW:
                return GAME_STATUS['ACTIVE_PLAYER_WINS']
            elif np.max(me_can_win) == (IN_A_ROW - 1):
                return GAME_STATUS['THREE_OF_FOUR']
            elif np.max(enemy_can_win) == (IN_A_ROW - 1):
                return GAME_STATUS['UNDEFENDED_POSITION']
        return GAME_STATUS['NO_WINNING_MOVE']

    @property
    def active_player(self):
        '''
        Only called after step which increments turn
        :return: The active player object
        '''
        assert self.turn == self.board.size - np.count_nonzero(self.board==0)
        turn = self.turn - 1
        return self.players[turn % 2]

    @property
    def inactive_player(self):
        '''
        Only called after step which increments turn
        :return: The inactive player object
        '''
        assert self.turn == self.board.size - np.count_nonzero(self.board==0)
        turn = self.turn - 1
        return self.players[(turn + 1) % 2]

    @property
    def max_turns(self):
        if isinstance(self.board, np.ndarray):
            return self.board.size
        elif isinstance(self.board, list):
            return len(self.board)
        else:
            raise NotImplementedError