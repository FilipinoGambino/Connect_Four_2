import numpy as np
from scipy.signal import convolve2d
import logging

from ..utility_constants import BOARD_SIZE, IN_A_ROW, VICTORY_KERNELS

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
        self.current_player = 1

    def update(self, obs):
        self.board = np.array(obs['board'], dtype=np.uint8).reshape(BOARD_SIZE)
        self.turn = obs['step']
        self.current_player = obs['mark']

    def step(self, action):
        logging.info('Game Stepping')
        row = self.get_lowest_available_row(action)
        self.board[row, action] = self.current_player

    def get_lowest_available_row(self, column):
        for row in range(BOARD_SIZE[0]-1,-1,-1):
            if self.board[row,column] == 0:
                return row
        raise StopIteration(f"Column {column} is full. {self.board[:,column]}")

    def game_end(self):
        for kernel in VICTORY_KERNELS:
            win = convolve2d(self.board == self.current_player, kernel, mode="valid")
            if np.max(win) == IN_A_ROW:
                return True
        return False