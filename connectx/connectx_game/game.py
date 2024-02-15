import numpy as np
from scipy.signal import convolve2d
import logging

from ..connectx_game.game_objects import Player
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

    def game_end(self):
        p1 = self.players[0]
        p2 = self.players[1]
        for kernel in VICTORY_KERNELS:
            convolutions1 = convolve2d(self.board == p1.mark, kernel, mode="valid")
            convolutions2 = convolve2d(self.board == p2.mark, kernel, mode="valid")
            if np.max(convolutions1) == IN_A_ROW:
                return p1
            if np.max(convolutions2) == IN_A_ROW:
                return p2
        return 'No Winner'