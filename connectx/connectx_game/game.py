import numpy as np
from scipy.signal import convolve2d
import logging

from ..utility_constants import IN_A_ROW, BOARD_SIZE

logging.basicConfig(
    format=(
        "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
    ),
    level=0,
)
horizontal_kernel = np.ones([1, IN_A_ROW], dtype=np.uint8)
vertical_kernel = np.transpose(horizontal_kernel)
diag1_kernel = np.eye(IN_A_ROW, dtype=np.uint8)
diag2_kernel = np.fliplr(diag1_kernel)

class Game:
    victory_kernels = [
        horizontal_kernel,
        vertical_kernel,
        diag1_kernel,
        diag2_kernel,
    ]
    def initialize(self):
        self.board = np.zeros(shape=BOARD_SIZE, dtype=np.uint8)
        self.turn = 0

    def update(self, obs):
        logging.info('Game Updating')
        self.board = np.array(obs['board'], dtype=np.uint8).reshape(BOARD_SIZE)


    def step(self, action, mark):
        logging.info('Game Stepping')
        row = self.get_lowest_available_row(action)
        self.board[row, action] = mark

    def get_lowest_available_row(self, column):
        for row in range(BOARD_SIZE[0]-1,-1,-1):
            if self.board[row,column] == 0:
                return row
        raise StopIteration(f"Column {column} is full. {self.board[:,column]}")

    def mark_wins(self, mark):
        for kernel in self.victory_kernels:
            win1 = convolve2d(self.board == mark, kernel, mode="valid")
            if np.max(win1) == IN_A_ROW:
                return True
        return False