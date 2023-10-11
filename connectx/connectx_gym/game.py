import numpy as np
from scipy.signal import convolve2d

IN_A_ROW = 4

class Game:
    horizontal_kernel = np.ones([1,IN_A_ROW], dtype=np.uint8)
    vertical_kernel = np.transpose(horizontal_kernel)
    diag1_kernel = np.eye(IN_A_ROW, dtype=np.uint8)
    diag2_kernel = np.fliplr(diag1_kernel)
    detection_kernels = [horizontal_kernel, vertical_kernel, diag1_kernel, diag2_kernel]

    def __init__(self, config=None):
        self.config = config

        rows = config.rows
        cols = config.cols
        self.board = np.zeros((rows, cols), dtype=int)

        self.players = [1, 2]
        self.winner = None
        self.turn = 0
        self.done = False
        self.info = {}

    def step(self, col):
        self.turn = (self.turn + 1) % 2
        self.board[self.board[:,col].argmax()-1, col] = self.players[self.turn]

        if self.turn > IN_A_ROW and (self.winning_move or self.no_valid_moves):
            self.winner = self.players[self.turn]
            self.done = True

    def reset(self):
        self.board = np.zeros((self.config.rows, self.config.cols), dtype=np.uint8)
        self.turn = 0
        self.done = False
        self.info = {}

    def winning_move(self):
        for kernel in self.detection_kernels:
            if (convolve2d(self.board == self.players[self.turn], kernel, mode="valid") == IN_A_ROW).any():
                return True
        return False

    def no_valid_moves(self):
        return ~(self.board == 0).any()
