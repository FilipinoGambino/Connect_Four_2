import numpy as np

from game_constants import Constants

IN_A_ROW = Constants.IN_A_ROW

class Player:
    def __init__(self, mark):
        self.mark = mark
        self.victory = False

    def winner(self):
        self.victory = True

    def loser(self):
        self.victory = False

class GameBoard:
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.board = np.zeros([height, width], dtype=np.uint8)

        horizontal_kernel = np.ones([1, IN_A_ROW], dtype=np.uint8)
        vertical_kernel = np.transpose(horizontal_kernel)
        diag1_kernel = np.eye(IN_A_ROW, dtype=np.uint8)
        diag2_kernel = np.fliplr(diag1_kernel)

        self.victory_kernels = [horizontal_kernel, vertical_kernel, diag1_kernel, diag2_kernel]

    def place_mark(self, col, mark):
        row = self.lowest_valid_rows()[col]
        self.board[row, col] = mark

    def lowest_valid_rows(self):
        mask = self.board != 0
        return np.where(mask.any(axis=0), mask.argmax(axis=0), self.height) - 1


if __name__ == '__main__':
    gb = GameBoard(6, 7)
    gb.board[:, 1] = 2
    gb.board[:, 2] = 1
    gb.board[:, 3] = 2
    gb.board[:, 4] = 0
    gb.board[:, 5] = 2

    gb.board[0, 1] = 0
    print(gb.board, '\n')
    print(gb.lowest_valid_rows())