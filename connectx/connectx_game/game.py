import numpy as np

from ..connectx_game.game_objects import Player
from ..utility_constants import BOARD_SIZE


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
        raise StopIteration(f"Column {column} is full. {self.board}")

    @property
    def active_player(self):
        '''
        Only called after step which increments turn
        :return: The active player object
        '''
        # assert self.turn == self.board.size - np.count_nonzero(self.board==0)
        return self.players[self.turn % 2]

    @property
    def inactive_player(self):
        '''
        Only called after step which increments turn
        :return: The inactive player object
        '''
        # assert self.turn == self.board.size - np.count_nonzero(self.board==0)
        return self.players[(self.turn + 1) % 2]

    @property
    def max_turns(self):
        if isinstance(self.board, np.ndarray):
            return self.board.size
        elif isinstance(self.board, list):
            return len(self.board)
        else:
            raise NotImplementedError