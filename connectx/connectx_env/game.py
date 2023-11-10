from scipy.signal import convolve2d

from .game_objects import Player, GameBoard

IN_A_ROW = 4

class Game:
    def __init__(self, config=None):
        self.config = config

        height = config.rows
        width = config.cols
        self.board = GameBoard(height, width)

        self.players = [Player(1), Player(2)]
        self.turn = 0
        self.done = False
        self.info = {}

    def step(self, col):
        self.turn += 1
        active_player = self.turn % 2
        inactive_player = (self.turn + 1) % 2

        self.board.place_mark(col, self.players[active_player].mark)

        if self.turn > IN_A_ROW and self.winning_move:
            self.players[active_player].winner()
            self.players[inactive_player].loser()
            self.done = True
        elif self.turn > IN_A_ROW and self.no_valid_moves:
            # Tie
            [player.winner() for player in self.players]

    def winning_move(self):
        for kernel in self.board.victory_kernels:
            if (convolve2d(self.board == self.players[self.turn], kernel, mode="valid") == IN_A_ROW).any():
                return True
        return False

    def no_valid_moves(self):
        return ~(self.board == 0).any()