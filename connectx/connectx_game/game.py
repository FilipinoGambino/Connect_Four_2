import numpy as np
from scipy.signal import convolve2d

class Game:
    def __init__(self, conf):
        self.board_shape = (conf['rows'], conf['columns'])
        self.board = np.zeros(self.board_shape, dtype=np.uint8)
        self.inarow = conf['inarow']
        self.max_turns = conf['rows'] * conf['columns']
        self.remaining_time = conf['agentTimeout']
        self.turn = 0
        self.mark = 1 # Placeholder for building buffers from the observation space
        self.reward = 0

        horizontal_kernel = np.ones([1, self.inarow], dtype=np.uint8)
        vertical_kernel = np.transpose(horizontal_kernel)
        diag1_kernel = np.eye(self.inarow, dtype=np.uint8)
        diag2_kernel = np.fliplr(diag1_kernel)

        self.victory_kernels = [
            horizontal_kernel,
            vertical_kernel,
            diag1_kernel,
            diag2_kernel,
        ]

    def update(self, obs):
        print('updating')
        self.remaining_time = obs['remainingOverageTime']
        self.board = np.array(obs['board'], dtype=np.uint8).reshape(self.board_shape)
        self.turn = obs['step']
        self.mark = obs['mark']
        self.opp_mark = None

    def step(self, action):
        print('stepping')
        row = self.get_lowest_available_row(action)
        self.board[row, action] = self.mark

    def get_lowest_available_row(self, column):
        for row in range(self.board_shape[0]-1,-1,-1):
            if self.board[row,column] == 0:
                return row
        raise StopIteration(f"Column {column} is full. {self.board[:,column]}")

    def is_win(self):
        for kernel in self.victory_kernels:
            my_win = convolve2d(self.board == self.mark, kernel, mode="valid")
            opp_win = convolve2d(self.board == self.mark, kernel, mode="valid")
            if np.max(my_win) == self.inarow:
                self.reward = 1
            elif np.max(opp_win) == self.inarow:
                self.reward = -1
        self.reward = 0


if __name__=='__main__':
    from kaggle_environments import make
    from random import choice
    import logging

    logging.basicConfig(
        format=(
            "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "\n%(message)s"
        ),
        level=0,
    )

    def my_agent(observation, configuration):
        game = Game(configuration)
        game.update(observation)
        logging.info(game.board)
        return choice([c for c in range(configuration.columns) if observation.board[c] == 0])


    config = make('connectx').configuration
    # env.run([my_agent, 'random'])