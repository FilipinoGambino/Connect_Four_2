import numpy as np

class Game:
    def __init__(self, conf):
        self.board_shape = (conf['rows'], conf['columns'])
        self.board = np.zeros(self.board_shape)
        self.inarow = conf['inarow']
        self.max_turns = conf['rows'] * conf['columns']
        self.remaining_time = conf['agentTimeout']
        self.turn = 0
        self.mark = 1 # Placeholder for buffers

    def update(self, obs):
        self.remaining_time = obs['remainingOverageTime']
        self.board = np.array(obs['board']).reshape(self.board_shape)
        self.turn = obs['step']
        self.mark = obs['mark']

    def step(self, action):
        row = self.get_lowest_available_row(action)
        self.board[row, action] = self.mark

    def get_lowest_available_row(self, column):
        for row in range(self.board_shape[0]):
            if self.board[row,column] != 0:
                return row - 1
        return self.board_shape[0] - 1


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
    env.run([my_agent, 'random'])