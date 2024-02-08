import numpy as np

class Game:
    def _initialize(self, conf):
        self.board_shape = (conf['rows'], conf['columns'])
        self.board = np.zeros(self.board_shape)
        self.inarow = conf['inarow']
        self.max_turns = conf['rows'] * conf['columns']

    def _update(self, obs):
        self.remaining_time = obs['remainingOverageTime']
        self.board = np.array(obs['board']).reshape(self.board_shape)
        self.step = obs['step']
        self.mark = obs['mark']


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
        game = Game()
        game._initialize(configuration)
        game._update(observation, configuration)
        logging.info(game.board)
        return choice([c for c in range(configuration.columns) if observation.board[c] == 0])


    env = make('connectx')
    env.run([my_agent, 'random'])