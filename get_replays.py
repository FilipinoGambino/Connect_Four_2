from kaggle_environments import make

import json
import random

env = make('connectx')
env.reset()
for idx in range(500):
    player1 = random.choice(['negamax', 'random'])
    player2 = random.choice(['negamax', 'random'])
    env.run([player1, player2])
    with open(f'C:\\Users\\nick.gorichs\\PycharmProjects\\Connect_Four_2\\connectx\\replays\\{idx:0>3}_{player1}_{player2}.json', 'w') as file:
        data = {}
        for ix,step in enumerate(env.steps):
            data[f"step_{ix:0>2}"] = step
        json.dump(data, file)
    env.reset()