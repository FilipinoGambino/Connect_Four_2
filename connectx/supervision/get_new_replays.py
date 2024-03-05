from kaggle_environments import make
import json

env = make('connectx')

for idx in range(1000):
    print(idx)
    env.reset()

    env.run(['negamax', 'negamax'])

    path = f'.\\base_replays\\{idx:0>4}.json'
    with open(path, 'w') as file:
        json.dump(env.toJSON(), file)