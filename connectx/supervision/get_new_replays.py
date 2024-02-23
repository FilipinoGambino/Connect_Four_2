from kaggle_environments import make
import json

env = make('connectx')

for idx in range(100):
    env.reset()

    env.run(['negamax', 'negamax'])

    path = f'.\\connectx\\supervision\\base_replays\\{idx:0>3}.json'
    with open(path, 'w') as file:
        json.dump(env.toJSON(), file)