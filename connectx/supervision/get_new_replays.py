from kaggle_environments import make
import json
import time

env = make('connectx', {"rows": 6, "columns": 7, "inarow": 4})

start = time.time()
for idx in range(1000):
    if idx % 10 == 0 and idx > 0:
        print(f"{idx}: {(time.time() - start):.1f} seconds for 10 games")
        start = time.time()
    env.reset()

    env.run(['negamax', 'negamax'])

    path = f'.\\base_replays\\{idx:0>3}.json'
    with open(path, 'w') as file:
        json.dump(env.toJSON(), file)