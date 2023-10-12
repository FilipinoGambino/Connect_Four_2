from kaggle_environments import make

import json
import random

env = make('connectx')

for idx in range(5):
    env.reset()

    player1 = random.choice(['negamax', 'random'])
    player2 = random.choice(['negamax', 'random'])

    env.run([player1, player2])

    data = {}
    for ix, step in enumerate(env.steps):
        data[f'{ix:0>2}'] = {
            'board': step[0]['observation']['board'],
            'player_1': {
                'status': step[0]['status'],
                'mark': step[0]['observation']['mark'],
                'remainingOverageTime': step[0]['observation']['remainingOverageTime'],
                'action': step[0]['action'],
                'reward': step[0]['reward'],
                'info': step[0]['info']
            },
            'player_2': {
                'status': step[1]['status'],
                'mark': step[1]['observation']['mark'],
                'remainingOverageTime': step[1]['observation']['remainingOverageTime'],
                'action': step[1]['action'],
                'reward': step[1]['reward'],
                'info': step[1]['info']
            }
        }

    path = f'.\\connectx\\base_replays\\{idx:0>3}_{player1}_{player2}.json'
    with open(path, 'w') as file:
        json.dump(data, file)