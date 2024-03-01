import numpy as np
import json
from pathlib import Path

replays = list(Path(__file__).parent.glob('replays\\*.json'))

for replay in replays:
    with open(replay) as data:
        game = json.load(data)
        for step in game['steps']:
            action = max(step[0]['action'], step[1]['action'])
            turn = step[0]['observation']['step']
            board = step[0]['observation']['board']
            print(f"{turn}:{action}\n{np.array(board).reshape((6,7))}")
        break