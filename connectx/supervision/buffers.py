import json
from pathlib import Path

replays = list(Path(__file__).parent.glob('*.json'))

for replay in replays:
    with open(replay) as data:
        game = json.load(data)
        for key,val in game.items():
            if key in ['steps', 'rewards','statuses']:
                print(key)
                print(game[key])
        break