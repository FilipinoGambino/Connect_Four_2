import json
from pathlib import Path

replays = list(Path(__file__).parent.glob('replays\\*.json'))

for replay in replays:
    with open(replay) as data:
        game = json.load(data)
        print(game['steps'])
        break