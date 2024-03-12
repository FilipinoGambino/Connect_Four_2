import json
import math
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

BOARD_SIZE = (6,7)

step_names = [f"idx_{idx}" for idx in range(math.prod(BOARD_SIZE))]
full_df = pd.DataFrame(columns=[*step_names, 'turn', 'p1_active'])

replays = list(Path(__file__).parent.glob('base_replays\\*.json'))
for replay in replays:
    df = pd.DataFrame(columns=[*step_names, 'turn', 'p1_active'])
    with open(replay) as data:
        game = json.load(data)
        state = np.zeros(shape=(math.prod(BOARD_SIZE)), dtype=np.int64)
        actions = np.zeros(shape=math.prod(BOARD_SIZE) + 2, dtype=np.int64)
        rows = np.full(fill_value=BOARD_SIZE[0], shape=(BOARD_SIZE[-1]), dtype=np.int64)
        for step in game['steps']:
            turn = step[0]['observation']['step']
            board = step[0]['observation']['board']
            action = max(step[0]['action'], step[1]['action'])
            actions[turn] = action

            if step[0]['status'] == 'ACTIVE':
                active_p = step[0]['observation']['mark']
            elif step[1]['status'] == 'ACTIVE':
                active_p = step[1]['observation']['mark']
            else:
                continue

            if turn > 0:
                try:
                    last_action_position = np.logical_xor(board, state)
                    state = np.where(last_action_position, turn, state)
                except ValueError:
                    raise ValueError(f"{board}\n{state}")

            d = pd.DataFrame({key:val for key,val in zip(df.columns, [*state, turn, active_p - 1 == 0])}, index=[0])
            df = pd.concat([df, d], ignore_index=True)

        df['action'] = actions[1:df.shape[0]+1]
    full_df = pd.concat([full_df, df], ignore_index=True)

full_df.to_pickle(f'.\\pandas_replays\\stacked_replays.pkl')
print('DONE!')