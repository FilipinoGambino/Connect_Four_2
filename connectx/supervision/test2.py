import pandas as pd
import json

df = pd.read_pickle('.\\pandas_replays\\stacked_replays.pkl')
print(df)
# df = df.astype(int)
# df = df.astype({'p1_active':'bool'})
# df.to_pickle(f'.\\pandas_replays\\stacked_replays.pkl')
