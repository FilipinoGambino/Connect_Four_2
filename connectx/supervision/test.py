import numpy as np
import pandas as pd
# pd.set_option('display.max_columns', None)

df = pd.read_pickle('.\\pandas_replays\\stacked_replays.pkl')
print(df.shape)
print(df[['action','turn','p1_active','idx_35','idx_36','idx_37','idx_38','idx_39','idx_40','idx_41']].head(5))
# for idx in range(5):
#     board = df.iloc[idx,:42].to_numpy().reshape(6,7)
#     print(df.loc[idx,'action'])
#     print(board)