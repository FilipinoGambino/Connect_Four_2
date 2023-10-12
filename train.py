import numpy as np
import os
import json

path = '.\\connectx\\base_replays\\'
for dir, folders, files in os.walk(path):
    fnames = files

for fname in fnames:
    with open(f"{path}{fname}", 'r') as data:
        match = json.load(data)
    break

for key,val in match.items():
    print(key)
    for key_,val_ in val.items():
        if key_ == "board":
            print(np.array(val_).reshape((6,7)))
        else:
            print(val_)
    print()