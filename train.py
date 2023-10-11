import os
import json

path = 'C:\\Users\\nick.gorichs\\PycharmProjects\\Connect_Four_2\\connectx\\replays\\'
for dir, folders, files in os.walk(path):
    fnames = files

for fname in fnames:
    with open(f"{path}{fname}", 'r') as data:
        match = json.load(data)
    break

data = {}
for _,step in match.items():
    data['player_1'] = {
        'status': step[0]['status'],
        'action': step[0]['action'],
        'reward': step[0]['reward'],
        'info': step[0]['info']
    }