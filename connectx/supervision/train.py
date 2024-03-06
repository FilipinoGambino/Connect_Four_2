import logging
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from typing import Dict, Tuple
# pd.set_option('display.max_columns', None)

from connectx.nns import create_model

logging.basicConfig(
    format=(
        "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
    ),
    level=0,
)


class ConnectFourDataset(Dataset):
    def __init__(self, fname, transform):
        self.df = pd.read_pickle(fname)
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        series = self.df.iloc[idx,:]
        return self.transform(series)

def convert_to_obs_space(x: pd.Series) -> Tuple[np.ndarray, Dict]:
    action = np.zeros(shape=(7), dtype=np.float32)
    action[int(x['action'])] = 1.
    p1_active = np.full(shape=(6,7), fill_value=x['p1_active'], dtype=np.int64)
    norm_turn = np.full(shape=(1,1), fill_value=x['turn'] / 42, dtype=np.float32)
    board_by_turn = x[:42].to_numpy()
    p1_board = np.where(board_by_turn % 2 == 1, 1, 0)
    p2_board = np.where(np.logical_and(board_by_turn > 0, board_by_turn % 2 == 0), 2, 0)
    board = p1_board + p2_board
    available_actions_mask = np.array(board.reshape(6,7).all(axis=0), dtype=bool)
    flipped_aam = np.array(np.fliplr(board.reshape(6,7)).all(axis=0), dtype=bool).copy()

    keys = [
        "active_player_t-0",
        "inactive_player_t-1",
        "active_player_t-1",
        "inactive_player_t-2",
        "active_player_t-2",
        "inactive_player_t-3",
        "active_player_t-3"
    ]
    obs = {"inactive_player_t-0": np.where(board==2, 1, 0).reshape(6,7)}
    for key in keys:
        last_move = np.argmax(board_by_turn)
        board_by_turn[last_move] = 0
        board[last_move] = 0
        if key.startswith('inactive'):
            obs[key] = np.where(board==2, 1, 0).reshape(6,7)
        elif key.startswith('active'):
            obs[key] = np.where(board==1, 1, 0).reshape(6,7)
        else:
            raise Exception(f"{key} is invalid")

    # Keeping the order the same as RL observation space
    obs = {k: v for k, v in sorted(obs.items(), key=lambda item: item[0])}
    obs['p1_active'] = p1_active
    obs['turn'] = norm_turn
    flipped_obs = {key:np.fliplr(val).copy() for key,val in obs.items()}

    output = dict(
        obs=dict(obs=obs, info={"available_actions_mask":available_actions_mask}),
        flipped_obs=dict(obs=flipped_obs, info={"available_actions_mask":flipped_aam})
    )

    return output, action

def get_data(flags):
    ds = ConnectFourDataset('.\\pandas_replays\\stacked_replays.pkl', convert_to_obs_space)
    train_data, valid_data = random_split(ds, [flags.train_pct, flags.valid_pct])
    train_loader = DataLoader(train_data, batch_size=flags.train_bs, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=flags.valid_bs, shuffle=True)
    return train_loader, valid_loader

def train(flags):
    train_loader, valid_loader = get_data(flags)

    model = create_model(flags, device='cpu')
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Training model with {n_trainable_params} parameters")

    criterion = flags.criterion_class()
    optimizer = flags.optimizer_class(model.parameters(), **flags.optimizer_kwargs)
    scheduler = flags.lrschedule_class(optimizer, **flags.lrschedule_kwargs)

    for epoch in range(flags.epochs):
        running_loss = 0.0
        game_steps = 0
        for data in train_loader:
            obs, labels = data

            optimizer.zero_grad()

            reg_outputs = model.sample_actions(obs['obs'])
            flipped_outputs = model.sample_actions(obs['flipped_obs'])
            # print(f"reg: {reg_outputs['policy_logits']}")
            # print(f"flp: {flipped_outputs['policy_logits']}")
            outputs = torch.add(reg_outputs['policy_logits'],torch.fliplr(flipped_outputs['policy_logits'])) / 2
            # print(outputs)

            probs = F.softmax(outputs, dim=-1)

            loss = criterion(probs, labels)
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item()
            game_steps += 1

        logging.info(f"Epoch: {epoch + 1:02d} | lr: {scheduler.get_last_lr()[0]:.2e} | loss: {running_loss / game_steps:.3f}")
        scheduler.step()

    logging.info('Finished Training')

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        flags.name + ".pt",
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
        },
        flags.name + "_weights.pt"
    )

    def validate():
        correct = 0
        total = 0

        with torch.no_grad():
            for data in valid_loader:
                obs, labels = data
                labels = torch.argmax(labels, dim=-1)
                outputs = model.select_best_actions(obs['obs'])

                predicted = outputs['actions'].squeeze(-1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        logging.info(f'Accuracy of the network on validation set: {100 * correct // total} %')

    validate()