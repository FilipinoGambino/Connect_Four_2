from typing import Dict, Optional, Tuple, Union

import gym.spaces
import torch
from torch import nn

class DictInputLayer(nn.Module):
    @staticmethod
    def forward(
            x: Dict[str, Union[Dict, torch.Tensor]]
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor], Optional[torch.Tensor]]:
        return (x["obs"],
                x["info"]["input_mask"],
                x["info"]["available_actions_mask"],
                x["info"].get("subtask_embeddings", None))

class ConvEmbeddingInputLayer(nn.Module):
    def __init__(
            self,
            obs_space,
            embedding_dim,
            out_dim,
            n_merge_layers,
    ):
        super(ConvEmbeddingInputLayer, self).__init__()
        cont_embs = []
        disc_embs = []
        embeddings = dict()
        for key, val in obs_space.spaces.items():
            if isinstance(val, gym.spaces.MultiBinary):
                embeddings = nn.Embedding(2, embedding_dim)
            elif isinstance(val, gym.spaces.Box):
                print(val.shape)
            else:
                raise NotImplementedError(f"{val} is not an accepted observation space.")


    def forward(self, x):
        x, input_mask = x

if __name__=="__main__":
    import numpy as np
    from connectx.connectx_gym import obs_spaces

    o = obs_spaces.FixedShapeContinuousObs()
    spaces = o.get_obs_spec()
    for key,val in spaces.items():
        shape = val.shape
        print(f"{key}: {val} with shape {np.prod(shape)}")