from typing import Callable, Dict, Optional, Tuple, Union

import gym.spaces
import torch
from torch import nn
import numpy as np

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
            activation: Callable = nn.LeakyReLU,
    ):
        super(ConvEmbeddingInputLayer, self).__init__()
        cont_channels = 0
        emb_channels = 0
        embeddings = dict()
        self.keys_to_op = dict()

        for key,val in obs_space.spaces.items():
            if isinstance(val, gym.spaces.MultiBinary):
                n_embeddings = 2
                self.keys_to_op[key] = "embedding"
                embeddings[key] = nn.Embedding(n_embeddings, embedding_dim)
                emb_channels += n_embeddings
            elif isinstance(val, gym.spaces.Box):
                cont_channels += np.prod(val.shape[:2])
                self.keys_to_op[key] = "continuous"
            else:
                raise NotImplementedError(f"{val} is not an accepted observation space.")

        self.embeddings = nn.ModuleDict(embeddings)
        cont_embs = [
            nn.Conv2d(cont_channels, out_dim, (1,1)),
            activation()
        ]
        emb_merge = [
            nn.Conv2d(emb_channels, out_dim, (1,1)),
            activation()
        ]
        merger_layers = nn.Conv2d(out_dim * 2, out_dim, (1, 1))
        self.continuous_space_embedding = nn.Sequential(*cont_embs)
        self.embedding_merger = nn.Sequential(*emb_merge)
        self.merger = nn.Sequential(merger_layers)

    def forward(self, x):
        x, input_mask = x
        cont_outs = []
        emb_outs = dict()
        for key,op in self.keys_to_op.items():
            in_tensor = x[key]
            if op == "embedding":
                emb_outs[key] = self.embeddings[key](in_tensor)
            elif op == "continuous":
                cont_outs.append(in_tensor)
            else:
                raise RuntimeError(f"Unknown operation: {op}")
        continuous_outs = self.continuous_space_embedding(torch.cat(cont_outs, dim=1))
        embedding_outs = self.embedding_merger(torch.cat([emb_tensor for emb_tensor in emb_outs.values()], dim=1))
        merged_outs = self.merger(torch.cat([continuous_outs, embedding_outs], dim=1))
        return merged_outs, input_mask