from typing import Dict, Optional, Tuple, Union

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
    def __init__(self):
        super(ConvEmbeddingInputLayer, self).__init__()

    def forward(self, x):
        x, input_mask = x
