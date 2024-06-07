from typing import Iterable

import torch.nn.functional as F
from torch import nn

from .kan import KAN


class KANBlocks(nn.ModuleList):
    def __init__(
        self,
        n_blocks: int,
        layers_hidden: Iterable,
        grid_size: int,
        spline_order: int 
    ):
        super().__init__()
        
        self.n_blocks = n_blocks
        
        for _ in range(n_blocks):
            kan_block = KAN(layers_hidden, grid_size, spline_order)
            self.append(kan_block)

    def forward(self, inputs, targets):
        loss_total = 0
        
        for input, target, block in zip(inputs, targets, self):
            output = block(input)
            loss = F.huber_loss(output, target) / self.n_blocks
            loss_total += loss
        
        return loss_total
