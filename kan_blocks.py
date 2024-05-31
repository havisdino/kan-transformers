from torch import nn
import torch.nn.functional as F
from kan import KAN


class KANBlocks(nn.ModuleList):
    def __init__(
        self,
        n_blocks: int,
        layers_hidden: list | tuple,
        grid_size: int,
        spline_order: int 
    ):
        super().__init__()
        
        self.n_blocks = n_blocks
        
        for _ in range(n_blocks):
            kan_block = KAN(layers_hidden, grid_size, spline_order)
            self.append(kan_block)
    
    def loss_distill(self, inputs, targets):
        assert len(inputs) == len(targets) == self.n_blocks
        
        loss = 0
        
        for input, target, block in zip(inputs, targets, self):
            output = block(input)
            loss += F.mse_loss(output, target) / self.n_blocks
        
        return loss 
    