from torch import nn
import torch.nn.functional as F
from kan import KAN


class KAN2D(KAN):    
    def forward(self, x):
        B, L, D = x.size()
        x = x.view(-1, D)
        output = super().forward(x)
        output = output.view(B, L, output.size(-1))
        return output


class KANBlocks(nn.ModuleList):
    def __init__(
        self,
        n_blocks: int,
        width: list,
        grid: int = 3,
        k: int = 3
    ):
        super().__init__()
        
        self.n_blocks = n_blocks
        self.d_in = width[0]
        self.d_out = width[-1]
        
        for _ in range(n_blocks):
            kan_block = KAN2D(width, grid, k, device='cuda')
            self.append(kan_block)
    
    def loss_distill(self, inputs, targets):
        assert len(inputs) == len(targets) == self.n_blocks
        
        loss = 0
        
        for input, target, block in zip(inputs, targets, self):
            output = block(input)
            loss += F.mse_loss(output, target)
        
        return loss
    