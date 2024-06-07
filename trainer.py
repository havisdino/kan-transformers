from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn

from logger import TensorBoardLogger
from utils import save_checkpoint


@dataclass
class Trainer:
    kan_blocks: nn.Module
    gpt: nn.Module
    optimizer: torch.optim.Optimizer
    scaler: torch.cuda.amp.GradScaler
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler
    test_interval: int
    checkpoint_interval: int
    checkpoint_retention: int
        
    def __post_init__(self):
        if dist.get_rank() == 0:
            self.logger = TensorBoardLogger('logs')        
        self.epoch = 1
    
    def train_step(self, kan_inputs, kan_targets):
        self.kan_blocks.train()
        with torch.autocast('cuda', torch.float16):
            loss = self.kan_blocks(kan_inputs, kan_targets)
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.lr_scheduler.step()
            
        return loss.item()
    
    @torch.no_grad() 
    def gpt_forward(self, input_ids):
        self.gpt.eval()
        n_positions = input_ids.size(1)
        with torch.autocast('cuda', torch.float16):
            outputs = self.gpt(input_ids)
        return outputs
            
    def train(self, train_loader, n_steps):
        if dist.get_rank() == 0:
            self.logger.set_n_steps(n_steps)
        
        data_iter = iter(train_loader)
        
        for step in range(1, 1 + n_steps):
            try:
                input_ids = next(data_iter)
            except StopIteration:
                self.epoch += 1
                data_iter = iter(train_loader)
                input_ids = next(data_iter)
                
            input_ids = input_ids.to(dist.get_rank())
            
            outputs = self.gpt_forward(input_ids)
                
            kan_inputs = outputs['kan_inputs']
            kan_targets = outputs['kan_targets']
            
            train_loss = self.train_step(kan_inputs, kan_targets)
            
            if dist.get_rank() == 0:
                self.logger.log(
                    self.epoch, train_loss=train_loss,
                    lr=self.optimizer.param_groups[0]['lr']
                )
              
                if step % self.checkpoint_interval == 0:
                    save_checkpoint(
                        self.kan_blocks, self.optimizer, self.scaler, self.lr_scheduler,
                        step, self.checkpoint_interval, self.checkpoint_retention
                    )
            dist.barrier()
