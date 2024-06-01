from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from logger import TensorBoardLogger
from utils import save_checkpoint


@dataclass
class Trainer:
    kan_blocks: nn.Module
    gpt: nn.Module
    optimizer: torch.optim.Optimizer
    scaler: torch.cuda.amp.GradScaler
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None
    test_interval: int
    checkpoint_interval: int
    checkpoint_retention: int
        
    def __post_init__(self):
        maxlen = 1024
        causal_mask = torch.ones(maxlen, maxlen, device='cuda').tril() == 0
        self.causal_mask = torch.where(causal_mask, -float('inf'), 0)
        self.logger = TensorBoardLogger('logs')
        
        self.epoch = 1
    
    def _get_attention_mask(self, n_positions):
        return self.causal_mask[None, :n_positions, :n_positions]
    
    def get_loss_distill(self, inputs, targets):    
        return self.kan_blocks(inputs, targets)
    
    def get_loss(self, kan_inputs, kan_targets):
        with torch.autocast('cuda', torch.float16):
            loss = self.get_loss_distill(kan_inputs, kan_targets)
        return loss
    
    def train_step(self, kan_inputs, kan_targets):
        loss = self.get_loss(kan_inputs, kan_targets)
        
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            
        return loss.detach().item()
    
    @torch.no_grad()
    def evaluate(self, test_loader):
        test_losses = []
        for input_ids in test_loader:
            input_ids = input_ids.cuda()
            
            outputs = self.gpt_forward(input_ids)
            
            kan_inputs = outputs['kan_inputs']
            kan_targets = outputs['kan_targets']
            
            loss = self.get_loss(kan_inputs, kan_targets)
            test_losses.append(loss.item())
        
        test_loss = sum(test_losses) / len(test_losses)
        print(f'test_loss: {test_loss}')
        return test_loss
    
    @torch.no_grad() 
    def gpt_forward(self, input_ids):
        self.gpt.eval()
        n_positions = input_ids.size(1)
        with torch.autocast('cuda', torch.float16):
            attention_mask = self._get_attention_mask(n_positions)
            outputs = self.gpt(input_ids, attention_mask)
        return outputs
            
    def train(self, rank, train_loader, test_loader, n_steps):
        data_iter = iter(train_loader)
        # train_loader.sampler.set_epoch(self.epoch)
        # test_loader.sampler.set_epoch(self.epoch)
        
        for step in range(1, 1 + n_steps):
            try:
                input_ids = next(data_iter)
            except StopIteration:
                self.epoch += 1
                train_loader.sampler.set_epoch(self.epoch)
                test_loader.sampler.set_epoch(self.epoch)
                data_iter = iter(train_loader)
                input_ids = next(data_iter)
            input_ids = input_ids.to(rank)
            
            outputs = self.gpt_forward(input_ids)
                
            kan_inputs = outputs['kan_inputs']
            kan_targets = outputs['kan_targets']
            
            train_loss = self.train_step(kan_inputs, kan_targets)
            
            if step % self.test_interval == 0:
                test_loss = self.evaluate(test_loader)
                self.logger.log(train_loss=train_loss, test_loss=test_loss)
            else:
                self.logger.log(train_loss=train_loss)
                
            if step % self.checkpoint_interval == 0:
                save_checkpoint(
                    self.kan_blocks, self.optimizer, self.scaler, self.lr_scheduler,
                    step, self.checkpoint_retention
                )
            dist.barrier()
