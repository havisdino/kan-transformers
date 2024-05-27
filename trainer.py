from dataclasses import dataclass

import torch
from tqdm import tqdm

from gpt2 import GPT2Model
from kan_blocks import KANBlocks
from logger import TensorBoardLogger


@dataclass
class Trainer:
    kan_blocks: KANBlocks
    gpt: GPT2Model
    optimizer: torch.optim.Optimizer
    scaler: torch.cuda.amp.GradScaler
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None
    test_interval: int = 10
        
    def __post_init__(self):
        maxlen = 1024
        causal_mask = torch.ones(maxlen, maxlen, device='cuda').tril() == 0
        self.causal_mask = torch.where(causal_mask, -float('inf'), 0)
        self.logger = TensorBoardLogger('logs')
    
    def _get_attention_mask(self, n_positions):
        return self.causal_mask[None, :n_positions, :n_positions]
    
    def get_loss(self, kan_inputs, kan_targets):
        with torch.autocast('cuda', torch.float16):
            loss = self.kan_blocks.loss_distill(kan_inputs, kan_targets)
        return loss
    
    def train_step(self, kan_inputs, kan_targets):
        self.kan_blocks.train()
        
        loss = self.get_loss(kan_inputs, kan_targets)
        
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            
        return loss.detach().item()
    
    @torch.no_grad()
    def evaluate(self, test_loader):
        self.kan_blocks.train(False)
        
        test_losses = []
        for input_ids in tqdm(test_loader, desc='eval', leave=False):
            input_ids = input_ids.cuda()
            
            outputs = self.gpt_forward(input_ids)
            
            kan_inputs = outputs['kan_inputs']
            kan_targets = outputs['kan_targets']
            
            loss = self.get_loss(kan_inputs, kan_targets)
            test_losses.append(loss.item())
        
        return sum(test_losses) / len(test_losses)
    
    @torch.no_grad() 
    def gpt_forward(self, input_ids):
        self.gpt.eval()
        n_positions = input_ids.size(1)
        with torch.autocast('cuda', torch.float16):
            attention_mask = self._get_attention_mask(n_positions)
            outputs = self.gpt(input_ids, attention_mask)
        return outputs
            
    def train(self, train_loader, test_loader, n_steps=1):
        data_iter = iter(train_loader)
        
        for step in range(1, 1 + n_steps):
            try:
                input_ids = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                input_ids = next(data_iter)
            input_ids = input_ids.cuda()
            
            outputs = self.gpt_forward(input_ids)
                
            kan_inputs = outputs['kan_inputs']
            kan_targets = outputs['kan_targets']
            
            train_loss = self.train_step(kan_inputs, kan_targets)
            
            if step % self.test_interval:
                test_loss = self.evaluate(test_loader)
                self.logger.log(train_loss=train_loss, test_loss=test_loss)
            else:
                self.logger.log(train_loss=train_loss)
