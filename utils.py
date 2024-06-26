import os
from argparse import Namespace

import torch
import yaml
from torch import nn


class Config(Namespace):
    @staticmethod
    def from_dict(config_dict):
        config = Config()
        for key, value in config_dict.items():
            if isinstance(value, dict):
                value = Config.from_dict(value)
            setattr(config, key, value)
        return config
    
    @staticmethod
    def from_yaml(path):
        with open(path) as file:
            config_dict = yaml.safe_load(file)
        return Config.from_dict(config_dict)
    
    def to_dict(self):
        def _to_dict(config):
            config_dict = vars(config)
            for key, value in config_dict.items():
                if isinstance(value, Config):
                    config_dict[key] = _to_dict(value)
            return config_dict
        return _to_dict(self)


def count_params(model):
    if isinstance(model, nn.DataParallel):
        n_params = sum(p.numel() for p in model.module.parameters())
    elif isinstance(model, nn.Module):
        n_params = sum(p.numel() for p in model.parameters())
        
    print(f'Parameters: {n_params:,}')
    return n_params


def save_checkpoint(kan_blocks, optimizer, scaler, lr_scheduler, step, cp_interval, retention):
    kan_state_dict = kan_blocks.module.state_dict()
    
    checkpoint = dict(
        kan_blocks=kan_state_dict,
        optimizer=optimizer.state_dict(),
        scaler=scaler.state_dict(),
        lr_scheduler=lr_scheduler.state_dict()
    )
    
    dir = './checkpoints'
    if not os.path.exists(dir):
        os.makedirs(dir)
    
    file_to_remove = f'kanblocks_{step - cp_interval * retention}.pt'
    path_to_remove = os.path.join(dir, file_to_remove)
    if os.path.exists(path_to_remove):
        os.remove(path_to_remove)
    
    file_name = f'kanblocks_{step}.pt'
    save_path = os.path.join(dir, file_name)
    
    torch.save(checkpoint, save_path)
