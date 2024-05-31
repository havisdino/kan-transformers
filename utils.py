from argparse import Namespace
from torch import nn

import yaml


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


def count_params(model):
    if isinstance(model, nn.DataParallel):
        n_params = sum(p.numel() for p in model.module.parameters())
    elif isinstance(model, nn.Module):
        n_params = sum(p.numel() for p in model.parameters())
        
    print(f'Parameters: {n_params:,}')
    return n_params
