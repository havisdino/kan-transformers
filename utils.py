import yaml
from argparse import Namespace


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
