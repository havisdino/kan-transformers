import torch
from torch.utils.tensorboard import SummaryWriter
from abc import ABC, abstractmethod
from tqdm.auto import tqdm


class Logger(ABC):
    @abstractmethod
    def log(self, **kwargs):
        pass
    
    @abstractmethod
    def close(self):
        pass
    

class TensorBoardLogger(Logger, SummaryWriter):
    def __init__(self, log_dir=None, comment="", purge_step=None, max_queue=10, flush_secs=120, filename_suffix=""):
        super().__init__(log_dir, comment, purge_step, max_queue, flush_secs, filename_suffix)
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        self.global_step = 1
        
        self.pbar = tqdm()
    
    def log(self, **kwargs):        
        for tag, value in kwargs.items():
            self.add_scalar(tag, value, self.global_step)
        
        self.pbar.set_postfix(**kwargs)
        self.pbar.update()    
        
        self.global_step += 1
    
    def close(self):
        SummaryWriter.close(self)
