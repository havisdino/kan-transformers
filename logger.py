import torch
from torch.utils.tensorboard import SummaryWriter
from abc import ABC, abstractmethod
import logging


logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S'
)


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
    
    def log(self, **kwargs):
        info = [f'step: {self.global_step}']
        
        for tag, value in kwargs.item():
            self.add_scalar(tag, value, self.global_step)
            info.append(f'{tag}: {value:.4f}')
        
        msg = ' - '.join(info)
        logging.info(msg)
            
        self.global_step += 1
    
    def close(self):
        SummaryWriter.close(self)
