import os
from tokenizers import Tokenizer
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from dataset import get_loaders
from gpt2 import GPT2Model
from kan_blocks import KANBlocks
from lr_scheduler import RectifiedLinearLR
from trainer import Trainer
from utils import Config


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    dist.init_process_group('nccl', rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def main(rank, world_size, config):
    kan_blocks = KANBlocks(**vars(config.kan_blocks))
    kan_blocks = kan_blocks.to(rank)
    kan_blocks = DDP(kan_blocks, [rank], rank, find_unused_parameters=True)
    
    gpt = GPT2Model(config.gpt)
    gpt = gpt.to(rank)
    gpt = DDP(gpt, [rank], rank, find_unused_parameters=True)
    
    
    optimizer = torch.optim.Adam(kan_blocks.parameters(), lr=1.)
    scaler = torch.cuda.amp.GradScaler()
    lr_scheduler = RectifiedLinearLR(optimizer, **vars(config.train.lr))
    
    trainer = Trainer(
        kan_blocks, gpt, optimizer, scaler, lr_scheduler,
        config.train.test_interval, config.train.checkpoint_interval,
        config.train.checkpoint_retention
    )
    
    tokenizer = Tokenizer.from_pretrained('gpt2')
    
    train_loader, test_loader = get_loaders(
        rank, world_size, config.data.train_path, config.data.test_path,
        config.data.n_tokens, config.train.batch_size, tokenizer
    )
    
    trainer.train(train_loader, test_loader, config.train.n_steps)
    cleanup()
    
    
if __name__ == '__main__':
    import torch.multiprocessing as mp
    
    config = Config.from_yaml('config.yml')

    world_size = config.train.world_size
    
    mp.spawn(
        main,
        args=(world_size, config),
        nprocs=world_size
    )
