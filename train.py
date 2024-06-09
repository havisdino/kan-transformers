import os

import torch
import torch.distributed as dist
from tokenizers import Tokenizer
from torch.nn.parallel import DistributedDataParallel as DDP

from dataset import get_data_loader
from model.gpt2 import GPT2Model
from model.kan_blocks import KANBlocks
from lr_scheduler import RectifiedLinearLR
from trainer import Trainer
from utils import Config


def setup(rank, master_addr, master_port, device_ids):
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(master_port)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, device_ids))
    
    dist.init_process_group('nccl', rank=rank, world_size=len(device_ids))


def cleanup():
    dist.destroy_process_group()


def main(rank, world_size, config):
    setup(
        rank, config.distributed.master_addr,
        config.distributed.master_port,
        config.distributed.device_ids
    )
    
    kan_blocks = KANBlocks(**vars(config.kan_blocks))
    kan_blocks = kan_blocks.to(rank)
    kan_blocks = DDP(kan_blocks, [rank], rank)
    
    gpt = GPT2Model(config.gpt)
    gpt.load_state_dict(torch.load(config.train.pretrain_path, 'cpu'))
    gpt = gpt.to(rank)
    gpt = DDP(gpt, [rank], rank)
    
    optimizer = torch.optim.Adam(kan_blocks.parameters(), lr=1.)
    scaler = torch.cuda.amp.GradScaler()
    lr_scheduler = RectifiedLinearLR(optimizer, **vars(config.train.lr))
    
    trainer = Trainer(
        kan_blocks, gpt, optimizer, scaler, lr_scheduler,
        config.train.test_interval, config.train.checkpoint_interval,
        config.train.checkpoint_retention
    )
    
    tokenizer = Tokenizer.from_pretrained('gpt2')
    
    train_loader = get_data_loader(
        rank, world_size, config.data.train_paths,
        config.data.n_tokens, config.train.batch_size, tokenizer
    )
    
    trainer.train(train_loader, config.train.n_steps)
    cleanup()
    
    
if __name__ == '__main__':
    import torch.multiprocessing as mp
    
    config = Config.from_yaml('config.yml')
    world_size = len(config.distributed.device_ids)
    mp.spawn(
        main,
        args=(world_size, config),
        nprocs=world_size
    )
