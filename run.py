from tokenizers import Tokenizer
import torch
from dataset import get_loaders
from gpt2 import GPT2Model
from kan_blocks import KANBlocks
from lr_scheduler import RectifiedLinearLR
from trainer import Trainer
from utils import Config


def main(config):
    kan_blocks = KANBlocks(**vars(config.kan_blocks))
    kan_blocks = kan_blocks.cuda()
    
    gpt = GPT2Model(config.gpt)
    gpt = gpt.cuda()
    
    optimizer = torch.optim.Adam(kan_blocks.parameters(), lr=1.)
    scaler = torch.cuda.amp.GradScaler()
    lr_scheduler = RectifiedLinearLR(optimizer, **vars(config.train.lr))
    trainer = Trainer(kan_blocks, gpt, optimizer, scaler, lr_scheduler, config.train.test_interval)
    
    tokenizer = Tokenizer.from_pretrained('gpt2')
    
    train_loader, test_loader = get_loaders(
        train_data_path=config.data.train_path,
        test_data_path=config.data.test_path,
        batch_size=config.train.batch_size,
        tokenizer=tokenizer
    )
    
    trainer.train(train_loader, test_loader, config.train.n_steps)
    
    
if __name__ == '__main__':    
    config = Config.from_yaml('config.yml')
    main(config)
