from tokenizers import Tokenizer
import torch
from torch.utils.data import DataLoader
from dataset import CSVTextDataset
from gpt2 import GPT2Model
from kan_blocks import KANBlocks
from trainer import Trainer
from utils import Config


def get_loaders(train_data_path, test_data_path, batch_size, tokenizer):
    train_dataset = CSVTextDataset(train_data_path, 1024, tokenizer)
    test_dataset = CSVTextDataset(test_data_path, 1024, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size, num_workers=2, prefetch_factor=2)
    test_loader = DataLoader(test_dataset, batch_size, num_workers=2, prefetch_factor=2)
    
    return train_loader, test_loader


def main(config):
    kan_blocks = KANBlocks(**vars(config.kan_blocks))
    kan_blocks = kan_blocks.cuda()
    
    gpt = GPT2Model(config.gpt)
    gpt = gpt.cuda()
    
    optimizer = torch.optim.AdamW(kan_blocks.parameters(), config.train.learning_rate)
    scaler = torch.cuda.amp.GradScaler()
    lr_scheduler = None     # need to be reviewed
    trainer = Trainer(kan_blocks, gpt, optimizer, scaler, lr_scheduler)
    
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