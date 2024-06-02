from dataclasses import dataclass
import json
from typing import List, Optional
from tokenizers import Tokenizer
import torch
from torch.utils.data import IterableDataset, DataLoader
import polars as pl


@dataclass(eq=False)
class CSVTextDataset(IterableDataset):
    csv_path: str
    n_tokens: int
    tokenizer: Tokenizer
    column: str = 'text'
    limit: Optional[int] = None
    n_overlap: int = 32
    end_token: str = '<|endoftext|>'
    
    def __post_init__(self):
        assert 0 <= self.n_overlap < self.n_tokens
        df = pl.read_csv(self.csv_path, columns=[self.column])
        self.df = df.select(pl.col(self.column).shuffle())
        self.ids_cache = []
        self.end_token_id = self.tokenizer.encode(self.end_token).ids[0]
    
    def generate_sequences(self):
        N = self.n_tokens
        R = self.n_overlap
        count = 0
        for row in self.df.iter_rows(named=self.column):
            text = row[self.column]
            
            while len(self.ids_cache) > N:
                yield self.ids_cache[:N]
                count += 1
                if self.limit and count == self.limit:
                    return
                self.ids_cache = self.ids_cache[N - R:]
            
            ids = self.tokenizer.encode(text).ids
            ids.append(self.end_token_id)
            self.ids_cache.extend(ids)
            
    def __iter__(self):
        return self.generate_sequences()


@dataclass(eq=False)
class DistributedJsonlTextDataset(IterableDataset):
    rank: int
    world_size: int
    file_paths: List[str]
    n_tokens: int
    tokenizer: Tokenizer
    key: str = 'text'
    n_overlap: int = 32
    end_token: str = '<|endoftext|>'
    
    def __post_init__(self):
        assert 0 <= self.n_overlap < self.n_tokens
        self.ids_cache = []
        self.end_token_id = self.tokenizer.encode(self.end_token).ids[0]
        self.count = 0
    
    def generate_sequences(self):
        N = self.n_tokens
        R = self.n_overlap
        
        for file_path in self.file_paths:
            with open(file_path) as file:                    
                for line in file:
                    text = json.loads(line)[self.key]
                    while len(self.ids_cache) > N:
                        if self.count >= self.rank and (self.count - self.rank) % self.world_size == 0:
                            yield self.ids_cache[:N]
                        self.count += 1
                        self.ids_cache = self.ids_cache[N - R:]
                    
                    ids = self.tokenizer.encode(text).ids
                    ids.append(self.end_token_id)
                    self.ids_cache.extend(ids)
    
    def __iter__(self):
        return self.generate_sequences()
   

def collate_fn(batch):
    return torch.tensor(batch, dtype=torch.int32)


def get_train_loader(rank, world_size, train_data_path, n_tokens, batch_size, tokenizer):
    train_dataset = DistributedJsonlTextDataset(rank, world_size, train_data_path, n_tokens, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size, collate_fn=collate_fn, drop_last=True)
    return train_loader
