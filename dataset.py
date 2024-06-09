import json
from dataclasses import dataclass
from typing import List

import torch
from tokenizers import Tokenizer
from torch.utils.data import IterableDataset, DataLoader


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


def collate_fn_lm(batch):
    batch = torch.tensor(batch, dtype=torch.long)
    input_ids = batch[:, :-1]
    target_ids = batch[:, 1:]
    return input_ids, target_ids


def get_data_loader(rank, world_size, data_paths, n_tokens, batch_size, tokenizer, n_overlap=32, lm=False):
    n_tokens = n_tokens + 1 if lm else n_tokens
    train_dataset = DistributedJsonlTextDataset(
        rank, world_size, data_paths, n_tokens, tokenizer, n_overlap=n_overlap
    )
    cfn = collate_fn_lm if lm else collate_fn
    train_loader = DataLoader(train_dataset, batch_size, collate_fn=cfn, drop_last=True)
    return train_loader
