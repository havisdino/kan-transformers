import json
from dataclasses import dataclass
from typing import List

import torch
from torch.nn.utils.rnn import pad_sequence
from tokenizers import Tokenizer
from torch.utils.data import IterableDataset, DataLoader


@dataclass(eq=False)
class DistributedZeroShotDataset(IterableDataset):
    rank: int
    world_size: int
    file_paths: List[str]
    n_tokens: int
    tokenizer: Tokenizer
    eos_token: str = '<|endoftext|>'
    
    def __post_init__(self):
        self.count = 0
        self.eos_token_id = self.tokenizer.encode(self.eos_token).ids[0]
    
    def generate_sequences(self):
        for file_path in self.file_paths:
            with open(file_path) as file:
                for line in file:
                    if self.count >= self.rank and (self.count - self.rank) % self.world_size == 0:
                        data = json.loads(line)
                        prompt = data['prompt']
                        classes = data['classes']
                        target = data['target']
                        
                        prompt_ids = self.tokenizer.encode(prompt).ids
                        class_ids = [self.tokenizer.encode(c).ids for c in classes]
                        
                        input_ids = [torch.tensor((prompt_ids + ids)[:self.n_tokens]) for ids in class_ids]
                        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.eos_token_id)
                        
                        attention_mask = torch.where(input_ids == self.eos_token_id, 0, 1)
                        
                        yield dict(input_ids=input_ids, attention_mask=attention_mask), target

                    count += 1
    
    def __iter__(self):
        return self.generate_sequences()
   

def collate_fn(batch):
    inputs, target = batch
    return inputs, target


def get_data_loader(rank, world_size, data_paths, n_tokens, tokenizer):
    dataset = DistributedZeroShotDataset(rank, world_size, data_paths, n_tokens, tokenizer)
    return DataLoader(dataset, collate_fn=collate_fn)
