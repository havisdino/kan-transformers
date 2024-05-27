from dataclasses import dataclass
from typing import Optional
from tokenizers import Tokenizer
import torch
from torch.utils.data import IterableDataset, Dataset
from torchtext import transforms
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
        self.end_token_id = self.tokenizer.encode(self.end_token)
    
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
            

def collate_fn(input_ids):
    target_ids = [item[1:] for item in input_ids]
    input_ids = [item[:-1] for item in input_ids]
    input_ids = transforms.F.to_tensor(input=input_ids, dtype=torch.int32)
    target_ids = transforms.F.to_tensor(input=target_ids, dtype=torch.long)
    return input_ids, target_ids
