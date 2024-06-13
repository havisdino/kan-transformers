from tokenizers import Tokenizer
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from argparse import ArgumentParser
from tqdm import tqdm
import os

from .zeroshot_data import get_data_loader
from model.gpt2 import GPT2LMHeadModel
from model.kangpt import KANGPTLMHeadModel
from utils import Config


@torch.no_grad()
def evaluate(model, data_loader):
    model.eval()
    
    pbar = tqdm(desc='eval') if dist.get_rank() == 0 else None
    
    n_true_preds = torch.zeros([])
    
    for i, (inputs, target) in enumerate(data_loader, 1):
        assert isinstance(inputs, dict)
        assert 'input_ids' in inputs.keys()
        assert 'attention_mask' in inputs.keys()
        assert isinstance(target, int)
        
        inputs = inputs.to(dist.get_rank())
        
        with torch.autocast('cuda', torch.float16):
            logits = model(**inputs)
            log_probs = logits.log_softmax(-1)
            indices = torch.tensor([[target]]).repeat(inputs.size(0), 1)
            log_probs = log_probs.gather(-1, indices)
            choice = log_probs.argmax(0)
            
            if choice == target:
                n_true_preds += 1.
            
            acc = n_true_preds / i
            dist.all_reduce(acc, dist.ReduceOp.AVG)
            
            if pbar is not None:
                pbar.set_postfix(f'acc: {acc.item()}')
                pbar.update()
            
    return acc


def setup(rank, master_addr, master_port, device_ids: str):
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    os.environ['CUDA_VISIBLE_DEVICES'] = device_ids
    dist.init_process_group('nccl', rank=rank, world_size=len(device_ids.split(',')))


def main(rank, args):
    setup(rank, args.master_addr, args.master_port, args.device_ids)
    
    config = Config.from_yaml(args.config)
    
    if args.type == 'kangpt':
        model = KANGPTLMHeadModel(config)
    elif args.type == 'gpt':
        model = GPT2LMHeadModel(config)
        
    model.load_state_dict(torch.load(args.model, 'cpu'))    
    model.to(rank)
    model = DDP(model)
    
    tokenizer = Tokenizer.from_pretrained('gpt2')
    
    data_loader = get_data_loader(
        rank,
        world_size=len(args.device_ids.split(',')),
        data_paths=(args.data,),
        n_tokens=1024,
        tokenizer=tokenizer
    )
    
    acc = evaluate(model, data_loader)
    
    if rank == 0:
        print(f'Zero-shot accuracy: {acc}')
    
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--type', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--master-addr', type=str, default='localhost')
    parser.add_argument('--master-port', type=str, default='9999')
    parser.add_argument('--device-ids', type=str, default='0')
    
    args = parser.parse_args()
    
    mp.spawn(main, (args,), nprocs=len(args.device_ids.split(',')))
    