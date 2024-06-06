import os
from argparse import ArgumentParser

import torch
import yaml

from kan_blocks import KANBlocks
from kangpt import KANGPTLMHeadModel
from utils import Config


def main(config, kan_ckp, gpt_ckp, lmhead_ckp):
    kangpt = KANGPTLMHeadModel(config.gpt)
    kangpt.lm_head.load_state_dict(torch.load(lmhead_ckp, 'cpu'))
    
    kan_blocks = KANBlocks(**vars(config.kan_blocks))
    kan_blocks.load_state_dict(torch.load(kan_ckp, 'cpu')['kan_blocks'])    
    
    kangpt.transformer.load_state_dict(torch.load(gpt_ckp, 'cpu'), strict=False)
    for kangpt_block, kan_block in zip(kangpt.transformer.h, kan_blocks):
        kangpt_block.kan = kan_block
        
    save_dir = './pretrained'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'kangpt_lmhead.pt')
    
    torch.save(kangpt.state_dict(), save_path)
    
    with open('pretrained/kangpt_lmhead_config.yml', 'w') as file:
        yaml.dump(config.gpt.to_dict(), file)
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--kan', type=str, required=True)
    parser.add_argument('--gpt', type=str, required=True)
    parser.add_argument('--lmhead', type=str, required=True)
    
    args = parser.parse_args()
    
    config = Config.from_yaml('config.yml')
    setattr(config.gpt, 'kan_grid_size', config.kan_blocks.grid_size)
    setattr(config.gpt, 'kan_spline_order', config.kan_blocks.spline_order)
    setattr(config.gpt, 'kan_layers_hidden', config.kan_blocks.layers_hidden)
    
    main(config, args.kan, args.gpt, args.lmhead)
