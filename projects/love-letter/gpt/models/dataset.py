import random
import re
import torch
from torch.utils.data import Dataset
import os
from typing import List, Tuple
from .tokenizer import LoveLetterTokenizer
import yaml

class LoveLetterDataset(Dataset):
    def __init__(self, tokenizer: LoveLetterTokenizer, config_path):
        self.tokenizer = tokenizer
        
        # Load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.seq_length = config['model']['seq_length']

        max_logs = config['data']['max_logs'] if 'max_logs' in config['data'] else None
        data_dir = config['data']['data_dir']
        self.config = config['data']
        count = 0
        
        self.examples: List[List[int]] = []
        for filename in os.listdir(data_dir):
            if filename.endswith('.log'):
                with open(os.path.join(data_dir, filename), 'r') as f:
                    text = f.read()
                    lines = text.split('\n')
                    lines = lines[4:]

                    if (random.random() < 2.0 / 3.0 and config['data']['type'] == 'mixed') or config['data']['type'] == 'pov':
                        # Randomly choose which player's hidden info to remove
                        remove_p1_hidden = random.random() < 0.5
                        my_player = 'p2' if remove_p1_hidden else 'p1'
                        filtered_lines = []
                        
                        # Filter out hidden information based on random choice
                        for line in lines:
                            if remove_p1_hidden:
                                if '|p1|hidden' not in line:
                                    filtered_lines.append(line)
                            else:
                                if '|p2|hidden' not in line:
                                    filtered_lines.append(line)
                        lines = filtered_lines
                    log = '\n'.join(lines)
                    tokens = self.tokenizer.tokenize(log)

                    if tokens:  # Only add non-empty sequences
                        self.examples.append(tokens)
                    count += 1
            if (max_logs is not None) and (count >= max_logs):
                break

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = self.examples[idx]
        
        # Create input and target sequences
        if len(tokens) > self.seq_length + 1:
            tokens = tokens[-(self.seq_length + 1):]

        pad_length = self.seq_length - len(tokens) + 1
        padded_tokens = tokens + [self.tokenizer.special_tokens['PAD']] * pad_length # [seq_length + 1]
        
        # x: everything except the last token
        x = torch.tensor(padded_tokens[:-1], dtype=torch.long)  # [seq_length]
        # y: everything except the first token
        y = torch.tensor(padded_tokens[1:], dtype=torch.long)   # [seq_length]
        
        
        return x, y
    
    def get_padding_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        """Create padding mask where 1 indicates non-pad tokens."""
        # [seq_length]
        padding_mask = (tokens != self.tokenizer.special_tokens['PAD'])
        # For attention, we'd typically expand dims: [1, 1, seq_len, seq_len].
        return padding_mask