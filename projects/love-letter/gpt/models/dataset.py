import random
from more_itertools import padded
import torch
from torch.utils.data import Dataset
import os
from typing import List, Tuple
from .tokenizer import SPECIAL_TOKENS, LoveLetterTokenizer
from .config_types import ModelConfig, DataConfig

class LoveLetterDataset(Dataset):
    def __init__(self, tokenizer: LoveLetterTokenizer, data_config: DataConfig, model_config: ModelConfig):
        self.tokenizer = tokenizer
        self.seq_length = model_config['seq_length']

        max_logs = data_config.get('max_logs')
        data_dir = data_config['data_dir']
        self.config = data_config
        count = 0

        
        self.examples: List[tuple[List[int], bool, List[int], List[int]]] = []
        for filename in os.listdir(data_dir):
            if filename.endswith('.log'):
                with open(os.path.join(data_dir, filename), 'r') as f:
                    text = f.read()
                    lines = text.split('\n')
                    lines = lines[4:]

                    victory = False
                    guess = '1'
                    my_card = '1'
                    guesses = []
                    my_cards = []
                    if (random.random() < 2.0 / 3.0 and data_config['type'] == 'mixed') or data_config['type'] == 'pov':
                        # Randomly choose which player's hidden info to remove
                        remove_p1_hidden = random.random() < 0.5
                        my_player = 'p2' if remove_p1_hidden else 'p1'
                        filtered_lines = []
                        
                        # Filter out hidden information based on random choice
                        for line in lines:
                            if 'info' in line:
                                # Update what needs to be guessed
                                if my_player == 'p2':
                                    guess = line.split('|')[3]
                                    my_card = line.split('|')[4]
                                if my_player == 'p1':
                                    guess = line.split('|')[4]
                                    my_card = line.split('|')[3]
                                continue
                            if remove_p1_hidden:
                                if '|p1|hidden' not in line:
                                    filtered_lines.append(line)
                                    guesses.extend([SPECIAL_TOKENS[guess]] * len(line.split('|')))
                                    my_cards.extend([SPECIAL_TOKENS[my_card]] * len(line.split('|')))
                            else:
                                if '|p2|hidden' not in line:
                                    filtered_lines.append(line)
                                    guesses.extend([SPECIAL_TOKENS[guess]] * len(line.split('|')))
                                    my_cards.extend([SPECIAL_TOKENS[my_card]] * len(line.split('|')))
                            if '|end|p1|win' in line:
                                victory = (my_player == 'p1')
                            if '|end|p2|win' in line:
                                victory = (my_player == 'p2')

                        lines = filtered_lines
                    log = '\n'.join(lines)
                    tokens = self.tokenizer.tokenize(log)

                    if tokens:  # Only add non-empty sequences
                        self.examples.append((tokens, victory, guesses, my_cards))
                    count += 1
            if (max_logs is not None) and (count >= max_logs):
                break

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        tokens, victory, guesses, my_cards = self.examples[idx]

        assert len(tokens) == len(guesses)
        assert len(tokens) == len(my_cards)
        
        # Create input and target sequences
        if len(tokens) > self.seq_length + 1:
            tokens = tokens[-(self.seq_length + 1):]

        pad_length = self.seq_length - len(tokens) + 1
        padded_tokens = tokens + [self.tokenizer.special_tokens['PAD']] * pad_length # [seq_length + 1]
        padded_guesses = guesses + [self.tokenizer.special_tokens['PAD']] * (self.seq_length - len(guesses)) # [seq_length]
        padded_my_cards = my_cards + [self.tokenizer.special_tokens['PAD']] * (self.seq_length - len(my_cards)) # [seq_length]
        
        # x: everything except the last token
        x = torch.tensor(padded_tokens[:-1], dtype=torch.long)  # [seq_length]
        # y: everything except the first token 
        y = torch.tensor(padded_tokens[1:], dtype=torch.long)   # [seq_length]
        # y' for the value head
        y_value = torch.tensor([1 if victory else -1] * self.seq_length, dtype=torch.float32)  # [1]

        # y_guesses: guesses for the opponent's hand
        y_guesses = torch.tensor(padded_guesses, dtype=torch.long)  # [seq_length]

        # y_my_cards: guesses for my current hand
        y_my_cards = torch.tensor(padded_my_cards, dtype=torch.long)  # [seq_length]



        
        
        return x, y, y_value, y_guesses, y_my_cards
    
    def get_padding_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        """Create padding mask where 1 indicates non-pad tokens."""
        # [seq_length]
        padding_mask = (tokens != self.tokenizer.special_tokens['PAD'])
        # For attention, we'd typically expand dims: [1, 1, seq_len, seq_len].
        return padding_mask