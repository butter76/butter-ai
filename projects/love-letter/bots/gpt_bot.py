#!/usr/bin/env python3

import random
from sre_parse import SPECIAL_CHARS
import sys
from typing import Any
from dataclasses import dataclass

import torch
torch.set_default_dtype(torch.bfloat16)
torch.set_printoptions(profile="full")
from gpt.models.gpt_ll import LoveLetterTransformer
from gpt.models.tokenizer import SPECIAL_TOKENS, LoveLetterTokenizer
from utils.bot import GameState, LoveLetterBot, Move, Hand, get_legal_actions
from common.config import BaseConfig


@dataclass
class GPTBotConfig(BaseConfig):
    """Configuration class for GPTBot with debug option."""
    debug: bool = False
    checkpoint: str = "./gpt/checkpoints/non-exist/model_epoch_40.pt"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    temp: float = 1
    depth: int = 20
    N: int = 128


class GPTBot(LoveLetterBot):
    """A bot that moves based on GPT-generated game logs and its value function
    """
    def __init__(self, config: GPTBotConfig):
        """Initialize the GPTBot with configuration.
        
        Args:
            config (GPTBotConfig): Configuration object containing bot settings
        """
        self.config = config

        # Initialize tokenizer and model
        tokenizer = LoveLetterTokenizer()
        self.tokenizer = tokenizer
        
        # Load checkpoint
        checkpoint_path = config.checkpoint
        checkpoint = torch.load(checkpoint_path)

        # Initialize model
        model = LoveLetterTransformer(
            vocab_size=tokenizer.vocab_size,
            model_config=checkpoint['model_config'],
        ).to(config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        self.model = model


    def p(self, s: Any):
        """Print debug information if debug mode is enabled.
        
        Args:
            s (Any): Message to print to stderr
        """
        if self.config.debug:
            print(f"[GPTBot Debug] {s}", file=sys.stderr, flush=True)

    def get_aps(self, play_log):
        play_log = play_log.rstrip('\n')
        play_tokens = self.tokenizer.tokenize(play_log)[:-1]
        output = self.model(self.tokenizer.pad_and_tensor(play_tokens))

        play_logits = output['policy']
        play_values = output['value']

        logits = play_logits[:, len(play_tokens) - 1, :]  # [1, vocab_size]
        probs = torch.softmax(logits, dim=-1)

        play_dict: dict[tuple[int, int | None], float] = {
            (1, None): probs[0, SPECIAL_TOKENS['1']].item(),
            (1, 2): probs[0, SPECIAL_TOKENS['1=2']].item(),
            (1, 3): probs[0, SPECIAL_TOKENS['1=3']].item(),
            (1, 4): probs[0, SPECIAL_TOKENS['1=4']].item(),
            (1, 5): probs[0, SPECIAL_TOKENS['1=5']].item(),
            (1, 6): probs[0, SPECIAL_TOKENS['1=6']].item(),
            (1, 7): probs[0, SPECIAL_TOKENS['1=7']].item(),    
            (1, 8): probs[0, SPECIAL_TOKENS['1=8']].item(),
            (2, None): probs[0, SPECIAL_TOKENS['2']].item(),
            (3, None): probs[0, SPECIAL_TOKENS['3']].item(),
            (4, None): probs[0, SPECIAL_TOKENS['4']].item(),
            (5, 1): probs[0, SPECIAL_TOKENS['5=p1']].item(),
            (5, 2): probs[0, SPECIAL_TOKENS['5=p2']].item(),
            (6, None): probs[0, SPECIAL_TOKENS['5']].item(),
            (7, None): probs[0, SPECIAL_TOKENS['7']].item(),
        }
        value_dict: dict[tuple[int, int | None], float] = {
            (1, None): play_values[0, len(play_tokens) - 1].item(),
            (1, 2): play_values[0, len(play_tokens) - 1].item(),
            (1, 3): play_values[0, len(play_tokens) - 1].item(),
            (1, 4): play_values[0, len(play_tokens) - 1].item(),
            (1, 5): play_values[0, len(play_tokens) - 1].item(),
            (1, 6): play_values[0, len(play_tokens) - 1].item(),
            (1, 7): play_values[0, len(play_tokens) - 1].item(),    
            (1, 8): play_values[0, len(play_tokens) - 1].item(),
            (2, None): play_values[0, len(play_tokens) - 1].item(),
            (3, None): play_values[0, len(play_tokens) - 1].item(),
            (4, None): play_values[0, len(play_tokens) - 1].item(),
            (5, 1): play_values[0, len(play_tokens) - 1].item(),
            (5, 2): play_values[0, len(play_tokens) - 1].item(),
            (6, None): play_values[0, len(play_tokens) - 1].item(),
            (7, None): play_values[0, len(play_tokens) - 1].item(),
        }
        return play_dict, value_dict

    def choose_move(self, lines: list[str], hand: Hand, state: GameState, time_limit: int | None) -> Move:
        """Choose which card to play and optional target.
        
        Args:
            hand (Hand): The cards currently in the bot's hand
            state (GameState): Current game state information
            time_limit (int | None): Maximum time allowed for the move
            
        Returns:
            Move: Tuple of (card_to_play, target) where target may be None
        """
        self.p(f"Hand: {hand}")
        self.p(f"State: {state}")
        self.p(f"Time limit: {time_limit}")

        my_player = f"p{1 if state['am_i_player_one'] else 2}"
        log = "\n".join(lines)

        legal_actions = get_legal_actions(hand, state)

        if len(legal_actions) == 1:
            return legal_actions[0]

        with torch.inference_mode():

            play_log = log + "\n" + f"|{my_player}|play"
            play_dict, value_dict = self.get_aps(play_log)
            self.p(f"Play log: {play_log}")
            self.p(f"Play dict: {play_dict}")

            # Create batch of all action sequences
            action_logs = []
            avs = {}
            game_count = {}
            x_tensor = {}
            length_tensor = {}

            for action in legal_actions:
                card_to_play, target = action
                action_str = f"|{my_player}|play|{card_to_play}"
                if target is not None:
                    if card_to_play == 1:
                        action_str += f"={target}"
                    elif card_to_play == 5:
                        action_str += f"=p{target}"
                continued_log = log + "\n" + action_str
                action_logs.append(continued_log)
                tokens = self.tokenizer.tokenize(continued_log)
                padded_tokens = self.tokenizer.pad_tokens(tokens)

                x_tensor[action] = padded_tokens
                length_tensor[action] = len(tokens)
                # x = torch.tensor([padded_tokens] * self.config.N, device=self.config.device, dtype=torch.long)
                # length = torch.tensor([len(tokens)] * self.config.N, device=self.config.device, dtype=torch.long)

                # x, length = self.model.generate_continuation(x, length, self.config.depth, temperature=1)
                # values = self.model.get_value(x)  # [batch_size, seq_len, 1]
                # final_values = values[torch.arange(self.config.N), length - 1, 0]  # [batch_size]

                # avs[action] = final_values.mean().item()
            random_actions = random.choices(list(play_dict.keys()), weights=list(play_dict.values()), k=self.config.N)
            x = torch.tensor([x_tensor[ra] for ra in random_actions], device=self.config.device, dtype=torch.long)
            length = torch.tensor([length_tensor[ra] for ra in random_actions], device=self.config.device, dtype=torch.long)

            x, length = self.model.generate_continuation(x, length, self.config.depth, temperature=1)
            value = self.model.get_value(x)  # [batch_size, seq_len, 1]
            
            final_values = value[torch.arange(len(random_actions)), length - 1, 0]
            for i, action in enumerate(random_actions):
                avs[action] += final_values[i].item()
                game_count[action] += 1

            for action in avs.keys():
                avs[action] += (self.config.N - game_count[action]) * value_dict[action]


            # # Tokenize all sequences at once
            # all_tokens = [self.tokenizer.tokenize(action_log) for action_log in action_logs]
            # padded_tokens = [self.tokenizer.pad_tokens(tokens) for tokens in all_tokens]

            #  # Create one big batch tensor
            # batch_size = len(legal_actions) * self.config.N
            # x = torch.tensor(padded_tokens * self.config.N, device=self.config.device, dtype=torch.long)
            # length = torch.tensor([len(tokens) for tokens in all_tokens] * self.config.N, device=self.config.device)

            # # Generate continuations for all sequences at once
            # x, length = self.model.generate_continuation(x, length, self.config.depth, temperature=1)

            # # Get values for all sequences
            # values = self.model.get_value(x)  # [batch_size, seq_len, 1]
            # final_values = values[torch.arange(batch_size), length - 1, 0]  # [batch_size]

            # # Reshape and average values for each action
            # final_values = final_values.view(len(legal_actions), self.config.N)
            # avs = {action: final_values[i].mean().item() for i, action in enumerate(legal_actions)}

            weights = []
            for action, value in avs.items():
               new_value = play_dict[action] * (((1 + value/self.config.N) / 2) ** self.config.temp)
               weights.append(new_value)

            for action, value in avs.items():
               new_value = play_dict[action] * (((1 + value/self.config.N) / 2) ** self.config.temp)
               self.p(f"Action: {action}, Value: {value} New Value {new_value / sum(weights) * 100:.2f}%")

            selected_action = random.choices(list(avs.keys()), weights=weights, k=1)[0]

            self.p(f"Selected action: {selected_action}")
                
            return selected_action


if __name__ == "__main__":
    config = GPTBotConfig.from_args_and_yaml('./config/gpt_bot.yaml')
    GPTBot(config).main()