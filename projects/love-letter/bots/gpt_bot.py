#!/usr/bin/env python3

import random
import sys
from typing import Any
from dataclasses import dataclass

import torch
torch.set_default_dtype(torch.bfloat16)
torch.set_printoptions(profile="full")
from gpt.models.gpt_ll import LoveLetterTransformer
from gpt.models.tokenizer import LoveLetterTokenizer
from utils.bot import GameState, LoveLetterBot, Move, Hand, get_legal_actions
from common.config import BaseConfig


@dataclass
class GPTBotConfig(BaseConfig):
    """Configuration class for GPTBot with debug option."""
    debug: bool = False
    checkpoint: str = "./gpt/checkpoints/model_epoch_40.pt"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    temp: float = 1
    depth: int = 20
    N: int = 15


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

    def get_aps(self, play_log) -> dict[tuple[int, int | None], float]:
        play_log = play_log.rstrip('\n')
        play_tokens = self.tokenizer.tokenize(play_log)[:-1]
        play_logits = self.model.get_policy(self.tokenizer.pad_and_tensor(play_tokens))

        logits = play_logits[:, len(play_tokens) - 1, :]  # [1, vocab_size]
        probs = torch.softmax(logits, dim=-1)

        guard_prob = probs[0, 29].item()
        prince_prob = probs[0, 33].item()

        guard_log = play_log + "|1"
        prince_log = play_log + "|5"
        
        guard_tokens = self.tokenizer.tokenize(guard_log)[:-1]
        prince_tokens = self.tokenizer.tokenize(prince_log)[:-1]
        
        guard_logits = self.model.get_policy(self.tokenizer.pad_and_tensor(guard_tokens))
        prince_logits = self.model.get_policy(self.tokenizer.pad_and_tensor(prince_tokens))
        
        guard_probs = torch.softmax(guard_logits[:, len(guard_tokens) - 1, :], dim=-1)
        prince_probs = torch.softmax(prince_logits[:, len(prince_tokens) - 1, :], dim=-1)

        play_dict = {
            (1, None): guard_prob * (guard_probs[0, 25].item() + guard_probs[0, 26].item()),
            (1, 2): guard_prob * guard_probs[0, 30].item(),
            (1, 3): guard_prob * guard_probs[0, 31].item(),
            (1, 4): guard_prob * guard_probs[0, 32].item(),
            (1, 5): guard_prob * guard_probs[0, 33].item(),
            (1, 6): guard_prob * guard_probs[0, 34].item(),
            (1, 7): guard_prob * guard_probs[0, 35].item(),    
            (1, 8): guard_prob * guard_probs[0, 36].item(),
            (2, None): probs[0, 30].item(),
            (3, None): probs[0, 31].item(),
            (4, None): probs[0, 32].item(),
            (5, 1): prince_prob * prince_probs[0, 4].item(),
            (5, 2): prince_prob * prince_probs[0, 5].item(),
            (6, None): probs[0, 34].item(),  
            (7, None): probs[0, 35].item(),
        }
        return play_dict

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
            play_dict = self.get_aps(play_log)
            self.p(f"Play log: {play_log}")
            self.p(f"Play dict: {play_dict}")

            # Create batch of all action sequences
            action_logs = []
            for action in legal_actions:
                card_to_play, target = action
                action_str = f"|{my_player}|play|{card_to_play}"
                if target is not None:
                    if card_to_play == 1:
                        action_str += f"|{target}"
                    elif card_to_play == 5:
                        action_str += f"|p{target}"
                continued_log = log + "\n" + action_str
                action_logs.append(continued_log)

            # Tokenize all sequences at once
            all_tokens = [self.tokenizer.tokenize(log) for log in action_logs]
            padded_tokens = [self.tokenizer.pad_tokens(tokens) for tokens in all_tokens]

             # Create one big batch tensor
            batch_size = len(legal_actions) * self.config.N
            x = torch.tensor(padded_tokens * self.config.N, device=self.config.device, dtype=torch.long)
            length = torch.tensor([len(tokens) for tokens in all_tokens] * self.config.N, device=self.config.device)

            # Generate continuations for all sequences at once
            x, length = self.model.generate_continuation(x, length, self.config.depth, temperature=1)

            # Get values for all sequences
            values = self.model.get_value(x)  # [batch_size, seq_len, 1]
            final_values = values[torch.arange(batch_size), length - 1, 0]  # [batch_size]

            # Reshape and average values for each action
            final_values = final_values.view(len(legal_actions), self.config.N)
            avs = {action: final_values[i].mean().item() for i, action in enumerate(legal_actions)}

            weights = []
            for action, value in avs.items():
                new_value = play_dict[action] * (((1 + value) / 2) ** self.config.temp)
                self.p(f"Action: {action}, Value: {value} New Value: {new_value}")
                weights.append(new_value)

            selected_action = random.choices(legal_actions, weights=weights, k=1)[0]

            self.p(f"Selected action: {selected_action}")
                
            return selected_action


if __name__ == "__main__":
    config = GPTBotConfig.from_args_and_yaml('./config/gpt_bot.yaml')
    GPTBot(config).main()