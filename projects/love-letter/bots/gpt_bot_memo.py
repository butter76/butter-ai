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

    def helper_continuation(self, node, depth, new_games):#blah
        #nodes represent a game state in a prefix tree. Includes a 
        #gamestate, a distribution over the next move,
            #gamestate
            #distribution over the next move,
            #end = number of games with this state as an endstate
            #next_char_dict = dict next_chars to node
            #weight = number of games which reach this node
            #model_value = how model views this node
            #total_z_value = z value of the node the actual z value is total/weight
        node.weight += new_games
        node.end += new_games
        node.total_z_value = None
        if WIN:
            node.total_z_value += new_games
            return
        if LOSE:
            node.total_z_value -= new_games
            return
        if node.distribution == None:
            node.distribution = exp(self.model.get_value(x=node.gamestate)) #also get node.model_value from this?
            node.value = BLAH
        if depth == 0:
            node.total_z_value += node.value*new_games
            return
        selected_next_chars = random.choices(?, weights=node.distribution, k=1)[node.end]
        for char in selected_next_chars:
            if char not in node.next_char_dict:
                node.next_char_dict[char] = Node(node.gamestate.append(char))
        for next_char in node.next_char_dict:
            helper_continuation(self, node.next_char_dict[next_char], depth-1,selected_next_chars.count(next_char))
        return

    def generate_continuation_tree(self, node, legal_actions):
        #fix for guard and prince
        for char in node.next_char_dict:
            if char not in legal_actions:
                node.weight -= node.next_char_dict[char].weight
                del node.next_char_dict[char]
                #remove char from dict? is this done by the above line?
        weights = [node.distribution[action] for action in legal_actions]
        selected_next_chars = random.choices(legal_actions, weights=weights, k=1)[self.number_of_continuations-node.weight]
        for char in selected_next_chars:
            if char not in node.next_char_dict:
                node.next_char_dict[char] = Node(node.gamestate.append(char))
        for next_char in node.next_char_dict:
            helper_continuation(self, node.next_char_dict[next_char], self.depth-1,selected_next_chars.count(next_char))
        return
        

    def compute_z_value(node):
        if node.total_z_value:
            return node.total_z_value
        total_z_value = 0
        for char in node.next_char_dict:
            total_z_value += compute_z_value(node.next_char_dict[char])
        return total_z_value

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
            generate_continuation_tree(self, node, legal_actions)
            play_weights = []
            for char in node.next_char_dict:
                action_node = node.next_char_dict[char]
                play_weights.append(action_node.value**(self.number_of_continuations-action_node.weight))*(compute_z_value(action_node.value)**action_node.weight)

            selected_action = random.choices(node.next_char_dict, weights=weights, k=1)[0]

            self.p(f"Selected action: {selected_action}")
                
            return selected_action #fix the return a tiny bit


if __name__ == "__main__":
    config = GPTBotConfig.from_args_and_yaml('./config/gpt_bot.yaml')
    GPTBot(config).main()