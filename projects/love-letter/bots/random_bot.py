#!/usr/bin/env python3

import random
import sys
from typing import Any
from dataclasses import dataclass
from utils.bot import GameState, LoveLetterBot, Move, Hand
from common.config import BaseConfig


@dataclass
class RandomBotConfig(BaseConfig):
    """Configuration class for RandomBot with debug option."""
    debug: bool = False


class RandomBot(LoveLetterBot):
    """A bot that makes random moves in Love Letter while following game rules.
    
    This bot chooses cards randomly from valid options while respecting the game's
    constraints like the Countess and Princess rules.
    """
    def __init__(self, config: RandomBotConfig):
        """Initialize the RandomBot with configuration.
        
        Args:
            config (RandomBotConfig): Configuration object containing bot settings
        """
        self.config = config

    def p(self, s: Any):
        """Print debug information if debug mode is enabled.
        
        Args:
            s (Any): Message to print to stderr
        """
        if self.config.debug:
            print(f"[RandomBot Debug] {s}", file=sys.stderr, flush=True)

    def choose_move(self, hand: Hand, state: GameState, time_limit: int | None) -> Move:
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
        
        # If holding Countess (7) with King (6) or Prince (5), must play Countess
        if 7 in hand and (6 in hand or 5 in hand):
            self.p("Countess restriction... Forced to play Countess")
            return 7, None
            
        # If Princess (8) is one of our cards, never discard it if we have choice
        if 8 in hand and len(hand) > 1:
            self.p("Princess restriction... Forced to keep Princess")
            card_candidates = [c for c in hand if c != 8]
        else:
            card_candidates = hand[:]
            
        self.p(f"Card candidates: {card_candidates}")
        # Choose random card from valid candidates
        card_to_play = random.choice(card_candidates)
        
        # Handle targeting for Guard and Prince
        target = None
        if card_to_play == 1 and not state['opponent_protected']:  # Guard
            target = random.randint(2, 8)
        elif card_to_play == 5:  # Prince
            if not state['opponent_protected']:
                target = random.randint(1, 2)
            else:
                target = 1 if state['am_i_player_one'] else 2

        self.p(f"Chosen card: {card_to_play}")
        self.p(f"Chosen target: {target}")
            
        return card_to_play, target


if __name__ == "__main__":
    config = RandomBotConfig.from_args_and_yaml('./config/random_bot.yaml')
    RandomBot(config).main()