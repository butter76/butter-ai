
from typing import Any, Protocol, TypedDict
import sys


class GameState(TypedDict):
    am_i_player_one: bool
    opponent_protected: bool 
    winner: str | None

Card = int
Target = int | None
Move = tuple[Card, Target]
Hand = list[Card]

def get_legal_actions(hand: Hand, state: GameState) -> list[Move]:
    legal_actions = []
    
    for _, card in enumerate(hand):
        if _ == 1 and hand[0] == hand[1]:
            # duplicate card
            break
        if card == 7 and (6 in hand or 5 in hand):
            return [(7, None)]
        
        if card == 1:
            if not state['opponent_protected']:
                legal_actions.extend((card, t) for t in range(2, 9))
            else:
                legal_actions.append((1, None))
        elif card == 5:  # Cards that need targets
            if not state['opponent_protected']:
                legal_actions.extend((card, t) for t in [1, 2])
            else:  # Prince
                legal_actions.append((card, 1 if state['am_i_player_one'] else 2))
        elif card == 8:
            continue
        else:
            legal_actions.append((card, None))
            
    return legal_actions

def get_legal_strs(state: GameState, legal_actions: list[Move]) -> list[str]:
    s = []
    for action, target in legal_actions:
        player =  'p1' if state['am_i_player_one'] else 'p2'
        if target is not None:
            s.append(f'|{player}|play|{action}|{target}')
        else:
            s.append(f'|{player}|play|{action}')

    return s

def parse_game_state(lines: list[str]) -> tuple[Hand, GameState]:
    """Parse the game log to extract current hand and game state"""

    lines = lines[4:]

    hand = []
    am_i_player_one = None
    my_player_num = None
    opponent_protected = False
    winner = None

    turnLine = lines[-1]
    am_i_player_one = turnLine.split('|')[2] == 'p1'
    my_player_num = 1 if am_i_player_one else 2
    opponent_num = 2 if am_i_player_one else 1
    
    for line in lines:
        parts = line.split('|')
        if len(parts) < 2:
            continue
            
        # Track our draws
        if parts[1] == f'p{my_player_num}' and parts[2] == 'hidden' and parts[3] == 'draw':
            hand.append(int(parts[4]))

        # Track when we play cards (remove from hand)
        if parts[1] == f'p{my_player_num}' and parts[2] == 'play':
            played_card = int(parts[3])
            if played_card in hand:
                hand.remove(played_card)

        # Track King swaps
        if parts[1] == 'swap':
            p1_card = int(parts[4])
            p2_card = int(parts[5])
            if f'p{my_player_num}' == parts[2]:  # We initiated swap
                hand = [p2_card]
            elif f'p{my_player_num}' == parts[3]:  # We were target of swap
                hand = [p1_card]

        # Track Prince discards
        if parts[1] == f'p{my_player_num}' and parts[2] == 'discard':
            discarded_card = int(parts[3])
            if discarded_card in hand:
                hand.remove(discarded_card)

        # Track Handmaid protection
        if len(parts) > 2 and parts[2] == 'play' and parts[3] == '4':
            opponent_protected = True
        if parts[1] == 'turn' and parts[2] != f'p{my_player_num}':
            opponent_protected = False

        if parts[1] == 'game':
            winner = parts[2]
            break
            
    return hand, {
        'am_i_player_one': am_i_player_one,
        'opponent_protected': opponent_protected,
        'winner': winner
    }


class LoveLetterBot(Protocol):
    def choose_move(self, hand: list[int], state: GameState, time_limit: int | None) -> Move:
        ...

    def main(self):
        # Read input line by line continuously
        for line in sys.stdin:
            if line.startswith('move'):
                logs = []
                time_limit = int(line.split(' ')[1])
                
                # Read subsequent log lines
                while True:
                    log_line = sys.stdin.readline().strip()
                    if not log_line:
                        break
                    logs.append(log_line)
                
                # Parse game state
                hand, game_state = parse_game_state(logs)
            
                # Choose move
                card_to_play, target = self.choose_move(hand, game_state, time_limit)
            
                # Output move and flush stdout
                if target is not None:
                    print(f"{card_to_play} {target}", flush=True)
                else:
                    print(card_to_play, flush=True)
