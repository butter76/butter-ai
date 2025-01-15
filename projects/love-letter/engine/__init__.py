from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Callable, cast
import random
from datetime import datetime

@dataclass
class PlayerState:
	hand: List[int]
	protected: bool  # whether Handmaid (4) effect is active
	eliminated: bool  # track if player is knocked out

class LoveLetterEngine:
	def __init__(self, game_id: str, 
				 log_callback: Optional[Callable[[str], None]] = None,
				 debug: bool = False, deck: Optional[List[int]] = None):
		if deck is None:
			deck = self._generate_deck()
			random.shuffle(deck)
		# Initialize deck (16 cards)
		# Cards: 1(Guard)x5, 2(Priest)x2, 3(Baron)x2, 4(Handmaid)x2,
		# 5(Prince)x2, 6(King)x1, 7(Countess)x1, 8(Princess)x1
		self.deck = deck
		
		# Remove top card (face-down)
		self.removed_card = self.deck.pop(0) if self.deck else None
		
		# Initialize player states
		self.players = []
		for _ in range(2):
			card = self.deck.pop(0) if self.deck else None
			self.players.append(PlayerState(
				hand=[card] if card is not None else [],
				protected=False,
				eliminated=False
			))
		
		self.current_player_idx = 0
		self.game_id = game_id
		self.game_ended = False
		self.debug = debug
		self.log_callback = log_callback or (lambda x: None)
		
	def _generate_deck(self) -> List[int]:
		return [
			1,1,1,1,1,  # Guards (5)
			2,2,        # Priests (2)
			3,3,        # Barons (2)
			4,4,        # Handmaids (2)
			5,5,        # Princes (2)
			6,          # King (1)
			7,          # Countess (1)
			8           # Princess (1)
		]
	
	def log(self, line: str) -> None:
		self.log_callback(line)
		if self.debug:
			print(f"[DEBUG][Engine][{self.game_id}] {line}")
	
	def start_game_log(self, player_names: List[str]) -> None:
		self.log(f"|game|{self.game_id}")
		self.log(f"|timestamp|{datetime.utcnow().isoformat()}Z")
		for idx, name in enumerate(player_names, 1):
			self.log(f"|player|p{idx}|{name}")
		self.log("|gamestart")
		
		# Log initial draws
		for idx, player in enumerate(self.players, 1):
			if player.hand:
				self.log(f"|p{idx}|hidden|draw|{player.hand[0]}")
		
		# Player 1 starts
		self.log("|turn|p1")
	
	def get_current_player_index(self) -> int:
		return self.current_player_idx
	
	def draw_card_for_current_player(self) -> None:
		if self.game_ended:
			return
			
		current_player = self.players[self.current_player_idx]
		if current_player.eliminated:
			return
			
		if self.deck:
			drawn = self.deck.pop(0)
			current_player.hand.append(drawn)
			self.log(f"|p{self.current_player_idx + 1}|hidden|draw|{drawn}")
		elif self.removed_card is not None:
			current_player.hand.append(self.removed_card)
			self.log(f"|p{self.current_player_idx + 1}|hidden|draw|{self.removed_card}")
			self.removed_card = None
			
		self.log(f"|yourmove|p{self.current_player_idx + 1}")
	
	def make_move(self, player_idx: int, card_to_play: int, target: int | None = None) -> None:
		if self.game_ended:
			return
			
		if player_idx != self.current_player_idx:
			self.log(f"|lose|p{player_idx + 1}|invalid")
			self.log(f"|end|p{(player_idx + 1) % 2 + 1}|win")
			self.game_ended = True
			return
		
		current_player = self.players[player_idx]
		other_idx = (player_idx + 1) % 2
		other_player = self.players[other_idx]
		
		# Clear Handmaid protection at start of turn
		current_player.protected = False
		
		# Validate card is in hand
		if card_to_play not in current_player.hand:
			self.log(f"|lose|p{player_idx + 1}|invalid")
			self.log(f"|end|p{other_idx + 1}|win")
			self.game_ended = True
			return
		
		# Enforce Countess rule
		if (7 in current_player.hand and 
			(6 in current_player.hand or 5 in current_player.hand) and 
			card_to_play != 7):
			self.log(f"|lose|p{player_idx + 1}|invalid")
			self.log(f"|end|p{other_idx + 1}|win")
			self.game_ended = True
			return
		
		# Validate Guard guess
		if (card_to_play == 1 and 
			((not other_player.protected and (target is None or target < 2 or target > 8)) or 
			(other_player.protected and target is not None))):
			self.log(f"|lose|p{player_idx + 1}|invalid")
			self.log(f"|end|p{other_idx + 1}|win")
			self.game_ended = True
			return
		
		# Validate Prince target
		if (card_to_play == 5 and 
			(target is None or target <= 0 or target > 2 or 
			 (target == other_idx + 1 and other_player.protected))):
			self.log(f"|lose|p{player_idx + 1}|invalid")
			self.log(f"|end|p{other_idx + 1}|win")
			self.game_ended = True
			return
		
		# Play the card
		play_log = f"|p{player_idx + 1}|play|{card_to_play}"
		if target is not None:
			if card_to_play == 1:
				play_log += f"|{target}"
			elif card_to_play == 5:
				play_log += f"|p{target}"
		self.log(play_log)
		
		# Remove played card from hand
		current_player.hand.remove(card_to_play)
		
		# Handle card effects
		if card_to_play == 1 and not other_player.protected:  # Guard
			if not other_player.eliminated and other_player.hand[0] == target:
				other_player.eliminated = True
				self.log(f"|lose|p{other_idx + 1}|guard")
				self.log(f"|end|p{player_idx + 1}|win")
				self.game_ended = True
				return
				
		elif card_to_play == 2 and not other_player.protected:  # Priest
			if not other_player.eliminated:
				self.log(f"|p{other_idx + 1}|reveal|{other_player.hand[0]}")
				
		elif card_to_play == 3 and not other_player.protected:  # Baron
			if not other_player.eliminated:
				my_card = current_player.hand[0]
				their_card = other_player.hand[0]
				if my_card > their_card:
					other_player.eliminated = True
					self.log(f"|lose|p{other_idx + 1}|baron")
					self.log(f"|end|p{player_idx + 1}|win")
					self.game_ended = True
					return
				elif their_card > my_card:
					current_player.eliminated = True
					self.log(f"|lose|p{player_idx + 1}|baron")
					self.log(f"|end|p{other_idx + 1}|win")
					self.game_ended = True
					return
				
		elif card_to_play == 4:  # Handmaid
			current_player.protected = True
			
		elif card_to_play == 5:  # Prince
			target_idx = cast(int, target) - 1
			target_player = self.players[target_idx]
			if not target_player.eliminated:
				discarded = target_player.hand[0]
				self.log(f"|p{target_idx + 1}|discard|{discarded}")
				target_player.hand.pop(0)
				
				if discarded == 8:  # Princess
					target_player.eliminated = True
					if target_idx == player_idx:
						self.log(f"|lose|p{player_idx + 1}|princess")
						self.log(f"|end|p{other_idx + 1}|win")
					else:
						self.log(f"|lose|p{other_idx + 1}|princess")
						self.log(f"|end|p{player_idx + 1}|win")
					self.game_ended = True
					return
				
				# Draw new card
				if self.deck:
					new_card = self.deck.pop(0)
					target_player.hand.append(new_card)
					self.log(f"|p{target_idx + 1}|hidden|draw|{new_card}")
				elif self.removed_card is not None:
					target_player.hand.append(self.removed_card)
					self.log(f"|p{target_idx + 1}|hidden|draw|{self.removed_card}")
					self.removed_card = None
					
		elif card_to_play == 6 and not other_player.protected:  # King
			if not other_player.eliminated:
				my_hand = current_player.hand
				their_hand = other_player.hand
				self.log(f"|swap|p{player_idx + 1}|p{other_idx + 1}|{my_hand[0]}|{their_hand[0]}")
				current_player.hand = their_hand
				other_player.hand = my_hand
				
		elif card_to_play == 8:  # Princess
			current_player.eliminated = True
			self.log(f"|lose|p{player_idx + 1}|princess")
			self.log(f"|end|p{other_idx + 1}|win")
			self.game_ended = True
			return
			
		# End of turn
		if not self.game_ended:
			if not self.deck:
				self.log("|nodraw")
				# Compare hands
				p1_card = self.players[0].hand[0]
				p2_card = self.players[1].hand[0]
				if p1_card > p2_card:
					self.log("|lose|p2|highest")
					self.log("|end|p1|win")
				else:
					self.log("|lose|p1|highest")
					self.log("|end|p2|win")
				self.game_ended = True
				return
				
			self.current_player_idx = (self.current_player_idx + 1) % 2
			self.log(f"|turn|p{self.current_player_idx + 1}")
	
	def is_game_over(self) -> bool:
		return self.game_ended