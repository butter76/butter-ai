from operator import is_
import sys
import json
import os
from typing import List, Dict, Optional, Tuple, cast
from dataclasses import dataclass
import argparse

from engine import LoveLetterEngine

@dataclass
class GameMove:
	player: str
	action: str
	card: Optional[int] = None
	target: Optional[str | int] = None

class GameValidator:
	def __init__(self, log_path: str):
		self.log_path = log_path
		self.input_log: List[str] = []
		self.simulated_log: List[str] = []
		self.game_id: str = ""
		self.players: Dict[str, str] = {}
		self.current_hands: Dict[str, List[int]] = {"p1": [], "p2": []}
		self.is_after_turn = False
		
	def read_log(self) -> None:
		"""Read and store the input log file"""
		with open(self.log_path, 'r') as f:
			self.input_log = [line.strip() for line in f if line.strip()]

	def extract_draw_sequence(self) -> List[Tuple[int, bool]]:
		"""Extract the sequence of all draws from the log in order"""
		draws = []
		is_after_turn = False
		
		for line in self.input_log:
			parts = line.split('|')[1:]  # Skip empty first element
			if not parts:
				continue
				
			if parts[0] == 'turn':
				is_after_turn = True
				continue
				
			if len(parts) >= 4 and parts[1] == 'hidden' and parts[2] == 'draw':
				draws.append((int(parts[3]), is_after_turn))
				is_after_turn = False
		return draws

	def generate_deck_from_draws(self, draws: List[int]) -> List[int]:
		"""Generate a deck that starts with the given draw sequence"""
		# Start with the draws we know
		deck = draws[:]
		
		# Get the standard deck
		standard_deck = [
			1,1,1,1,1,  # Guards (5)
			2,2,        # Priests (2)
			3,3,        # Barons (2)
			4,4,        # Handmaids (2)
			5,5,        # Princes (2)
			6,          # King (1)
			7,          # Countess (1)
			8           # Princess (1)
		]
		
		# Remove cards we've seen from the standard deck
		for card in deck:
			standard_deck.remove(card)
			
		# Add remaining cards to complete the deck
		deck.extend(standard_deck)
		return deck[-1:] + deck[:-1]
			
	def parse_header(self) -> Tuple[str, Dict[str, str]]:
		"""Parse game ID and players from log header"""
		game_id = ""
		players = {}
		
		for line in self.input_log[:4]:  # First 4 lines contain header info
			parts = line.split('|')
			if parts[1] == 'game':
				game_id = parts[2]
			elif parts[1] == 'player':
				players[parts[2]] = parts[3]
				
		return game_id, players
	
	def parse_move(self, line: str) -> Optional[GameMove]:
		"""Parse a single line of the game log into a structured move"""
		parts = line.split('|')[1:]  # Skip empty first element
		if not parts:
			return None
			
		if parts[0] in ['p1', 'p2']:
			player = parts[0]
			action = parts[1]
			
			if action == 'hidden':
				if parts[2] == 'draw':
					target = 1 if self.is_after_turn else 0
					self.is_after_turn = False
					return GameMove(player=player, action='draw', card=int(parts[3]), target=target)
			elif action == 'play':
				if len(parts) > 3:
					# Has target
					target = parts[3]
					if target.startswith('p'):
						# Prince target
						return GameMove(player=player, action='play', card=int(parts[2]), target=target)
					else:
						# Guard guess
						return GameMove(player=player, action='play', card=int(parts[2]), target=int(parts[3]))
				else:
					return GameMove(player=player, action='play', card=int(parts[2]))
		elif parts[0] == 'turn':
			self.is_after_turn = True
					
		return None

	def validate_game(self) -> bool:
		"""
		Main validation function that:
		1. Reads input log
		2. Simulates game
		3. Compares logs
		"""
		self.read_log()
		self.is_after_turn = False
		
		# Parse game info
		self.game_id, self.players = self.parse_header()
		
		simulated_log = []
		def log_line(line: str) -> None:
			simulated_log.append(line)
			
		# Extract draws and generate deck
		draws = self.extract_draw_sequence()
		deck = self.generate_deck_from_draws([draw for (draw, _) in draws])
			
		# Initialize engine with our predetermined deck
		engine = LoveLetterEngine(game_id=self.game_id, log_callback=log_line, deck=deck)
		
		# Start game with player names
		engine.start_game_log([self.players['p1'], self.players['p2']])


		
		first_draws = 2
		# Process each move from the log
		for line in self.input_log[4:]:  # Skip header
			move = self.parse_move(line)
			if move:
				if move.action == 'draw' and move.target == 1:
					engine.draw_card_for_current_player()
				elif move.action == 'play':
					player_idx = int(move.player[1]) - 1
					engine.make_move(
						player_idx,
						cast(int, move.card),
						move.target if isinstance(move.target, int) else 
						int(move.target[1]) if move.target and move.target.startswith('p') else None
					)

		
		# Compare logs
		return self.compare_logs(simulated_log)
	
	def compare_logs(self, simulated_log: List[str]) -> bool:
		"""Compare input and simulated logs"""
		# Remove timestamps as they will differ
		input_without_timestamp = [
			line for line in self.input_log 
			if not line.startswith('|timestamp')
		]
		simulated_without_timestamp = [
			line for line in simulated_log 
			if not line.startswith('|timestamp')
		]
		
		if len(input_without_timestamp) != len(simulated_without_timestamp):
			print(f"Log length mismatch: {len(input_without_timestamp)} vs {len(simulated_without_timestamp)}")
			return False
			
		for i, (input_line, sim_line) in enumerate(zip(input_without_timestamp, simulated_without_timestamp)):
			if input_line != sim_line:
				print(f"Mismatch at line {i}:")
				print(f"Input:     {input_line}")
				print(f"Simulated: {sim_line}")
				return False
				
		return True

def validate_file(file_path: str) -> bool:
	"""Validate a single game log file with error handling"""
	try:
		validator = GameValidator(file_path)
		return validator.validate_game()
	except Exception as e:
		print(f"Error validating {file_path}: {str(e)}")
		return False

def main():
	parser = argparse.ArgumentParser(description='Validate Love Letter game logs')
	parser.add_argument('log_dir', help='Directory containing game log files to validate')
	args = parser.parse_args()

	if not os.path.isdir(args.log_dir):
		print(f"Error: {args.log_dir} is not a directory")
		sys.exit(1)

	# Find all files in the directory
	log_files = [f for f in os.listdir(args.log_dir) if os.path.isfile(os.path.join(args.log_dir, f))]
	
	if not log_files:
		print(f"No files found in {args.log_dir}")
		sys.exit(1)

	successful_validations = 0
	total_files = len(log_files)

	print(f"Found {total_files} files to validate")
	
	for file_name in log_files:
		file_path = os.path.join(args.log_dir, file_name)
		print(f"\nValidating {file_name}...")
		
		if validate_file(file_path):
			print(f"✅ {file_name}: Log validation successful")
			successful_validations += 1
		else:
			print(f"❌ {file_name}: Log validation failed")

	success_percentage = (successful_validations / total_files) * 100
	print(f"\nValidation Summary:")
	print(f"Successfully validated: {successful_validations}/{total_files} files ({success_percentage:.1f}%)")
	
	# Exit with success if all files validated successfully
	sys.exit(0 if successful_validations == total_files else 1)

if __name__ == "__main__":
	main()