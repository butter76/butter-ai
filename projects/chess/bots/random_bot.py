from typing import cast
import chess
import chess.engine
import random
import asyncio

class RandomBot(chess.engine.Protocol):
    def __init__(self):
        self.name = "RandomBot"
        self.author = "Example"

    async def initialize(self):
        ...
    
    async def ping(self):
        ...
    
    async def configure(self, options):
        ...
    
    async def send_opponent_information(self, *, opponent = None, engine_rating = None):
        ...

    async def play(self, board, limit, *, game = None, info = ..., ponder = False, draw_offered = False, root_moves = None, options = ..., opponent = None):
        # Get list of legal moves from current position
        legal_moves = list(board.legal_moves)
        
        # Choose a random move
        move = random.choice(legal_moves)
        
        # Return the chosen move with info
        return chess.engine.PlayResult(move, None)
      
    def analysis(self, board, limit = None, *, multipv = None, game = None, info = ..., root_moves = None, options = ...):
        return super().analysis(board, limit, multipv=multipv, game=game, info=info, root_moves=root_moves, options=options)


    async def send_game_result(self, board, winner = None, game_ending = None, game_complete = True):
        ...

    async def quit(self):
        ...

async def main():
    engine = RandomBot()
    board = chess.Board()
    while True:
        line = input().strip()
        if line == "uci":
            print("id name RandomBot")
            print("id author Example")
            print("option name Debug type check default false")
            print("option name MultiPV type spin default 1 min 1 max 500")
            print("uciok")
        elif line == "isready":
            print("readyok")
        elif line == "ucinewgame":
            board = chess.Board()
        elif line.startswith("setoption"):
            # Handle setoption commands
            parts = line.split(" ", 4)
            if len(parts) >= 5 and parts[1] == "name" and parts[3] == "value":
                option_name = parts[2]
                option_value = parts[4]
                await engine.configure({option_name: option_value})
        elif line.startswith("position"):
            parts = line.split(" ", 2)
            if len(parts) >= 2:
                if parts[1] == "startpos":
                    board = chess.Board()
                    if len(parts) > 2 and parts[2].startswith("moves"):
                        moves = parts[2].split()[1:]
                        for move in moves:
                            board.push_uci(move)
                elif parts[1].startswith("fen"):
                    fen = " ".join(parts[1:]).replace("fen ", "")
                    if "moves" in fen:
                        fen, moves_part = fen.split("moves")
                        board = chess.Board(fen.strip())
                        moves = moves_part.strip().split()
                        for move in moves:
                            board.push_uci(move)
                    else:
                        board = chess.Board(fen)
        elif line.startswith("go"):
            parts = line.split()
            time_limit = chess.engine.Limit()
            i = 0
            while i < len(parts):
                if parts[i] == "wtime" and i + 1 < len(parts):
                    time_limit.white_clock = float(parts[i + 1]) / 1000
                    i += 2
                elif parts[i] == "btime" and i + 1 < len(parts):
                    time_limit.black_clock = float(parts[i + 1]) / 1000
                    i += 2
                elif parts[i] == "winc" and i + 1 < len(parts):
                    time_limit.white_inc = float(parts[i + 1]) / 1000
                    i += 2
                elif parts[i] == "binc" and i + 1 < len(parts):
                    time_limit.black_inc = float(parts[i + 1]) / 1000
                    i += 2
                elif parts[i] == "movetime" and i + 1 < len(parts):
                    time_limit.time = float(parts[i + 1]) / 1000
                    i += 2
                elif parts[i] == "depth" and i + 1 < len(parts):
                    time_limit.depth = int(parts[i + 1])
                    i += 2
                else:
                    i += 1
            result = await engine.play(board, time_limit)
            if result.move is not None:
                print(f"bestmove {result.move.uci()}")
        elif line == "stop":
            # Handle stop command if engine is thinking
            pass
        elif line == "quit":
            await engine.quit()
            break

if __name__ == "__main__":
    asyncio.run(main())