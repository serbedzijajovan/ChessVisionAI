import glob
import random
import threading

import chess
import numpy as np
from stockfish import Stockfish

from config import RAW_DATA_PATH, STOCKFISH_PATH

file_lock = threading.Lock()


# ********** Helper Functions **********
def is_game_over(board):
    return (
            board.is_checkmate() or
            board.is_stalemate() or
            board.is_insufficient_material() or
            board.can_claim_threefold_repetition() or
            board.can_claim_fifty_moves() or
            board.can_claim_draw()
    )


def find_next_idx():
    """
    Retrieve all .npy files and return the next index after the highest found.
    """
    files = glob.glob(f"{RAW_DATA_PATH}/*.npy")
    if len(files) == 0:
        return 1

    highest_idx = max(int(f.split("moves_and_positions")[-1].split(".npy")[0]) for f in files)
    return highest_idx + 1


def save_data(positions, moves):
    """
    Save moves and positions to a .npy file.
    """
    # Reshape moves and positions into 2D arrays (columns)
    positions_array = np.array(positions).reshape(-1, 1)
    moves_array = np.array(moves).reshape(-1, 1)
    positions_and_moves = np.concatenate((positions_array, moves_array), axis=1)

    next_idx = find_next_idx()
    np.save(f"{RAW_DATA_PATH}/moves_and_positions{next_idx}.npy", positions_and_moves)
    print("Saved successfully")


def run_game(num_moves, filename="movesAndPositions1.npy"):
    """
    Run a saved game up to a specified number of moves.
    """
    game_data = np.load(f"data/{filename}")

    # Extract moves from the data (first column)
    saved_moves = game_data[:, 1]
    if num_moves > len(saved_moves):
        print(f"Must enter a lower number of moves than maximum game length. Game length here is: {len(saved_moves)}")
        return

    # Play the specified number of moves
    board = chess.Board()
    for i in range(num_moves):
        move = saved_moves[i]
        board.push_san(move)

    return board


def get_stockfish_move(stockfish, played_moves):
    """
    Get a move from Stockfish based on the current position.
    """
    stockfish.set_position(played_moves)
    moves = stockfish.get_top_moves(3)

    if not moves:
        return None
    elif len(moves) == 1:
        return moves[0]["Move"]
    elif len(moves) == 2:
        return random.choices(moves, weights=(80, 20), k=1)[0]["Move"]
    else:
        return random.choices(moves, weights=(80, 15, 5), k=1)[0]["Move"]


def mine_games_thread(num_games, stockfish_path):
    stockfish = Stockfish(path=stockfish_path)
    for _ in range(num_games):
        MAX_MOVES = 500
        current_game_positions = []
        current_game_moves = []
        board = chess.Board()

        for _ in range(MAX_MOVES):
            move = get_stockfish_move(stockfish, current_game_moves)
            if not move:
                break

            current_game_positions.append(stockfish.get_fen_position())
            current_game_moves.append(move)

            board.push_uci(move)
            if is_game_over(board):
                print("Game is over")
                break
        save_data(current_game_positions, current_game_moves)


def mine_games(num_games, num_threads=8):
    threads = []
    for _ in range(num_threads):
        thread = threading.Thread(target=mine_games_thread, args=(num_games // num_threads, STOCKFISH_PATH))
        thread.start()
        threads.append(thread)

    # Wait for all threads to complete
    for t in threads:
        t.join()


if __name__ == '__main__':
    mine_games(20)
