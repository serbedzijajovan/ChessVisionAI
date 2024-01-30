import time

import chess
import torch

from config import TRAINING_PATH
from model_training.model import ChessModel

SAVED_MODELS_PATH = f"{TRAINING_PATH}/saved_models"

# Load best model path from file
f = open(f"{SAVED_MODELS_PATH}/bestModel.txt", "r")
bestLoss = float(f.readline())
model_path = f"{SAVED_MODELS_PATH}/{f.readline()}"
f.close()

saved_model = ChessModel()
saved_model.load_state_dict(torch.load(model_path))
board = chess.Board()


def input_uci():
    print("id name MyEngine")
    print("id author SC")
    print("uciok")


def input_set_option():
    # set options
    pass


def input_is_ready():
    print("readyok")


def input_uci_new_game():
    # add code
    pass


def input_position(args):
    moves_start_index = -1
    if args[1] == "startpos":
        board.reset()
        moves_start_index = 3 if len(args) > 2 and args[2] == "moves" else -1
    elif args[1] == "fen":
        fen = ' '.join(args[2:8])
        board.set_fen(fen)
        moves_start_index = 9 if len(args) > 9 and args[8] == "moves" else -1

    # Play moves if they exist
    if moves_start_index != -1:
        moves = args[moves_start_index:]
        for move in moves:
            if chess.Move.from_uci(move) in board.legal_moves:
                board.push_uci(move)
            else:
                print("Illegal move:", move)
                break


def input_go():
    time.sleep(0.5)

    # Get the best move from Stockfish
    best_move = saved_model.predict(board)

    # Play the best move on the board
    if best_move:
        best_move_uci = best_move.uci()
        if best_move in board.legal_moves:
            print("bestmove", best_move_uci)
        else:
            print("Illegal move suggested by Stockfish:", best_move_uci)
    else:
        print("No move suggested by Stockfish")


if __name__ == "__main__":
    while True:
        args = input().split()
        if not args:
            continue

        if args[0] == "uci":
            input_uci()
        elif args[0] == "setoption":
            input_set_option()
        elif args[0] == "isready":
            input_is_ready()
        elif args[0] == "ucinewgame":
            input_uci_new_game()
        elif args[0] == "position":
            input_position(args)
        elif args[0] == "go":
            input_go()
        elif args[0] == "print":
            print(board)
        elif args[0] == "quit":
            break
