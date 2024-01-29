import os
from typing import Optional

import chess
import numpy as np
from gym_chess.alphazero.move_encoding import utils

from config import RAW_DATA_PATH, PREPARED_DATA_PATH, DATA_PATH


def encode_queen(move: chess.Move) -> Optional[int]:
    """
    Encodes a queen's move in chess to a unique integer representation.

    Args:
        move (chess.Move): A chess move to be encoded.

    Returns:
        int: An integer representing the encoded queen move, or None if the move is not a queen move.
    """

    # num_types = 56 -> There are 56 types of moves with queen, 8 directions * 7 squares max. distance
    queen_directions = utils.IndexedTuple(
        (+1, 0),
        (+1, +1),
        (0, +1),
        (-1, +1),
        (-1, 0),
        (-1, -1),
        (0, -1),
        (+1, -1),
    )

    # Unpack the move's from and to squares
    from_rank, from_file, to_rank, to_file = utils.unpack(move)

    # Calculate the move's delta in rank and file
    delta = (to_rank - from_rank, to_file - from_file)

    is_horizontal = delta[0] == 0
    is_vertical = delta[1] == 0
    is_diagonal = abs(delta[0]) == abs(delta[1])
    is_queen_move = (is_horizontal or is_vertical or is_diagonal)

    if not is_queen_move:
        return None

    direction = tuple(np.sign(delta))
    distance = np.max(np.abs(delta))

    # Calculate the move type index
    direction_idx = queen_directions.index(direction)
    distance_idx = distance - 1
    move_type = np.ravel_multi_index(
        multi_index=([direction_idx, distance_idx]),
        dims=(8, 7)
    )

    # Encode the move to an integer
    action = np.ravel_multi_index(
        multi_index=(from_rank, from_file, move_type),
        dims=(8, 8, 73)
    )

    return action


def encode_knight(move: chess.Move) -> Optional[int]:
    """
    Encodes a knight's move in chess to a unique integer representation.

    Args:
        move (chess.Move): A chess move to be encoded.

    Returns:
        int: An integer representing the encoded knight move, or None if the move is not a knight move.
    """

    # num_types = 8 -> There are 8 types of moves with knight
    type_offset = 56  # Offset for knight moves in the action array.

    # Directions a knight can move, as deltas in rank and file.
    knight_directions = utils.IndexedTuple(
        (+2, +1), (+1, +2), (-1, +2), (-2, +1),
        (-2, -1), (-1, -2), (+1, -2), (+2, -1),
    )

    # Unpack the move's from and to squares
    from_rank, from_file, to_rank, to_file = utils.unpack(move)

    # Calculate the move's delta in rank and file
    delta = (to_rank - from_rank, to_file - from_file)
    if delta not in knight_directions:
        return None

    # Determine the type of knight move
    knight_move_type = knight_directions.index(delta)
    move_type = type_offset + knight_move_type

    # Encode the move to an integer
    action = np.ravel_multi_index(
        multi_index=(from_rank, from_file, move_type),
        dims=(8, 8, 73)
    )

    return action


def encode_underpromotion(move: chess.Move) -> Optional[int]:
    """
    Encodes an underpromotion move in chess to a unique integer representation.

    Args:
        move (chess.Move): A chess move to be encoded.

    Returns:
        int: An integer representing the encoded underpromotion move, or None if the move is not an underpromotion.
    """
    # NUM_TYPES = 9 -> 3 directions * 3 piece types (Knight, Bishop, Rook)
    type_offset = 64
    directions = utils.IndexedTuple(-1, 0, +1)  # Left, straight, right
    promotions = utils.IndexedTuple(
        chess.KNIGHT,
        chess.BISHOP,
        chess.ROOK
    )

    # Unpack the move's from and to squares
    from_rank, from_file, to_rank, to_file = utils.unpack(move)

    # Check if the move is a valid underpromotion
    is_underpromotion = (
            move.promotion in promotions and
            from_rank == 6 and
            to_rank == 7
    )

    if not is_underpromotion:
        return None

    # Determine the direction and type of underpromotion
    delta_file = to_file - from_file
    direction_idx = directions.index(delta_file)
    promotion_idx = promotions.index(move.promotion)

    # Calculate the underpromotion type index
    underpromotion_type = np.ravel_multi_index(
        multi_index=(direction_idx, promotion_idx),
        dims=(3, 3)
    )

    # Encode the move to an integer
    move_type = type_offset + underpromotion_type
    encoded_action = np.ravel_multi_index(
        multi_index=(from_rank, from_file, move_type),
        dims=(8, 8, 73)
    )

    return encoded_action


def encode_move(move_uci: str, board: chess.Board) -> int:
    """
    Encodes a chess move into a unique integer representation.

    Args:
        move_uci (str): The chess move in UCI (Universal Chess Interface) format.
        board (chess.Board): The current chess board state.

    Returns:
        int: An integer representing the encoded move.

    Raises:
        ValueError: If the move is not valid or cannot be encoded.
    """
    # Convert the UCI format move to a chess.Move object
    move = chess.Move.from_uci(move_uci)

    # If it's Black's turn, rotate the board to encode from Black's perspective
    if board.turn == chess.BLACK:
        move = utils.rotate(move)

    # Try encoding the move as a queen move
    action = encode_queen(move)

    # If not a queen move, try encoding as a knight move
    if action is None:
        action = encode_knight(move)

    # If not a knight move, try encoding as an underpromotion
    if action is None:
        action = encode_underpromotion(move)

    # If the move couldn't be encoded, raise an error
    if action is None:
        raise ValueError(f"Move {move_uci} is not a valid move or it cannot be encoded!")

    return action


def encode_board(board: chess.Board) -> np.array:
    """
    Converts a chess board into a numpy array representation.

    Each piece type and color is encoded in a separate plane of the array.
    The first six planes represent the pieces of White, and the next six
    represent the pieces of Black. The last two planes are used for
    repetition counters.

    Args:
        board (chess.Board): The chess board to encode.

    Returns:
        np.array: A numpy array representing the board.
    """
    array = np.zeros((8, 8, 14), dtype=int)

    for square, piece in board.piece_map().items():
        rank, file = chess.square_rank(square), chess.square_file(square)
        piece_type, color = piece.piece_type, piece.color

        # Determine the offset based on the piece color
        offset = 0 if color == chess.WHITE else 6

        # Adjust the index for piece type (chess library uses 1-based indexing)
        idx = piece_type - 1

        # Set the corresponding array cell to 1 for the piece
        array[rank, file, idx + offset] = 1

    # Encode repetition counters for two-fold and three-fold repetitions
    array[:, :, 12] = board.is_repetition(2)
    array[:, :, 13] = board.is_repetition(3)

    return array


def encode_board_from_fen(fen: str) -> np.array:
    """
    Converts a FEN string to a numpy array representation of the chess board.

    Args:
        fen (str): The FEN string representing a chess board position.

    Returns:
        np.array: A numpy array representing the board.
    """
    board = chess.Board(fen)
    return encode_board(board)


def encode_all_moves_and_positions():
    """
    Encodes all moves and positions from the 'data/raw_data' folder.

    This function reads each file in the raw_data folder, encodes the moves and
    positions, and saves the encoded data in the 'data/prepared_Data' folder.
    """
    board = chess.Board()

    files = sorted(os.listdir(RAW_DATA_PATH))
    for idx, filename in enumerate(files):
        positions_and_moves = np.load(f'{RAW_DATA_PATH}/{filename}', allow_pickle=True)
        positions = positions_and_moves[:, 0]
        moves = positions_and_moves[:, 1]

        encoded_positions = []
        encoded_moves = []

        board.reset()
        for position, move in zip(positions, moves):
            try:
                # Encode the move and position
                encoded_position = encode_board(board)
                encoded_move = encode_move(move, board)

                encoded_positions.append(encoded_position)
                encoded_moves.append(encoded_move)

                board.push_uci(move)
            except:
                print(f'Error in file: {filename}, Move: {move}, Position: {position}, Index: {idx}')
                break

        np.save(f'{PREPARED_DATA_PATH}/moves{idx}', np.array(encoded_moves))
        np.save(f'{PREPARED_DATA_PATH}/positions{idx}', np.array(encoded_positions))


def encode_test():
    data = np.loadtxt(f'{DATA_PATH}/moves.txt', delimiter=',', dtype=str)  # Change dtype as needed
    moves = data[:, 0]
    positions = data[:, 1]

    board = chess.Board()
    board.reset()

    encoded_positions = []
    encoded_moves = []

    for position, move in zip(positions, moves):
        try:
            # Encode the move and position
            encoded_position = encode_board(board)
            encoded_move = encode_move(move, board)

            encoded_positions.append(encoded_position)
            encoded_moves.append(encoded_move)

            board.push_uci(move)
        except:
            print(f'Move: {move}, Position: {position}')
            break


if __name__ == '__main__':
    encode_all_moves_and_positions()
