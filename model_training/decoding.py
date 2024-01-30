from typing import Optional

import chess
import numpy as np
from gym_chess.alphazero.move_encoding import utils


def decode_queen(action: int) -> Optional[chess.Move]:
    """
    Decodes a queen's move from an integer representation to UCI notation.

    Args:
        action (int): An integer representing the encoded queen move.

    Returns:
        chess.Move: A chess move in UCI notation, or None if the action does not represent a queen move.
    """
    num_types = 56  # 8 directions * 7 squares maximum distance

    # Possible directions for a queen move, encoded as (delta rank, delta file).
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

    # Unravel the flat index to get the original multidimensional indices
    from_rank, from_file, move_type = np.unravel_index(action, (8, 8, 73))

    # Check if the action represents a queen move
    is_queen_move = move_type < num_types
    if not is_queen_move:
        return None

    # Determine the direction and distance of the move
    direction_idx, distance_idx = np.unravel_index(move_type, (8, 7))
    direction = queen_directions[direction_idx]
    distance = distance_idx + 1

    # Calculate the to-square of the move
    delta_rank = direction[0] * distance
    delta_file = direction[1] * distance
    to_rank = from_rank + delta_rank
    to_file = from_file + delta_file

    # Pack the from-square and to-square into UCI notation
    move = utils.pack(from_rank, from_file, to_rank, to_file)
    return move


def decode_knight(action: int) -> Optional[chess.Move]:
    """
    Decodes a knight's move from an integer representation to UCI notation.

    Args:
        action (int): An integer representing the encoded knight move.

    Returns:
        chess.Move: A chess move in UCI notation, or None if the action does not represent a knight move.
    """
    num_types = 8
    type_offset = 56  # Offset for knight moves in the action array.

    # Possible directions for a knight move, encoded as (delta rank, delta file).
    knight_directions = utils.IndexedTuple(
        (+2, +1), (+1, +2), (-1, +2), (-2, +1),
        (-2, -1), (-1, -2), (+1, -2), (+2, -1),
    )

    # Unravel the flat index to get the original multidimensional indices
    from_rank, from_file, move_type = np.unravel_index(action, (8, 8, 73))

    # Check if the action represents a knight move
    is_knight_move = type_offset <= move_type < type_offset + num_types
    if not is_knight_move:
        return None

    # Determine the type of knight move
    knight_move_type = move_type - type_offset
    delta_rank, delta_file = knight_directions[knight_move_type]

    # Calculate the to-square of the move
    to_rank = from_rank + delta_rank
    to_file = from_file + delta_file

    # Pack the from-square and to-square into UCI notation
    move = utils.pack(from_rank, from_file, to_rank, to_file)
    return move


def decode_underpromotion(action: int) -> Optional[chess.Move]:
    """
    Decodes an underpromotion move from an integer representation to UCI notation.

    Args:
        action (int): An integer representing the encoded underpromotion move.

    Returns:
        chess.Move: A chess move in UCI notation, or None if the action does not represent an underpromotion.
    """
    num_types = 9  # 3 directions * 3 piece types (Knight, Bishop, Rook)
    type_offset = 64

    # Possible directions and promotion types for an underpromotion
    directions = utils.IndexedTuple(-1, 0, +1)  # Left, straight, right
    promotions = utils.IndexedTuple(
        chess.KNIGHT,
        chess.BISHOP,
        chess.ROOK
    )

    # Unravel the flat index to get the original multidimensional indices
    from_rank, from_file, move_type = np.unravel_index(action, (8, 8, 73))

    # Check if the action represents an underpromotion
    is_underpromotion = type_offset <= move_type < type_offset + num_types
    if not is_underpromotion:
        return None

    # Determine the direction and type of underpromotion
    underpromotion_type = move_type - type_offset
    direction_idx, promotion_idx = np.unravel_index(underpromotion_type, (3, 3))
    direction = directions[direction_idx]
    promotion = promotions[promotion_idx]

    # Calculate the to-square of the move
    to_rank = from_rank + 1
    to_file = from_file + direction

    # Pack the from-square and to-square into UCI notation and set promotion
    move = utils.pack(from_rank, from_file, to_rank, to_file)
    move.promotion = promotion

    return move


def decode_move(action: int, board: chess.Board) -> Optional[chess.Move]:
    """
    Decodes a chess move from an integer representation to UCI notation.

    Args:
        action (int): An integer representing the encoded chess move.
        board (chess.Board): The current chess board state.

    Returns:
        chess.Move: A chess move in UCI notation.

    Raises:
        ValueError: If the action is not a valid encoded move.
    """
    # Try decoding the move as a queen, knight, or underpromotion move
    move = decode_queen(action)
    is_queen_move = move is not None
    if move is None:
        move = decode_knight(action)
    if move is None:
        move = decode_underpromotion(action)
    if move is None:
        raise ValueError(f"Action {action} is not a valid encoded move")

    # Actions encode moves from the perspective of the current player
    # If it's Black's turn, reorient the move
    if not board.turn:  # Black to move
        move = utils.rotate(move)

    # Check for pawn promotion to queen (implicit in queen move encoding)
    if is_queen_move:
        piece = board.piece_at(move.from_square)
        if piece is None:
            return None

        is_pawn = piece.piece_type == chess.PAWN
        to_rank = chess.square_rank(move.to_square)
        is_promoting_move = (to_rank == 7 and board.turn) or (to_rank == 0 and not board.turn)
        if is_pawn and is_promoting_move:
            move.promotion = chess.QUEEN

    return move
