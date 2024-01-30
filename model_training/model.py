import chess
import torch
import torch.nn as nn

from model_training.decoding import decode_move
from model_training.encoding import encode_board


class ChessModel(torch.nn.Module):
    def __init__(self):
        super(ChessModel, self).__init__()
        self.input_size = 896  # Flattened board size: 8 x 8 x 14
        self.output_size = 4672  # Number of unique moves (action space)

        # Model architecture
        self.activation = nn.ReLU()
        self.linear1 = nn.Linear(self.input_size, 1000)
        self.linear2 = nn.Linear(1000, 1000)
        self.linear3 = nn.Linear(1000, 1000)
        self.linear4 = nn.Linear(1000, 200)
        self.linear5 = nn.Linear(200, self.output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (Tensor): Input tensor of shape (batch size, input_size).

        Returns:
            Tensor: Output tensor of the model.
        """
        x = x.to(torch.float32)
        x = x.reshape(x.shape[0], -1)
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        x = self.activation(self.linear3(x))
        x = self.activation(self.linear4(x))
        x = self.linear5(x)
        return x

    def predict(self, board: chess.Board):
        """
        Predicts the best move for a given chess board.

        Args:
            board (chess.Board): The chess board to predict the move for.

        Returns:
            chess.Move: The predicted best move, or None if no legal move is found.
        """
        with torch.no_grad():
            encoded_board = encode_board(board)
            encoded_board = torch.from_numpy(encoded_board.reshape(1, -1))
            res = self.forward(encoded_board)
            probs = self.softmax(res).numpy()[0]

            for move_idx in probs.argsort()[::-1]:
                try:
                    uci_move = decode_move(move_idx, board)
                    if uci_move and uci_move in board.legal_moves:
                        return uci_move
                except:
                    continue

            # No legal moves found
            return None
