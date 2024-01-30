import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import TRAINING_PATH, PREPARED_DATA_PATH, SAVED_MODELS_PATH
from model import ChessModel

FRACTION_OF_DATA = 1
BATCH_SIZE = 4


def load_training_data():
    """
    Loads and prepares training data from the 'data/prepared_Data' folder.

    The data is split into training and test sets and loaded into PyTorch
    DataLoader objects for efficient training.
    """
    all_moves = []
    all_boards = []

    files = os.listdir(PREPARED_DATA_PATH)
    num_of_each = len(files) // 2  # Half are moves, the other half are positions

    # Load moves and positions from each file
    for i in range(num_of_each):
        try:
            moves = np.load(f"{PREPARED_DATA_PATH}/moves{i}.npy", allow_pickle=True)
            boards = np.load(f"{PREPARED_DATA_PATH}/positions{i}.npy", allow_pickle=True)
            if len(moves) != len(boards):
                print(f"ERROR ON i = {i}, {len(moves)} moves, {len(boards)} boards")
            all_moves.extend(moves)
            all_boards.extend(boards)
        except Exception as e:
            print(f"Error: could not load {i}, but is still going. Exception: {e}")

    # Truncate data based on the specified fraction
    data_limit = int(len(all_moves) * FRACTION_OF_DATA)
    all_moves = np.array(all_moves)[:data_limit]
    all_boards = np.array(all_boards)[:data_limit]
    assert len(all_moves) == len(all_boards), "Moves and boards must be of the same length"

    # Split data into training and test sets
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    all_boards = torch.from_numpy(all_boards).to(device)
    all_moves = torch.from_numpy(all_moves).to(device)

    # Create TensorDatasets and DataLoaders
    train_data_idx = int(len(all_moves) * 0.8)
    training_set = torch.utils.data.TensorDataset(all_boards[:train_data_idx], all_moves[:train_data_idx])
    test_set = torch.utils.data.TensorDataset(all_boards[train_data_idx:], all_moves[train_data_idx:])
    training_loader = torch.utils.data.DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    return training_loader, validation_loader


def train_one_epoch(model, optimizer, loss_fn, epoch_index, tb_writer, training_loader):
    """
    Trains the model for one epoch.

    Args:
        model: The neural network model being trained.
        optimizer: The optimizer used for training.
        loss_fn: The loss function used for training.
        epoch_index: The current epoch index.
        tb_writer: TensorBoard writer for logging.
        training_loader: DataLoader for the training data.

    Returns:
        float: The average loss for the last logged batch in this epoch.
    """
    running_loss = 0.0
    last_loss = 0.0

    # Iterate over the training data
    for i, (inputs, labels) in enumerate(training_loader):
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass: compute predictions and loss
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

        # Backward pass: compute gradient and update weights
        loss.backward()
        optimizer.step()

        # Accumulate loss and log
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000  # Average loss per batch
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.0

    return last_loss


def create_best_model_file():
    """
    Creates a file to store information about the best model if it doesn't exist.

    The file 'bestModel.txt' is created in the 'saved_models' directory with an initial
    high loss value and a placeholder for the model's file path.
    """
    # Create 'saved_models' directory if it doesn't exist
    folder_path = Path(SAVED_MODELS_PATH)
    folder_path.mkdir(exist_ok=True)

    # Path to the best model file
    best_model_path = folder_path / 'bestModel.txt'

    # Create 'bestModel.txt' if it doesn't exist
    if not best_model_path.exists():
        with best_model_path.open("w") as f:
            f.write("10000000\n")  # Initial high loss value
            f.write("testPath")  # Placeholder for model's file path


def save_best_model(validation_loss, best_model_path):
    """
    Saves the information about the best model to a file.

    Args:
        validation_loss: The validation loss of the best model.
        best_model_path: The file path to the best model.
    """
    # Path to the file storing the best model information
    best_model_info_path = f"{SAVED_MODELS_PATH}/bestModel.txt"

    # Save the best model's validation loss and file path
    with open(best_model_info_path, "w") as f:
        f.write(f"{validation_loss}\n")
        f.write(best_model_path)

    print(f"NEW BEST MODEL FOUND WITH LOSS: {validation_loss}")


def retrieve_best_model_info():
    """
    Retrieves information about the best model from a file.

    Returns:
        tuple: A tuple containing the best model's loss and file path.
    """
    # Path to the file storing the best model information
    best_model_info_path = f'{SAVED_MODELS_PATH}/bestModel.txt'

    # Read the best model's loss and file path
    with open(best_model_info_path, "r") as f:
        best_loss = float(f.readline())
        best_model_path = f.readline().strip()

    return best_loss, best_model_path


EPOCHS = 10
LEARNING_RATE = 0.01
MOMENTUM = 0.9


def run_training():
    """
    Runs the training process for the model.
    """
    training_loader, validation_loader = load_training_data()

    # Create file to store best model information
    create_best_model_file()
    best_loss, _ = retrieve_best_model_info()

    # Prepare for training
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(f'{TRAINING_PATH}/runs/fashion_trainer_{timestamp}')
    model = ChessModel()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_validation_loss = 1_000_000.

    # Training loop
    for epoch in tqdm(range(EPOCHS)):
        print(f'EPOCH {epoch + 1}:') if epoch % 5 == 0 else None

        # Training phase
        model.train()
        avg_loss = train_one_epoch(model, optimizer, loss_fn, epoch, writer, training_loader)

        # Validation phase
        model.eval()
        total_vloss = 0.0
        with torch.no_grad():
            for vinputs, vlabels in validation_loader:
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                total_vloss += vloss.item()

        avg_vloss = total_vloss / len(validation_loader)

        # Print and log loss every 5 epochs
        if epoch % 5 == 0:
            print(f'LOSS train {avg_loss} valid {avg_vloss}')

        writer.add_scalars('Training vs. Validation Loss',
                           {'Training': avg_loss, 'Validation': avg_vloss},
                           epoch + 1)
        writer.flush()

        # Save best model
        if avg_vloss < best_validation_loss:
            best_validation_loss = avg_vloss
            if best_loss > best_validation_loss:
                model_path = f'{SAVED_MODELS_PATH}/model_{timestamp}_{epoch}'
                torch.save(model.state_dict(), model_path)
                save_best_model(best_validation_loss, model_path)

    print(f"\n\nBEST VALIDATION LOSS FOR ALL MODELS: {best_loss}")


if __name__ == '__main__':
    run_training()
