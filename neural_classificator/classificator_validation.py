import glob
import os

import cv2
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, \
    classification_report

from config import DATA_PATH
from neural_classificator.classificators import ANNClassifier, HOGClassifier
from neural_classificator.square_detection import process_frame

ALPHABET = 'rnbqkpeRNBQKP'


# ******************** IMAGE PROCESSING ********************

def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


# ******************** ANN UTILS ********************


def fen_to_pieces(fen_position):
    """Converts a FEN position to a list of characters representing pieces."""
    char_board = []
    for char in fen_position:
        if char.isdigit():
            for _ in range(int(char)):
                char_board.append('e')
        elif char in ALPHABET:
            char_board.append(char)

    return char_board


# ******************** DATA HANDLING ********************

def get_training_data():
    """Loads and processes training data."""
    train_dir = os.path.join(DATA_PATH, 'pictures')
    images, labels = [], []

    for img_name in os.listdir(train_dir):
        label = img_name[2] if img_name[0] in ('b', 'w') else img_name[0]
        label = label.upper() if img_name[0] == 'w' else label

        img_path = os.path.join(train_dir, img_name)
        img = load_image(img_path)
        images.append(img)
        labels.append(label)

    return np.array(images), np.array(labels)


def process_video(video_path, fen_positions, classifier, y_pred, y_true):
    """Processes each frame of the video and predicts using ANN."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames - 1 != len(fen_positions):
        print(f"Mismatch: Number Of Frames: {total_frames - 1}, Board Positions: {len(fen_positions)}")
        cap.release()
        return None

    for i in range(total_frames - 1):
        ret, frame = cap.read()
        if not ret:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        squares = process_frame(rgb_frame)
        predicted_pieces = classifier.predict(squares)
        fen_pieces = fen_to_pieces(fen_positions[i])

        y_pred.extend(predicted_pieces)
        y_true.extend(fen_pieces)

    cap.release()


def extract_positions(fen_file_path):
    positions = []

    with open(fen_file_path, 'r') as file:
        for line in file:
            position = line.strip().split(' ')[0]
            positions.append(position)

    return positions


def create_and_train_classifier(classifier_name):
    images, labels = get_training_data()
    if classifier_name == "ANN":
        model_path = f'{DATA_PATH}/classificator/ann.keras'
        classifier = ANNClassifier(model_path)
    else:
        model_path = f'{DATA_PATH}/classificator/hog.keras'
        classifier = HOGClassifier(model_path)

    if not classifier.is_trained():
        classifier.train(images, labels)

    return classifier


def print_summary(classificator, y_pred, y_true):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    print(f"******************** {classificator} summary ********************")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print(classification_report(y_true, y_pred))


if __name__ == '__main__':
    classifier_name = "ANN"
    classifier = create_and_train_classifier(classifier_name)

    # Path to the folder containing .gif files
    folder_path = f'{DATA_PATH}/classificator-validation'
    pattern = os.path.join(folder_path, '*.gif')
    gif_files = glob.glob(pattern)

    y_pred = []
    y_true = []
    for video_path in gif_files:
        fen_file_path = video_path.rsplit('.', 1)[0] + '.txt'
        fen_positions = extract_positions(fen_file_path)
        process_video(video_path, fen_positions, classifier, y_pred, y_true)

    print_summary(classifier_name, y_pred, y_true)
