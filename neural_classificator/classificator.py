import os

import cv2
import numpy as np
from keras.layers import Dense
from keras.models import Sequential

from tensorflow.keras.optimizers import SGD

from config import DATA_PATH
from neural_classificator.square_detection import process_frame

ALPHABET = 'rnbqkpeRNBQKP'


def scale_to_range(image):
    """Scales image values to range [0, 1]."""
    return image / 255


def matrix_to_vector(image):
    """Flattens the image matrix to a vector."""
    return image.flatten()


def prepare_for_ann(images):
    """Prepares image regions for ANN by scaling and flattening."""
    return [matrix_to_vector(scale_to_range(image)) for image in images]


def create_and_train_ann(X_train, y_train, output_size=13, epochs=2000):
    """Creates and trains an artificial neural network."""
    ann = Sequential()
    ann.add(Dense(128, input_dim=8100, activation='sigmoid'))
    ann.add(Dense(output_size, activation='sigmoid'))
    ann.compile(loss='mean_squared_error', optimizer=SGD(learning_rate=0.01, momentum=0.9))
    ann.fit(np.array(X_train, np.float32), np.array(y_train, np.float32),
            epochs=epochs,
            batch_size=1,
            verbose=0,
            shuffle=False)

    return ann


def winner(output):
    """Returns the index of the highest value in the output."""
    return np.argmax(output)


def convert_image_to_binary(image):
    """Loads an image and converts it to a binary image."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 70, 255, cv2.THRESH_BINARY)
    return binary_image


def load_binary_image(path):
    """Loads an image and converts it to a binary image."""
    image = cv2.imread(path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 70, 255, cv2.THRESH_BINARY)
    return binary_image


def one_hot_encode(label, alphabet):
    """One-hot encodes a given label using the alphabet."""
    vec = np.zeros(len(alphabet), dtype=np.float32)
    vec[alphabet.index(label)] = 1
    return vec


def get_training_data():
    """Loads and processes training data."""
    train_dir = os.path.join(DATA_PATH, 'pictures')
    images, labels = [], []

    for img_name in os.listdir(train_dir):
        label = img_name[2] if img_name[0] in ('b', 'w') else img_name[0]
        label = label.upper() if img_name[0] == 'w' else label

        img_path = os.path.join(train_dir, img_name)
        img = load_binary_image(img_path)
        images.append(img)
        labels.append(one_hot_encode(label, ALPHABET))

    return np.array(images), np.array(labels)


def process_video(video_path):
    """Processes each frame of the video and predicts using ANN."""
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        squares = process_frame(frame)
        processed_squares = [convert_image_to_binary(square) for square in squares]
        output = ann.predict(np.array(prepare_for_ann(processed_squares), np.float32), verbose=0)

        for i, square in enumerate(squares):
            print(ALPHABET[winner(output[i])])
            cv2.imshow('Processed Frame', square)
            cv2.waitKey(0)

    cap.release()


if __name__ == '__main__':
    images, labels = get_training_data()
    ann = create_and_train_ann(prepare_for_ann(images), labels)
    gif_path = '../data/videos/board (2).gif'
    process_video(gif_path)
