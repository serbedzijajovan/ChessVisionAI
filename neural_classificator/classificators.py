import os

import cv2
import numpy as np
from keras import Sequential
from keras.models import load_model
from keras.src.layers import Dense
from keras.src.optimizers import SGD
from sklearn.svm import SVC


class ChessPieceClassifier:

    def train(self, images, labels):
        pass

    def predict(self, images):
        pass


class ANNClassifier(ChessPieceClassifier):
    def __init__(self, model_path):
        self._alphabet = 'rnbqkpeRNBQKP'
        self._model_path = model_path
        self._model = self._init_model()

    def _init_model(self):
        if os.path.exists(self._model_path):
            return load_model(self._model_path)
        else:
            return None

    def _create_network(self):
        ann = Sequential()
        ann.add(Dense(128, input_dim=8100, activation='sigmoid'))
        ann.add(Dense(len(self._alphabet), activation='sigmoid'))
        ann.compile(loss='mean_squared_error', optimizer=SGD(learning_rate=0.01, momentum=0.9))

        self._model = ann

    def _one_hot_encode(self, labels):
        one_hot_encoded = np.zeros((len(labels), len(self._alphabet)), dtype=np.float32)
        for i, label in enumerate(labels):
            one_hot_encoded[i, self._alphabet.index(label)] = 1
        return one_hot_encoded

    def _prepare_images_for_ann(self, images):
        converted_images = []
        for image in images:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            _, image = cv2.threshold(image, 70, 255, cv2.THRESH_BINARY)

            scaled_image = image / 255
            flattened_image = scaled_image.flatten()
            converted_images.append(flattened_image)

        return converted_images

    def train(self, images, labels):
        """Expects 2D RGB image and labels as chars"""

        # If not created
        if not self._model:
            self._create_network()

        prepared_images = self._prepare_images_for_ann(images)
        one_hot_encoded_labels = self._one_hot_encode(labels)

        self._model.fit(np.array(prepared_images, np.float32),
                        np.array(one_hot_encoded_labels, np.float32),
                        epochs=2000,
                        batch_size=1,
                        verbose=0,
                        shuffle=False)

        self._model.save(self._model_path)

    def is_trained(self):
        return self._model is not None

    def _decode_predictions(self, winner_indices):
        """Decodes the winner indices to their corresponding labels."""
        return [self._alphabet[index] for index in winner_indices]

    def predict(self, images):
        """Expects 2D RGB images and returns array of chars representing recognized pieces"""
        prepared_images = self._prepare_images_for_ann(images)
        recognized_pieces_probs = self._model.predict(np.array(prepared_images, np.float32), verbose=0)
        winners = [np.argmax(piece_prob) for piece_prob in recognized_pieces_probs]

        return self._decode_predictions(winners)


def _get_hog():
    img_size = (90, 90)
    nbins = 9
    cell_size = (8, 8)
    block_size = (3, 3)
    hog = cv2.HOGDescriptor(_winSize=(img_size[1] // cell_size[1] * cell_size[1],
                                      img_size[0] // cell_size[0] * cell_size[0]),
                            _blockSize=(block_size[1] * cell_size[1],
                                        block_size[0] * cell_size[0]),
                            _blockStride=(cell_size[1], cell_size[0]),
                            _cellSize=(cell_size[1], cell_size[0]),
                            _nbins=nbins)
    return hog


class HOGClassifier(ChessPieceClassifier):
    def __init__(self, model_path):
        self._hog = _get_hog()
        self._model_path = model_path
        self._model = None

    def _create_network(self):
        classifier = SVC(kernel='linear', probability=True)

        self._model = classifier

    def _prepare_images_for_hog(self, images):
        return [self._hog.compute(image) for image in images]

    def train(self, images, labels):
        """Expects 2D RGB image and labels as chars"""

        # If not created
        if not self._model:
            self._create_network()

        prepared_images = self._prepare_images_for_hog(images)
        labels = labels
        self._model.fit(prepared_images, labels)

    def is_trained(self):
        return self._model is not None

    def predict(self, images):
        """Expects 2D RGB images and returns array of chars representing recognized pieces"""

        prepared_images = self._prepare_images_for_hog(images)
        return self._model.predict(prepared_images)
