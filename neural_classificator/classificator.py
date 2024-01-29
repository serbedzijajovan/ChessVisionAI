from keras.layers import Dense
from keras.models import Sequential
from torch.nn import Sequential

from tensorflow.keras.optimizers import SGD


def scale_to_range(image):
    return image / 255


def matrix_to_vector(image):
    return image.flatten()


def prepare_for_ann(regions):
    ready_for_ann = []
    for region in regions:
        scale = scale_to_range(region)
        ready_for_ann.append(matrix_to_vector(scale))
    return ready_for_ann


def create_ann(output_size):
    ann = Sequential()
    ann.add(Dense(128, input_dim=784, activation='sigmoid'))
    ann.add(Dense(output_size, activation='sigmoid'))
    return ann


def train_ann(ann, X_train, y_train, epochs):
    X_train = np.array(X_train, np.float32)  # dati ulaz
    y_train = np.array(y_train, np.float32)  # zeljeni izlazi na date ulaze

    print("\nTraining started...")
    sgd = SGD(learning_rate=0.01, momentum=0.9)
    ann.compile(loss='mean_squared_error', optimizer=sgd)
    ann.fit(X_train, y_train, epochs=epochs, batch_size=1, verbose=0, shuffle=False)
    print("\nTraining completed...")
    return ann
