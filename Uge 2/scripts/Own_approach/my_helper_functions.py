import torch as tc
import numpy as np
from keras.datasets import mnist


def col_vector(data, datatype, req_grad=False) -> tc.Tensor:
    vector = tc.tensor(data, dtype=datatype, requires_grad=req_grad)
    return tc.reshape(vector, shape=(len(vector), 1))


def row_vector(data, datatype, req_grad=False) -> tc.Tensor:
    vector = tc.tensor(data, dtype=datatype, requires_grad=req_grad)
    return tc.reshape(vector, shape=(1, len(vector)))


def data_load(train_fraction: float,
              data_fraction: float,
              seed: int):
    assert type(train_fraction) is float, f'train_fraction should be type=float but is type={type(train_fraction)}'
    assert 0 < train_fraction < 1, f'train_fraction should be between 0 and 1.'
    assert type(data_fraction) is float, f'data_fraction should be type=float but is type={type(train_fraction)}'
    assert 0 < data_fraction <= 1, f'data_fraction should be between 0 and 1.'

    # The shape of our training data is (60000, 28, 28), this is
    # because there are 60,000 grayscale images with the dimension 28X28. (60000 matrices of 28x28 stacked)
    (train_X, train_y), (test_X, test_y) = mnist.load_data()

    # Reshaping each picture by flattening (28x28) -> (784)
    train_X = np.reshape(a=train_X, newshape=(train_X.shape[0], train_X[0].shape[0] * train_X[1].shape[1]))
    test_X = np.reshape(a=test_X, newshape=(test_X.shape[0], test_X[0].shape[0] * test_X[1].shape[1]))

    # Picking out all the pictures containing 1 and 0, s.t. we can perform binary classification
    train_binary_filter = (train_y == 0) | (train_y == 1)
    train_X = train_X[train_binary_filter]
    train_y = train_y[train_binary_filter]
    test_binary_filter = (test_y == 0) | (test_y == 1)
    test_X = test_X[test_binary_filter]
    test_y = test_y[test_binary_filter]

    # Collecting train_X, test_x and train_y, test_y to choose own training_size/test_size ratio
    X = np.empty(shape=(train_X.shape[0] + test_X.shape[0], test_X.shape[1]), dtype=float)
    X[:train_X.shape[0]] = train_X
    X[train_X.shape[0]:train_X.shape[0] + test_X.shape[0]] = test_X
    y = np.concatenate([train_y, test_y])

    # Normalising each feature vector in X:
    for data_point in range(X.shape[0]):
        X[data_point] *= 1.0 / np.sum(X[data_point])

    # Setting seed for random consistency
    np.random.seed(seed)

    # Shuffling indices randomly
    shuffle_indices = np.array([i for i in range(len(y))])
    np.random.shuffle(shuffle_indices)
    X, y = X[shuffle_indices], y[shuffle_indices]

    # Picking set sizes
    X, y = X[:int(data_fraction * X.shape[0]), :], y[:int(data_fraction * X.shape[0])]

    # Distributing X in train and test sets according to train_fraction
    _X_train, _X_test = X[:int(train_fraction * len(y))], X[int(train_fraction * len(y)):]
    _Y_train, _y_test = y[:int(train_fraction * len(y))], y[int(train_fraction * len(y)):]

    return (_X_train, _Y_train), (_X_test, _y_test)
