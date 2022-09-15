__author__ = "Sebastian Yde Madsen"
__copyright__ = "Copyright 2022, Deep Learning Project"
__credits__ = "Sebastian Yde Madsen"
__license__ = "MIT: https://opensource.org/licenses/MIT"
__version__ = "1.0"
__maintainer__ = "Sebastian Yde Madsen"
__email__ = "madsen3008@gmail.com"
__status__ = "Development"
__description__ = """ Binary classification on images of zeroes and ones filtered from the
                      MNIST handwritten digits' dataset [1], implementing Logistic Regression with
                      Binary Cross Entropy as a Loss function.
                      [1]: https://yann.lecun.com/exdb/mnist/ """

import numpy as np
from typing import Callable
import matplotlib.pyplot as plt
from keras.datasets import mnist
from tqdm import tqdm


def linear(x: np.ndarray,
           w: np.ndarray,
           b: float) -> float:
    assert len(x) == len(w), "w and x aren't of matching shapes"

    """ Linear helper function.

            Parameters:
            -----------
                x: np.ndarray vector of size (nr_features, 1)
                w: np.ndarray vector of size (nr_features, 1)
                b: float value

            Returns:
            --------
                result: float """

    _result = w.T @ x + b
    if np.isnan(_result):
        print(w, x, b)
    assert not np.isnan(_result), f'Linear result should be actual number but is: {_result}'
    return _result


def sigmoid(f: Callable,
            x: np.ndarray,
            w: np.ndarray,
            b: float) -> float:
    assert len(x) == len(w), "w and x aren't of matching shapes"

    """ Sigmoid helper function.

            Parameters:
            -----------
                f: std. python function callable
                x: np.ndarray vector of size (nr_features, 1)
                w: np.ndarray vector of size (nr_features, 1)
                b: float value

            Returns:
            --------
                result: float """
    _result = 1.0 / (1.0 + np.exp(-f(x, w, b)))
    assert not np.isnan(_result) and not np.isinf(_result), f'Return from sigmoid func is: {_result}'
    return _result


def binary_cross_entropy(f: Callable,
                         X: np.ndarray,
                         y: np.ndarray,
                         w: np.ndarray,
                         b: float) -> float:
    assert X.shape[1] == len(w)
    assert X.shape[0] == len(y)

    """ Binary Cross Entropy Loss function.

            Parameters:
            -----------
                f: std. python function callable
                X: np.ndarray matrix of size (nr_data_points, nr_features)
                y: np.ndarray vector of size (nr_data_points, 1)
                w: np.ndarray vector of size (nr_features, 1)
                b: float value

            Returns:
            --------
                result: float """
    _result = 0.0
    _NR_DATA_POINTS = X.shape[0]
    for data_point in range(_NR_DATA_POINTS):
        _sigma = sigmoid(f, X[data_point], w, b)
        _result += - (y[data_point] * np.log(_sigma) + (1 - y[data_point]) * np.log(1 - _sigma))
    _result *= 1.0 / _NR_DATA_POINTS
    assert not np.isnan(_result) and not np.isinf(_result), f'Return from Loss func is: {_result}'
    return _result


def binary_cross_entropy_grad(f: Callable,
                              X: np.ndarray,
                              y: np.ndarray,
                              w: np.ndarray,
                              b: float) -> tuple[np.ndarray, float]:
    assert X.shape[1] == len(w)
    assert X.shape[0] == len(y)

    """ Binary Cross Entropy Loss function gradient.

            Parameters:
            -----------
                f: std. python function callable
                X: np.ndarray matrix of size (nr_data_points, nr_features)
                y: np.ndarray vector of size (nr_data_points, 1)
                w: np.ndarray vector of size (nr_features, 1)
                b: float value

            Returns:
            --------
                tuple of (np.ndarray , float) """

    _NR_DATA_POINTS = X.shape[0]
    _w_grad = np.zeros(len(w), dtype=float)
    _b_deriv = 0.0
    for data_point in range(_NR_DATA_POINTS):
        _diff = (sigmoid(f, X[data_point], w, b) - y[data_point])
        _w_grad += _diff * X[data_point]
        _b_deriv += _diff  # Faster that calling (y-sigmoid) again
    return _w_grad / _NR_DATA_POINTS, _b_deriv / _NR_DATA_POINTS


def accuracy(f: Callable,
             X: np.ndarray,
             y: np.ndarray,
             w: np.ndarray,
             b: float):
    assert X.shape[1] == len(w)
    assert X.shape[0] == len(y)

    """ Accuracy function, calculates the fraction of labels 'y', 
        that is correctly calculated on 'X', given w and b.

               Parameters:
               -----------
                   f: std. python function callable
                   X: np.ndarray matrix of size (nr_data_points, nr_features)
                   y: np.ndarray vector of size (nr_data_points, 1)
                   w: np.ndarray vector of size (nr_features, 1)
                   b: float value

               Returns:
               --------
                   _accuracy: float """
    _NR_DATA_POINTS = X.shape[0]
    _y_estimator = np.zeros(_NR_DATA_POINTS, dtype=float)
    for _data_point in range(_NR_DATA_POINTS):
        if sigmoid(f, X[_data_point], w, b) > 0.5:
            _y_estimator[_data_point] = 1.0
    _accuracy = np.mean(_y_estimator == y)
    return _accuracy


def data_load(train_fraction: float,
              data_fraction: float):
    assert type(train_fraction) is float, f'train_fraction should be type=float but is type={type(train_fraction)}'
    assert 0 < train_fraction < 1, f'train_fraction should be between 0 and 1.'
    assert type(data_fraction) is float, f'data_fraction should be type=float but is type={type(train_fraction)}'
    assert 0 < data_fraction < 1, f'data_fraction should be between 0 and 1.'
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
    np.random.seed(0)

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


training_fraction = 0.8
data_fraction = 0.5
(X_train, y_train), (X_test, y_test) = data_load(training_fraction, data_fraction)
print('X_train: ' + str(X_train.shape), type(X_train))
print('y_train: ' + str(y_train.shape), type(y_train))
print('X_test: ' + str(X_test.shape), type(X_test))
print('y_test: ' + str(y_test.shape), type(y_test))

np.random.seed(0)
w = np.random.uniform(low=0, high=1, size=X_train.shape[1])
w *= 1.0 / np.sum(w)  # normalizing
b = np.random.uniform(low=0, high=1, size=1)[0]

nEpoch = 600
trainRate = 0.5  # pick a number less than 1
train_accuracies, train_losses = np.zeros(nEpoch), np.zeros(nEpoch)
test_accuracies, test_losses = np.zeros(nEpoch), np.zeros(nEpoch)

for epoch in tqdm(range(nEpoch)):

    # Calculate loss & accuracy
    train_losses[epoch] = binary_cross_entropy(linear, X_train, y_train, w, b)
    train_accuracies[epoch] = accuracy(linear, X_train, y_train, w, b)

    test_losses[epoch] = binary_cross_entropy(linear, X_test, y_test, w, b)
    test_accuracies[epoch] = accuracy(linear, X_test, y_test, w, b)

    # Calculating gradients
    w_grad, b_grad = binary_cross_entropy_grad(linear, X_train, y_train, w, b)

    # Updating w and b
    w -= trainRate * w_grad
    b -= trainRate * b_grad


fig, ax = plt.subplots(1,2,figsize=(14,7))

x_plot = [i + 1 for i in range(nEpoch)]

ax[0].set_title("Training Data", size=22)
ax[0].plot(x_plot, train_losses, label="Training Loss")
ax[0].plot(x_plot, train_accuracies, label="Training Accuracy")
ax[0].set_xlabel("Epoch nr.")
ax[0].legend()

ax[1].set_title("Test Data", size=22)
ax[1].plot(x_plot, test_losses, label="Test Loss")
ax[1].plot(x_plot, test_accuracies, label="Test Accuracy")
ax[1].set_xlabel("Epoch nr.")
ax[1].legend()

plt.show()


