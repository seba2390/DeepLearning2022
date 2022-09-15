__author__ = "Sebastian Yde Madsen"
__copyright__ = "Copyright 2022, Deep Learning Project"
__credits__ = "Sebastian Yde Madsen"
__license__ = "MIT: https://opensource.org/licenses/MIT"
__version__ = "2.0"
__maintainer__ = "Sebastian Yde Madsen"
__email__ = "madsen3008@gmail.com"
__status__ = "Development"
__description__ = """ Binary classification on images of zeroes and ones filtered from the
                      MNIST handwritten digits' dataset [1], implementing Logistic Regression with
                      Binary Cross Entropy as a Loss function.
                      [1]: https://yann.lecun.com/exdb/mnist/ """

from my_helper_functions import *
import numpy as np
from typing import Callable
import matplotlib.pyplot as plt
from keras.datasets import mnist
from tqdm import tqdm

_SEED = 12256


def linear(X: np.ndarray,
           w: np.ndarray,
           b: float) -> np.ndarray:
    assert type(X) is np.ndarray

    """ Linear helper function.

            Parameters:
            -----------
                x: np.ndarray vector of size (nr_features, 1)
                w: np.ndarray vector of size (nr_features, 1)
                b: float value

            Returns:
            --------
                result: np.ndarray of size (nr_data_points,1) """

    _NR_DATA_POINTS = X.shape[0]
    _result = (w.T @ X.T).T + (b * np.ones(_NR_DATA_POINTS))
    assert not np.isnan(_result.any()), f'Linear result should be actual number but is: {_result}'
    return _result


def sigmoid(f: Callable,
            X: np.ndarray,
            w: np.ndarray,
            b: float) -> np.ndarray:
    assert type(X) is np.ndarray

    """ Sigmoid helper function.

            Parameters:
            -----------
                f: std. python function callable
                x: np.ndarray vector of size (nr_features, 1)
                w: np.ndarray vector of size (nr_features, 1)
                b: float value

            Returns:
            --------
                result: np.ndarray of size (nr_data_points, 1) """

    _NR_DATA_POINTS = X.shape[0]
    _result = np.power((np.exp(-f(X, w, b)) + np.ones(_NR_DATA_POINTS)), -1.0)
    assert not np.isnan(_result.any()) and not np.isinf(_result.any()), f'Return from sigmoid func is: {_result}'
    return _result


def binary_cross_entropy(f: Callable,
                         X: np.ndarray,
                         y: np.ndarray,
                         w: np.ndarray,
                         b: float) -> np.ndarray:
    assert X.shape[1] == len(w), "w and X aren't of matching shapes."
    assert X.shape[0] == len(y), "X and y aren't of matching shapes."

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
    _NR_DATA_POINTS = X.shape[0]
    _ones = np.ones(_NR_DATA_POINTS)
    _sigma = sigmoid(f, X, w, b)
    _result = np.mean(-(y * np.log(_sigma) + (_ones - y) * np.log(_ones - _sigma)))
    assert not np.isnan(_result.any()) and not np.isinf(_result.any()), f'Return from Loss func is: {_result} '
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
    _NR_FEATURES = X.shape[1]
    _diff = np.reshape(sigmoid(f, X, w, b) - y, newshape=(len(y), 1))  # Faster that calling (y-sigmoid) again
    _w_grad = np.sum(((_diff @ np.ones((_NR_FEATURES, 1)).T) * X), axis=0)
    _b_deriv = np.sum(_diff)
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
    _y_estimator[sigmoid(f, X, w, b) > 0.5] = 1.0
    _accuracy = np.mean(_y_estimator == y)
    return _accuracy


training_fraction = 0.8
data_fraction = 0.5
(X_train, y_train), (X_test, y_test) = data_load(training_fraction, data_fraction, _SEED)
print('X_train: ' + str(X_train.shape), type(X_train))
print('y_train: ' + str(y_train.shape), type(y_train))
print('X_test: ' + str(X_test.shape), type(X_test))
print('y_test: ' + str(y_test.shape), type(y_test))

np.random.seed(_SEED)
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

fig, ax = plt.subplots(1, 1, figsize=(14, 7))

x_plot = [i + 1 for i in range(nEpoch)]

ax.plot(x_plot, train_losses, label="Training Loss")
ax.plot(x_plot, train_accuracies, label="Training Accuracy")

ax.plot(x_plot, test_losses, label="Test Loss")
ax.plot(x_plot, test_accuracies, label="Test Accuracy")

ax.set_xlabel("Epoch nr.")
ax.legend()

plt.show()
