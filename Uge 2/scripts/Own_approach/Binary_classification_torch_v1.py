__author__ = "Sebastian Yde Madsen"
__copyright__ = "Copyright 2022, Deep Learning Project"
__credits__ = "Sebastian Yde Madsen"
__license__ = "MIT: https://opensource.org/licenses/MIT"
__version__ = "3.0"
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
import torch as tc

_SEED = 123


def linear(X: tc.Tensor,
           w: tc.Tensor,
           b: float) -> tc.Tensor:
    assert type(X) is tc.Tensor and type(w) is tc.Tensor, "X and w should be torch tensor type objects."
    assert X.shape[1] == w.shape[0], "X and w doesn't have matching dimensions."
    assert w.shape[1] == 1, "w should by formatted as column vector w. shape: (nr_features, 1)"

    """ Linear helper function.

            Parameters:
            -----------
                X: torch tensor of size (nr_data_points, nr_features)
                w: torch tensor of size (nr_features, 1)
                b: float value

            Returns:
            --------
                _result: torch tensor of size (nr_data_points,1) """

    _NR_DATA_POINTS = X.shape[0]
    _prod = tc.matmul(tc.transpose(w, 0, 1), tc.transpose(X, 0, 1))
    _result = tc.transpose(_prod, 0, 1) + tc.full((_NR_DATA_POINTS, 1), b)
    return _result


def sigmoid(f: Callable,
            X: tc.Tensor,
            w: tc.Tensor,
            b: float) -> tc.Tensor:
    assert type(X) is tc.Tensor and type(w) is tc.Tensor, "X and w should be torch tensor type objects."
    assert X.shape[1] == w.shape[0], "X and w doesn't have matching dimensions."
    assert w.shape[1] == 1, "w should by formatted as column vector w. shape: (nr_features, 1)"

    """ Sigmoid helper function.

            Parameters:
            -----------
                f: std. python function callable
                X: torch tensor of size (nr_data_points, nr_features)
                w: torch tensor of size (nr_features, 1)
                b: float value

            Returns:
            --------
                _result: torch tensor of size (nr_data_points,1) """

    _NR_DATA_POINTS = X.shape[0]
    _ones = col_vector([1 for i in range(_NR_DATA_POINTS)], tc.float)
    _result = tc.pow((tc.exp(-f(X, w, b)) + _ones), - 1.0)
    return _result


def binary_cross_entropy(f: Callable,
                         X: tc.Tensor,
                         y: tc.Tensor,
                         w: tc.Tensor,
                         b: float) -> tc.Tensor:
    assert type(X) is tc.Tensor and type(w) is tc.Tensor, "X and w should be torch tensor type objects."
    assert X.shape[1] == w.shape[0], "X and w doesn't have matching dimensions."
    assert w.shape[1] == 1, "w should by formatted as column vector w. shape: (nr_features, 1)"
    assert X.shape[0] == y.shape[0], "X and y doesn't have matching dimensions."
    assert y.shape[1] == 1, "y should by formatted as column vector w. shape: (nr_features, 1)"

    """ Binary Cross Entropy Loss function.

            Parameters:
            -----------
                f: std. python function callable
                X: torch tensor of size (nr_data_points, nr_features)
                y: torch tensor of size (nr_data_points, 1)
                w: torch tensor of size (nr_features, 1)
                b: float value

            Returns:
            --------
                _result: torch tensor of size (1x1)  """
    _NR_DATA_POINTS = X.shape[0]
    _ones = col_vector([1 for i in range(_NR_DATA_POINTS)], tc.float)
    _sigma = sigmoid(f, X, w, b)
    _result = tc.mean(-(y * tc.log(_sigma) + (_ones - y) * tc.log(_ones - _sigma)))
    return _result


def binary_cross_entropy_grad(f: Callable,
                              X: tc.Tensor,
                              y: tc.Tensor,
                              w: tc.Tensor,
                              b: float) -> tuple[tc.Tensor, tc.Tensor]:
    assert type(X) is tc.Tensor and type(w) is tc.Tensor, "X and w should be torch tensor type objects."
    assert X.shape[1] == w.shape[0], "X and w doesn't have matching dimensions."
    assert w.shape[1] == 1, "w should by formatted as column vector w. shape: (nr_features, 1)"
    assert X.shape[0] == y.shape[0], "X and y doesn't have matching dimensions."
    assert y.shape[1] == 1, "y should by formatted as column vector w. shape: (nr_features, 1)"

    """ Binary Cross Entropy Loss function gradient.

            Parameters:
            -----------
                f: std. python function callable
                X: torch tensor of size (nr_data_points, nr_features)
                y: torch tensor of size (nr_data_points, 1)
                w: torch tensor of size (nr_features, 1)
                b: float value

            Returns:
            --------
                _result: tuple(torch tensor, torch tensor) """

    _NR_DATA_POINTS = X.shape[0]
    _NR_FEATURES = X.shape[1]
    _ones = col_vector([1 for i in range(_NR_FEATURES)], tc.float)
    _diff = sigmoid(f, X, w, b) - y  # Faster that calling (y-sigmoid) again
    _w_grad = tc.sum(tc.matmul(_diff, tc.transpose(_ones, 0, 1)) * X, 0)
    _b_deriv = tc.sum(_diff)
    return _w_grad / _NR_DATA_POINTS, _b_deriv / _NR_DATA_POINTS


my_x = tc.tensor(data=[[2, 1, 1], [1, 3, 1]], dtype=tc.float)
my_w = col_vector([2, 2, 2], tc.float)
my_y = col_vector([0, 1], tc.float)
my_b = -2.8

print(binary_cross_entropy_grad(linear, my_x, my_y, my_w, my_b))


def accuracy(f: Callable,
             X: tc.Tensor,
             y: tc.Tensor,
             w: tc.Tensor,
             b: float) -> float:
    assert type(X) is tc.Tensor and type(w) is tc.Tensor, "X and w should be torch tensor type objects."
    assert X.shape[1] == w.shape[0], "X and w doesn't have matching dimensions."
    assert w.shape[1] == 1, "w should by formatted as column vector w. shape: (nr_features, 1)"
    assert X.shape[0] == y.shape[0], "X and y doesn't have matching dimensions."
    assert y.shape[1] == 1, "y should by formatted as column vector w. shape: (nr_features, 1)"

    """ Accuracy function, calculates the fraction of labels 'y', 
        that is correctly calculated on 'X', given w and b.

            Parameters:
            -----------
                f: std. python function callable
                X: torch tensor of size (nr_data_points, nr_features)
                y: torch tensor of size (nr_data_points, 1)
                w: torch tensor of size (nr_features, 1)
                b: float value

           Returns:
           --------
               _accuracy: float) """

    _NR_DATA_POINTS = X.shape[0]
    _y_estimator = col_vector(data=[0 for i in range(_NR_DATA_POINTS)], datatype=tc.float)
    _y_estimator[sigmoid(f, X, w, b) > 0.5] = 1.0
    _accuracy = tc.sum(_y_estimator == y) / len(y)
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

# Transforming to torch objects
X_train = tc.tensor(data=list(X_train), dtype=tc.float)
y_train = col_vector(data=list(y_train), datatype=tc.float)
X_test = tc.tensor(data=list(X_test), dtype=tc.float)
y_test = col_vector(data=list(y_test), datatype=tc.float)
w = col_vector(data=list(w), datatype=tc.float)
b = float(b)

nEpoch = 600
trainRate = 0.5  # pick a number less than 1
train_accuracies, train_losses = tc.zeros(nEpoch, dtype=tc.float), tc.zeros(nEpoch, dtype=tc.float)
test_accuracies, test_losses = tc.zeros(nEpoch, dtype=tc.float), tc.zeros(nEpoch, dtype=tc.float)

for epoch in tqdm(range(nEpoch)):
    # Calculate loss & accuracy
    train_losses[epoch] = binary_cross_entropy(linear, X_train, y_train, w, b)
    train_accuracies[epoch] = accuracy(linear, X_train, y_train, w, b)

    test_losses[epoch] = binary_cross_entropy(linear, X_test, y_test, w, b)
    test_accuracies[epoch] = accuracy(linear, X_test, y_test, w, b)

    # Calculating gradients
    w_grad, b_grad = binary_cross_entropy_grad(linear, X_train, y_train, w, b)
    w_grad = col_vector(data=list(w_grad), datatype=tc.float)  # Setting w grad to col vector

    # Updating w and b
    w -= trainRate * w_grad
    b -= trainRate * b_grad

fig, ax = plt.subplots(1, 1, figsize=(14, 7))

x_plot = [i + 1 for i in range(nEpoch)]

ax.plot(x_plot, list(train_losses), label="Training Loss")
ax.plot(x_plot, list(train_accuracies), label="Training Accuracy")

ax.plot(x_plot, list(test_losses), label="Test Loss")
ax.plot(x_plot, list(test_accuracies), label="Test Accuracy")

ax.set_xlabel("Epoch nr.")
ax.legend()
plt.show()
