__author__ = "Sebastian Yde Madsen"
__copyright__ = "Copyright 2022, Deep Learning Project"
__credits__ = "Sebastian Yde Madsen"
__license__ = "MIT: https://opensource.org/licenses/MIT"
__version__ = "5.0"
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

training_fraction = 0.8
data_fraction = 0.5
(X_train, y_train), (X_test, y_test) = data_load(training_fraction, data_fraction, _SEED)
print('X_train: ' + str(X_train.shape), type(X_train))
print('y_train: ' + str(y_train.shape), type(y_train))
print('X_test: ' + str(X_test.shape), type(X_test))
print('y_test: ' + str(y_test.shape), type(y_test))

np.random.seed(_SEED)
_w = np.random.uniform(low=0, high=1, size=X_train.shape[1])
_w *= 1.0 / np.sum(_w)  # normalizing
_b = np.random.uniform(low=0, high=1, size=1)[0]

# Transforming to torch objects
X_train = tc.tensor(data=list(X_train), dtype=tc.float64)
y_train = col_vector(data=list(y_train), datatype=tc.float64)
X_test = tc.tensor(data=list(X_test), dtype=tc.float64)
y_test = col_vector(data=list(y_test), datatype=tc.float64)

w_col_vector_data = [[_w[i]] for i in range(len(_w))]  # To ensure the instantiation of w is a leaf node
w = tc.tensor(data=w_col_vector_data, dtype=tc.float64, requires_grad=True)
b = tc.tensor(data=[float(_b)], dtype=tc.float64, requires_grad=True)

nEpoch = 125
trainRate = 0.1  # pick a number less than 1
train_accuracies, train_losses = tc.zeros(nEpoch, dtype=tc.float64), tc.zeros(nEpoch, dtype=tc.float)
test_accuracies, test_losses = tc.zeros(nEpoch, dtype=tc.float), tc.zeros(nEpoch, dtype=tc.float)

_NR_FEATURES = X_train.shape[1]
_NR_TRAIN_DATA_POINTS, _NR_TEST_DATA_POINTS = X_train.shape[0], X_test.shape[0]
_train_N_ones = tc.ones((_NR_TRAIN_DATA_POINTS, 1), requires_grad=True, dtype=tc.float)
_test_N_ones = tc.ones((_NR_TEST_DATA_POINTS, 1), requires_grad=True, dtype=tc.float)
_d_ones = col_vector([1 for i in range(_NR_FEATURES)], tc.float)
for epoch in tqdm(range(nEpoch)):
    # Calculating Training loss
    _prod = tc.matmul(tc.transpose(w, 0, 1), tc.transpose(X_train, 0, 1))
    _lin_term = tc.transpose(_prod, 0, 1) + b * _train_N_ones
    _sigmoid = tc.pow((tc.exp(-_lin_term) + _train_N_ones), - 1.0)
    _BCE_loss = tc.mean(-(y_train * tc.log(_sigmoid) + (_train_N_ones - y_train) * tc.log(_train_N_ones - _sigmoid)))
    _BCE_loss.backward()

    # Updating w and b
    with tc.no_grad():
        w -= trainRate * w.grad
        b -= trainRate * b.grad

    # Calculating training accuracy
    _y_estimator = tc.zeros((_NR_TRAIN_DATA_POINTS, 1), dtype=tc.float64)
    _y_estimator[_sigmoid > 0.5] = 1.0
    _accuracy = tc.sum(_y_estimator == y_train) / len(y_train)

    # Storing training loss and accuracy values
    train_losses[epoch] = _BCE_loss
    train_accuracies[epoch] = _accuracy

    # Calculating Training loss
    _prod = tc.matmul(tc.transpose(w, 0, 1), tc.transpose(X_test, 0, 1))
    _lin_term = tc.transpose(_prod, 0, 1) + b * _test_N_ones
    _sigmoid = tc.pow((tc.exp(-_lin_term) + _test_N_ones), - 1.0)
    _BCE_loss = tc.mean(-(y_test * tc.log(_sigmoid) + (_test_N_ones - y_test) * tc.log(_test_N_ones - _sigmoid)))

    # Calculating training accuracy
    _y_estimator = tc.zeros((_NR_TEST_DATA_POINTS, 1), dtype=tc.float)
    _y_estimator[_sigmoid > 0.5] = 1.0
    _accuracy = tc.sum(_y_estimator == y_test) / len(y_test)

    # Storing test loss and accuracy values
    test_losses[epoch] = _BCE_loss
    test_accuracies[epoch] = _accuracy

fig, ax = plt.subplots(1, 1, figsize=(14, 7))
x_plot = [i + 1 for i in range(nEpoch)]

print(train_losses)
ax.plot(x_plot, train_losses.detach().numpy(), label="Training Loss")
ax.plot(x_plot, train_accuracies.detach().numpy(), label="Training Accuracy")

ax.plot(x_plot, test_losses.detach().numpy(), label="Test Loss")
ax.plot(x_plot, test_accuracies.detach().numpy(), label="Test Accuracy")

ax.set_xlabel("Epoch nr.")
ax.legend()
plt.show()
