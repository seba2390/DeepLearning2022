import numpy as np
import torch as tc
import torchvision as tv
from my_helper_functions_v2 import *
from tqdm import tqdm
import matplotlib.pyplot as plt
from copy import deepcopy

def linear(X: np.ndarray,
           W: np.ndarray) -> np.ndarray:
    assert type(X) is np.ndarray
    """ Linear helper function.

            Parameters:
            -----------
                X: np.ndarray vector of size (nr_data_points, nr_features)
                W: np.ndarray vector of size (nr_features, nr_classes)
                b: float value

            Returns:
            --------
                result: np.ndarray of size (nr_data_points,nr_classes) """
    _result = X @ W
    return _result


def softmax(X: np.ndarray) -> np.ndarray:
    assert type(X) is np.ndarray
    """
    Args:
        X: numpy array shape (n, d) each row is a data point
    Returns:
        res: numpy array shape (n, d)  where each row is the softmax transformation of the corresponding row in X i.e res[i, :] = softmax(X[i, :])
    """

    if X.ndim == 1:
        X = np.reshape(X, newshape=(1, X.shape[0]))
    _z_max = np.amax(X, axis=1, keepdims=True)
    _logsum = np.log(np.sum(np.exp(X - _z_max), axis=1, keepdims=True))
    _result = np.exp(X - _z_max - _logsum)
    assert X.shape == _result.shape
    return _result


def cross_entropy(X: np.ndarray,
                  Y: np.ndarray,
                  W: np.ndarray) -> np.ndarray:
    assert type(X) is np.ndarray
    assert type(Y) is np.ndarray
    assert type(W) is np.ndarray
    _result = -np.mean(np.log(np.sum((Y @ softmax(linear(X, W)).T), axis=0)))
    return _result


def cross_entropy_grad(X: np.ndarray,
                       Y: np.ndarray,
                       W: np.ndarray) -> np.ndarray:
    assert type(X) is np.ndarray
    assert type(Y) is np.ndarray
    assert type(W) is np.ndarray
    _result = -1 / X.shape[0] * X.T @ (Y - softmax(linear(X, W)))
    return _result


def predict(X: np.ndarray,
            W: np.ndarray) -> np.ndarray:
    _y_estimator = np.argmax(softmax(X @ W), axis=1)
    return _y_estimator


def accuracy(X: np.ndarray,
             Y: np.ndarray,
             W: np.ndarray):
    _y_real = np.argmax(Y, axis=1)
    _y_estimator = predict(X, W)
    _result = np.mean(_y_estimator == _y_real)
    return _result


_TRAINING_FRACTION, _DATA_FRACTION = 0.7, 0.1
_NR_DATA_POINTS, _NR_FEATURES = 20, 28 * 28
_NR_EPOCHS, _BATCH_SIZE = 100, 64
_LEARNING_RATE, _SEED = 0.05, 123
_NR_CLASSES = 10

(X_train, y_train), (X_test, y_test) = data_load(_TRAINING_FRACTION, _DATA_FRACTION, _SEED, False, False)
print('X_train: ' + str(X_train.shape), type(X_train))
print('y_train: ' + str(y_train.shape), type(y_train))
print('X_test: ' + str(X_test.shape), type(X_test))
print('y_test: ' + str(y_test.shape), type(y_test))

W = np.random.normal(0, 1, size=(_NR_FEATURES, _NR_CLASSES))
#for col in range(W.shape[1]):
#    W[:, col] *= 1.0/np.sum(W[:, col])

Y_train, Y_test = one_hot_encoding(y_train, _NR_CLASSES), one_hot_encoding(y_test, _NR_CLASSES)

nEpoch = 100
trainRate = 0.05 # pick a number less than 1
train_accuracies, train_losses = np.zeros(nEpoch), np.zeros(nEpoch)
test_accuracies, test_losses = np.zeros(nEpoch), np.zeros(nEpoch)
W_best, acc_best = None, 0.0
for epoch in tqdm(range(nEpoch)):
    # Calculate loss & accuracy
   # train_losses[epoch] = cross_entropy(X_train, Y_train, W)
    train_accuracies[epoch] = accuracy(X_train, Y_train, W)

   # test_losses[epoch] = cross_entropy(X_test, Y_test, W)
    test_accuracies[epoch] = accuracy(X_test, Y_test, W)
    if test_accuracies[epoch] > acc_best:
        acc_best = test_accuracies[epoch]
        W_best = deepcopy(W)

    # Calculating gradients
    W_grad = cross_entropy_grad(X_train, Y_train, W)

    # Updating W
    W -= trainRate * W_grad

fig, ax = plt.subplots(1, 1, figsize=(14, 7))

x_plot = [i + 1 for i in range(nEpoch)]

#ax.plot(x_plot, train_losses, label="Training Loss")
ax.plot(x_plot, train_accuracies, label="Training Accuracy")

#ax.plot(x_plot, test_losses, label="Test Loss")
ax.plot(x_plot, test_accuracies, label="Test Accuracy")

print("Best test accuracy: ", acc_best)

ax.set_xlabel("Epoch nr.")
ax.legend()

plt.show()

def number_check(W_learned: np.ndarray,
                 X_test: np.ndarray,
                 Y_test: np.ndarray,
                 idx: int):
    print("----------------")
    print("Model: ", np.argmax(softmax(X_test[idx].T @ W_learned)))
    print("Actual: ", np.argmax(Y_test[idx]))
    print("----------------")

for i in range(10):
    number_check(W_best,X_test,Y_test,i)