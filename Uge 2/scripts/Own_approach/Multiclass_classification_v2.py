import numpy as np
import torch as tc
import torchvision as tv
from my_helper_functions_v2 import *
from tqdm import tqdm
import matplotlib.pyplot as plt
from copy import deepcopy


def softmax_stable(x):
    _z_max = tc.amax(x, dim=1, keepdim=True)
    _log_sum = tc.log(tc.sum(tc.exp(x - _z_max), dim=1, keepdim=True))
    _result = tc.exp(x - _z_max - _log_sum)
    assert x.shape == _result.shape
    return _result


_TRAINING_FRACTION, _DATA_FRACTION = 0.5, 0.1
_NR_DATA_POINTS, _NR_FEATURES = 5, 28 * 28
_NR_EPOCHS, _BATCH_SIZE = 100, 64
_LEARNING_RATE, _SEED = 0.05, 123
_NR_CLASSES = 10

# Loading data
(X_train, y_train), (X_test, y_test) = data_load(_TRAINING_FRACTION, _DATA_FRACTION, _SEED, False, True)
print('X_train: ' + str(X_train.shape), type(X_train))
print('y_train: ' + str(y_train.shape), type(y_train))
print('X_test: ' + str(X_test.shape), type(X_test))
print('y_test: ' + str(y_test.shape), type(y_test))
Y_train, Y_test = one_hot_encoding(y_train, _NR_CLASSES), one_hot_encoding(y_test, _NR_CLASSES)

# Transforming to torch objects
_W = np.random.normal(0, 1, size=(_NR_FEATURES, _NR_CLASSES))
W = tc.tensor(data=list(_W), dtype=tc.float64, requires_grad=True)
X_train = tc.tensor(data=list(X_train), dtype=tc.float64, requires_grad=True)
X_test = tc.tensor(data=list(X_test), dtype=tc.float64, requires_grad=True)
Y_train = tc.tensor(data=list(Y_train), dtype=tc.float64, requires_grad=True)
Y_test = tc.tensor(data=list(Y_test), dtype=tc.float64, requires_grad=True)

nEpoch = 2000
trainRate = 0.9  # pick a number less than 1
train_accuracies, train_losses = np.empty(nEpoch), np.empty(nEpoch)
test_accuracies, test_losses = np.empty(nEpoch), np.empty(nEpoch)
W_best, acc_best = None, 0.0
_y_real = tc.argmax(Y_train, dim=1)      # only need to do once
_y_real_test = tc.argmax(Y_test, dim=1)  # only need to do once
softmax = tc.nn.Softmax(dim=1)
for epoch in tqdm(range(nEpoch)):
    # Calculate loss
    _lin_term = tc.matmul(X_train, W)
    _softmax = softmax_stable(_lin_term)
    _loss = tc.mean(-tc.log(tc.sum(tc.mul(Y_train, _softmax), dim=1)))
    # Calculating gradients
    _loss.backward()
    with tc.no_grad():
        W -= trainRate * W.grad

    # Calculating accuracy
    _y_estimator = tc.argmax(_softmax, dim=1)
    _accuracy = tc.sum(_y_estimator == _y_real) / len(y_train)
    train_accuracies[epoch] = _accuracy

    _y_test_estimator = tc.argmax(softmax_stable(tc.matmul(X_test, W)), dim=1)
    _accuracy = tc.sum(_y_test_estimator == _y_real_test) / len(y_test)
    test_accuracies[epoch] = _accuracy
    if test_accuracies[epoch] > acc_best:
        acc_best = train_accuracies[epoch]
        W_best = W

# print(train_accuracies)
fig, ax = plt.subplots(1, 1, figsize=(14, 7))

x_plot = [i + 1 for i in range(nEpoch)]

ax.plot(x_plot, train_accuracies, label="Training Accuracy")
ax.plot(x_plot, test_accuracies, label="Test Accuracy")

print("Best test accuracy: ", acc_best)

ax.set_xlabel("Epoch nr.")
ax.legend()

plt.show()
"""
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
"""
