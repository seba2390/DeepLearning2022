from scripts.Own_approach.my_helper_functions_v2 import *
from tqdm import tqdm
import matplotlib.pyplot as plt

class LogRegression(tc.nn.Module):
    def __init__(self, nr_features):
        super().__init__()
        self.input_layer = tc.nn.Linear(in_features=nr_features, out_features=1, bias=True)

    def forward(self, x):
        _x = self.input_layer(x)
        _x = tc.sigmoid(_x)
        return tc.squeeze(input=_x)


_TRAINING_FRACTION, _DATA_FRACTION = 0.8, 0.5
_NR_DATA_POINTS, _NR_FEATURES = 20, 28 * 28
_NR_EPOCHS, _BATCH_SIZE = 100, 64
_LEARNING_RATE, _SEED = 0.05, 123

(X_train, y_train), (X_test, y_test) = data_load(_TRAINING_FRACTION, _DATA_FRACTION, _SEED, True)
print('X_train: ' + str(X_train.shape), type(X_train))
print('y_train: ' + str(y_train.shape), type(y_train))
print('X_test: ' + str(X_test.shape), type(X_test))
print('y_test: ' + str(y_test.shape), type(y_test))
training_set = tc.utils.data.TensorDataset(tc.tensor(X_train).float(), tc.tensor(y_train).float())
test_set = tc.utils.data.TensorDataset(tc.tensor(X_test).float(), tc.tensor(y_test).float())
training_data_loader = tc.utils.data.DataLoader(training_set, batch_size=_BATCH_SIZE, shuffle=True)
model = LogRegression(nr_features=_NR_FEATURES)
optimizer = tc.optim.SGD(model.parameters(), lr=_LEARNING_RATE)
loss_fn = tc.nn.BCELoss()

accuracies = np.empty(shape=(_NR_EPOCHS,), dtype=float)
losses = np.empty(shape=(_NR_EPOCHS,), dtype=float)
for epoch in tqdm(range(_NR_EPOCHS)):
    _y_estimates, _y_actual, _losses = np.empty((0,)), np.empty((0,)), np.empty((0,))
    for _x_batch, _y_batch in training_data_loader:
        _y_estimate = model.forward(_x_batch)
        forward_pass = model.forward(_x_batch)
        loss = loss_fn(forward_pass, _y_batch)
        model.zero_grad()
        loss.backward()
        optimizer.step()
        _y_estimate = _y_estimate > 0.5
        _y_estimates = np.append(_y_estimates, _y_estimate)
        _y_actual = np.append(_y_actual, _y_batch)
        _losses = np.append(_losses, loss.detach().numpy())
    accuracies[epoch] = np.mean(_y_estimates == _y_actual)
    losses[epoch]     = np.mean(_losses)

fig, ax = plt.subplots(1, 1, figsize=(14, 7))

x_plot = [i + 1 for i in range(_NR_EPOCHS)]

ax.plot(x_plot, accuracies, label="Training accuracy")
ax.plot(x_plot, losses, label="Training Loss")
ax.legend()
plt.show()
