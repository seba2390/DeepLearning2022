import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from myDataSet import *
from tqdm import tqdm


class NeuralNet(torch.nn.Module):
    """ Inspired by: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html"""

    def __init__(self, channels_in: int,
                 nr_classes: int,
                 input_dimensions: tuple[int, int]) -> None:
        super(NeuralNet, self).__init__()
        self.input_dim = input_dimensions
        self.nr_classes = nr_classes
        self.device = None

        # layer 1: convolutional layer
        self.conv1 = torch.nn.Conv2d(in_channels=channels_in,
                                     out_channels=6,
                                     kernel_size=(5, 5),
                                     stride=(1, 1),
                                     padding=(0, 0))
        # activation function
        self.activation1 = torch.nn.ReLU()
        object_dim1 = self.output_dim(self.input_dim,
                                      self.conv1.kernel_size,
                                      self.conv1.stride,
                                      self.conv1.padding)
        # --------------------------------------#

        # layer 2: pooling layer
        self.pool1 = torch.nn.AvgPool2d(kernel_size=(2, 2),
                                        stride=(1, 1),
                                        padding=(0, 0))

        object_dim2 = self.output_dim(object_dim1,
                                      self.pool1.kernel_size,
                                      self.pool1.stride,
                                      self.pool1.padding)

        # --------------------------------------#

        # layer 3: convolutional layer
        self.conv2 = torch.nn.Conv2d(in_channels=6,
                                     out_channels=16,
                                     kernel_size=5,
                                     stride=(1, 1),
                                     padding=(0, 0))
        # activation function
        self.activation2 = torch.nn.ReLU()
        object_dim3 = self.output_dim(object_dim2,
                                      self.conv2.kernel_size,
                                      self.conv2.stride,
                                      self.conv2.padding)
        # --------------------------------------#

        # layer 4: pooling layer
        self.pool2 = torch.nn.AvgPool2d(kernel_size=(2, 2),
                                        stride=(1, 1),
                                        padding=(0, 0))

        object_dim4 = self.output_dim(object_dim3,
                                      self.pool2.kernel_size,
                                      self.pool2.stride,
                                      self.pool2.padding)
        # flattening
        self.flatten = torch.nn.Flatten(start_dim=1, end_dim=-1)

        # --------------------------------------#

        # layer 5: fully connected layer (lazy means input dim is automatically inferred)
        in_vector_length = self.conv2.out_channels * object_dim4[0] * object_dim4[1]
        self.lin1 = torch.nn.Linear(in_features=in_vector_length,
                                    out_features=120)

        # activation function
        self.activation3 = torch.nn.ReLU()

        # --------------------------------------#

        # layer 6: fully connected layer (lazy means input dim is automatically inferred)
        self.lin2 = torch.nn.Linear(in_features=self.lin1.out_features,
                                    out_features=self.nr_classes)

    @staticmethod
    def output_dim(data_size: tuple[int, int],
                   kernel_size: tuple[int, ...],
                   stride_size: tuple[int, ...] = (1, 1),
                   padding_size: tuple[int, ...] = (0, 0)) -> tuple[int, int]:
        """
        Calculates output shape of array after convolution w. specific
        kernel, padding and stride.

         Parameters:
        - data_size: tuple containing dimension of 2D input array.
        - kernel_size: tuple containing dimension of 2D kernel.
        - stride_size: tuple containing dimension of 2D stride.
        - padding_size: tuple containing dimension of 2D padding.

         Returns:
        - output_dimensions: tuple containing dimension of resulting 2D array.
        """

        out_height = ((data_size[0] - kernel_size[0] + 2 * padding_size[0]) // stride_size[0]) + 1
        out_width = ((data_size[1] - kernel_size[1] + 2 * padding_size[1]) // stride_size[1]) + 1

        output_dimensions = (out_height, out_width)
        return output_dimensions

    def predict(self, forwarded_data: torch.Tensor) -> torch.Tensor:
        """
        Takes the raw estimate from a forward pass (nr. data points, nr. classes)
        and sets the highest val in each row 1 and the rest zero.

         Parameters:
        - forwarded_data: torch.Tensor of shape (nr. data points, nr. classes)

         Returns:
        - _prediction: torc.Tensor of shape (nr. data points, nr. classes)

        """
        assert forwarded_data.shape[
                   1] == self.nr_classes, f'Forwarded data should be of shape (nr. data points, nr. classes).'
        assert len(forwarded_data.shape) == 2, f' Forwarded data should be a 2D tensor.'
        assert forwarded_data.device.type == self.device.type, f'forwarded_data is on device: {forwarded_data.device.type}, but should be on: {self.device.type}'
        for _row in forwarded_data:
            _row[_row == torch.max(_row)] = 1
            _row[_row != 1] = 0
            _row.to(self.device)
        assert forwarded_data.device.type == self.device.type, f'_prediction is on device: {forwarded_data.device.type}, but should be on: {self.device.type}'
        assert torch.all((forwarded_data == 0) | (
                    forwarded_data == 1)), f'forwarded_data should be one-hot encoded (only 1 and 0), but is: {forwarded_data}.'
        return forwarded_data

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Function performing the forward pass on 'data', i.e. sequentially
        applying each layer in the network.

         Parameters:
        - data: 4D torch tensor of shape (nr. data points, nr. channels, height, width).

         Returns:
        - data: 4D torch tensor of shape (nr. data points, nr. channels, height, width).

        """

        assert len(data.shape) == 4, f'Shape of X should be (nr. data points, nr. channels, height, width)'
        data = self.conv1(data)
        data = self.activation1(data)
        data = self.pool1(data)
        data = self.conv2(data)
        data = self.activation2(data)
        data = self.pool2(data)
        data = self.flatten(data)
        data = self.lin1(data)
        data = self.activation3(data)
        data = self.lin2(data)
        return data

    def accuracy(self, y_hat: torch.Tensor,
                 y_actual: torch.Tensor) -> float:
        """
        Function for calculating the accuracy of 'y_hat' against 'y_actual'
        in terms of: nr. agreements / nr. comparisons.

         Parameters:
        - y_hat: 2D torch tensor of shape (nr. datapoints, nr. classes) only containing 1 and 0.
        - y_actual: 2D torch tensor of shape (nr. datapoints, nr. classes) only containing 1 and 0.

         Returns:
        - _accuracy: float in range (0,1).
        """
        assert y_hat.shape == y_actual.shape, f'Inputs are not of matching shapes.'
        assert y_hat.device.type == self.device.type, f'y_hat is on device: {y_hat.device.type}, but should be on: {self.device.type}'
        assert y_actual.device.type == self.device.type, f'y_actual is on device: {y_actual.device.type}, but should be on: {self.device.type}'
        assert torch.all((y_hat == 0) | (y_hat == 1)), f'y_hat should be one-hot encoded (only 1 and 0).'
        assert torch.all((y_actual == 0) | (y_actual == 1)), f'y_actual should be one-hot encoded (only 1 and 0).'

        _nr_equals = torch.sum(y_hat * y_actual)
        _accuracy = _nr_equals / y_hat.shape[0]
        return _accuracy

    def train_network(self, train_data_batches: list[torch.Tensor],
                      train_labels_batches: list[torch.Tensor],
                      test_data_batches: list[torch.Tensor],
                      test_labels_batches: list[torch.Tensor],
                      epochs: int = 10,
                      device_name: str = "cuda") -> tuple[list[float], list[float], list[float], list[float]]:

        """
        Function for training the network, e.i. learn the optimal weights for each
        layer. Training is done using stochastic gradient descent, with Negative
        Log-Likelihood (also known as Cross Entropy) as a loss function.

         Parameters:
        - train_data_batches: list of 4D torch tensors, each of shape (nr. data points, nr. channels, height, width)
        - train_labels_batches: list of 4D torch tensors, each of shape (nr. data points, nr. channels, height, width)
        - test_data_batches: list of 4D torch tensors, each of shape (nr. data points, nr. channels, height, width)
        - test_labels_batches: list of 4D torch tensors, each of shape (nr. data points, nr. channels, height, width)
        - epochs: nr. of epochs (iteration) used to train.

         Returns:
        - _train_accuracies: list of floats w. avg. accuracies of training data at given epoch.
        - _train_losses: list of floats w. avg. loss of training data at given epoch.
        - _test_accuracies: list of floats w. avg. accuracies of test data at given epoch.
        - _test_losses: list of floats w. avg. loss of test data at given epoch.

        """
        self.device = torch.device('cpu')
        if device_name != 'cpu':
            if torch.cuda.is_available():
                self.device = torch.device(device_name)
                # Allocating model weights on GPU
                self.to(self.device)
                self.train()
                print("Current torch cuda version: ", torch.version.cuda)
                print("Cuda devices found by torch: ", torch.cuda.device_count())
                print("Current cuda device to be used by torch: ", torch.cuda.current_device())
                print("Name of cuda device: ", torch.cuda.get_device_name(0))
                t = torch.cuda.get_device_properties(0).total_memory
                print("Total memory in cuda device: ", t / 1e6, "MB")
                r = torch.cuda.memory_reserved(0)
                print("Total memory reserved in cuda device: ", r / 1e6, "MB")
                a = torch.cuda.memory_allocated(0)
                print("Total memory allocated in cuda device: ", a / 1e6, "MB")
                print("Total remaining memory in cuda device:: ", (t - r) / 1e6, "MB")
                # Allocating input on GPU
                for _batch in range(len(train_data_batches)):
                    train_data_batches[_batch] = train_data_batches[_batch].to(self.device)
                    train_labels_batches[_batch] = train_labels_batches[_batch].to(self.device)
                for _batch in range(len(test_data_batches)):
                    test_data_batches[_batch] = test_data_batches[_batch].to(self.device)
                    test_labels_batches[_batch] = test_labels_batches[_batch].to(self.device)
            else:
                self.to(self.device)
                self.train()
                print("No cuda device available to torch - defaulting to cpu.")

        _optimizer = torch.optim.Adam(params=self.parameters(),
                                      lr=0.0001,
                                      weight_decay=0.0001)
        _loss_function = torch.nn.CrossEntropyLoss()

        _train_accuracies, _train_losses = [], []
        _test_accuracies, _test_losses = [], []

        _best_test_accuracy = 0.0
        for epoch in tqdm(range(epochs)):
            _counter = 0
            _batch_train_acc, _batch_train_loss = 0.0, 0.0
            _batch_test_acc, _batch_test_loss = 0.0, 0.0
            # Training against training data
            for x_batch, y_batch in zip(train_data_batches, train_labels_batches):
                assert x_batch.device.type == self.device.type, f'x_batch is on device: {x_batch.device.type}, but should be on: {self.device.type}'
                assert y_batch.device.type == self.device.type, f'y_batch is on device: {y_batch.device.type}, but should be on: {self.device.type}'
                # Ensure that parameter gradients are 0.
                _optimizer.zero_grad()
                # Forward pass
                _y_hat = self.forward(x_batch)
                assert _y_hat.shape[0] == train_labels_batches[0].shape[0]
                assert _y_hat.shape[1] == self.nr_classes
                # Loss calculation + backprop + optimization
                _loss = _loss_function(_y_hat, y_batch)
                _loss.backward()
                _optimizer.step()
                # Saving losses and accuracies
                with torch.no_grad():
                    if self.device.type != 'cpu':
                        _batch_train_acc += self.accuracy(self.predict(_y_hat), y_batch).item()
                    else:
                        _batch_train_acc += self.accuracy(self.predict(_y_hat), y_batch)
                    _batch_train_loss += _loss.item()
                    _counter += 1
            _train_accuracies.append(_batch_train_acc / _counter)
            _train_losses.append(_batch_train_loss / _counter)
            # Testing model on test data
            for x_batch, y_batch in zip(test_data_batches, test_labels_batches):
                assert x_batch.device.type == self.device.type, f'x_batch is on device: {x_batch.device.type}, but should be on: {self.device.type}'
                assert y_batch.device.type == self.device.type, f'y_batch is on device: {y_batch.device.type}, but should be on: {self.device.type}'
                # Forward pass
                _y_hat = self.forward(x_batch)
                assert _y_hat.shape[0] == train_labels_batches[0].shape[0]
                assert _y_hat.shape[1] == self.nr_classes
                # Loss calculation
                _loss = _loss_function(_y_hat, y_batch)
                # Saving losses and accuracies
                with torch.no_grad():
                    if self.device.type != 'cpu':
                        _batch_test_acc += self.accuracy(self.predict(_y_hat), y_batch).item()
                    else:
                        _batch_test_acc += self.accuracy(self.predict(_y_hat), y_batch)
                    _batch_test_loss += _loss.item()

            # Saving instance of model that yields the highest test acc during training
            if _batch_test_acc / _counter > _best_test_accuracy:
                _best_test_accuracy = _batch_test_acc / _counter
                torch.save(obj=self.state_dict(), f="Model/model.pt")
                print("Model saved at epoch: ", epoch + 1, ", with test acc: ", _best_test_accuracy.item() * 100.0, "%")

            # Saving current loss and current accuracy
            _test_accuracies.append(_batch_test_acc / _counter)
            _test_losses.append(_batch_test_loss / _counter)

        # Removing from GPU and Allocating input on CPU
        if self.device.type == "cuda":
            self.device = torch.device('cpu')
            self.to(self.device)
            for _batch in range(len(train_data_batches)):
                train_data_batches[_batch] = train_data_batches[_batch].to(self.device)
                train_labels_batches[_batch] = train_labels_batches[_batch].to(self.device)
            for _batch in range(len(test_data_batches)):
                test_data_batches[_batch] = test_data_batches[_batch].to(self.device)
                test_labels_batches[_batch] = test_labels_batches[_batch].to(self.device)
        return _train_accuracies, _train_losses, _test_accuracies, _test_losses


class NeuralNet2(torch.nn.Module):
    """ Inspired by https://learn.microsoft.com/en-us/windows/ai/windows-ml/tutorials/pytorch-train-model """

    def __init__(self, channels_in: int,
                 nr_classes: int,
                 input_dimensions: tuple[int, int]) -> None:
        super(NeuralNet2, self).__init__()
        self.input_dim = input_dimensions
        self.nr_classes = nr_classes
        self.device = None

        # layer 1: convolutional layer
        self.conv1 = torch.nn.Conv2d(in_channels=channels_in,
                                     out_channels=12,
                                     kernel_size=(5, 5),
                                     stride=(1, 1),
                                     padding=(1, 1))
        # normalization function
        self.norm1 = torch.nn.BatchNorm2d(self.conv1.out_channels)
        # activation function
        self.activation1 = torch.nn.ReLU()
        object_dim1 = self.output_dim(self.input_dim,
                                      self.conv1.kernel_size,
                                      self.conv1.stride,
                                      self.conv1.padding)

        # --------------------------------------#

        # layer 2: convolutional layer
        self.conv2 = torch.nn.Conv2d(in_channels=self.conv1.out_channels,
                                     out_channels=12,
                                     kernel_size=(5, 5),
                                     stride=(1, 1),
                                     padding=(1, 1))
        # normalization function
        self.norm2 = torch.nn.BatchNorm2d(self.conv2.out_channels)
        # activation function
        self.activation2 = torch.nn.ReLU()
        object_dim2 = self.output_dim(object_dim1,
                                      self.conv2.kernel_size,
                                      self.conv2.stride,
                                      self.conv2.padding)

        # --------------------------------------#

        # layer 3: pooling layer
        self.pool1 = torch.nn.MaxPool2d(kernel_size=(2, 2),
                                        stride=(2, 2),
                                        padding=(0, 0))

        object_dim3 = self.output_dim(object_dim2,
                                      self.pool1.kernel_size,
                                      self.pool1.stride,
                                      self.pool1.padding)

        # --------------------------------------#

        # layer 4: convolutional layer
        self.conv3 = torch.nn.Conv2d(in_channels=self.conv2.out_channels,
                                     out_channels=24,
                                     kernel_size=(5, 5),
                                     stride=(1, 1),
                                     padding=(1, 1))
        # normalization function
        self.norm3 = torch.nn.BatchNorm2d(self.conv3.out_channels)
        # activation function
        self.activation3 = torch.nn.ReLU()
        object_dim4 = self.output_dim(object_dim3,
                                      self.conv3.kernel_size,
                                      self.conv3.stride,
                                      self.conv3.padding)

        # --------------------------------------#

        # layer 5: convolutional layer
        self.conv4 = torch.nn.Conv2d(in_channels=self.conv3.out_channels,
                                     out_channels=24,
                                     kernel_size=(5, 5),
                                     stride=(1, 1),
                                     padding=(1, 1))
        # normalization function
        self.norm4 = torch.nn.BatchNorm2d(self.conv4.out_channels)

        # activation function
        self.activation4 = torch.nn.ReLU()
        object_dim5 = self.output_dim(object_dim4,
                                      self.conv4.kernel_size,
                                      self.conv4.stride,
                                      self.conv4.padding)

        # --------------------------------------#

        # flattening
        self.flatten = torch.nn.Flatten(start_dim=1, end_dim=-1)

        # --------------------------------------#
        # layer 6: fully connected layer (lazy means input dim is automatically inferred)
        in_vector_length = self.conv4.out_channels * object_dim5[0] * object_dim5[1]
        self.lin1 = torch.nn.Linear(in_features=in_vector_length,
                                    out_features=self.nr_classes)

    @staticmethod
    def output_dim(data_size: tuple[int, int],
                   kernel_size: tuple[int, ...],
                   stride_size: tuple[int, ...] = (1, 1),
                   padding_size: tuple[int, ...] = (0, 0)) -> tuple[int, int]:
        """
        Calculates output shape of array after convolution w. specific
        kernel, padding and stride.

         Parameters:
        - data_size: tuple containing dimension of 2D input array.
        - kernel_size: tuple containing dimension of 2D kernel.
        - stride_size: tuple containing dimension of 2D stride.
        - padding_size: tuple containing dimension of 2D padding.

         Returns:
        - output_dimensions: tuple containing dimension of resulting 2D array.
        """

        out_height = ((data_size[0] - kernel_size[0] + 2 * padding_size[0]) // stride_size[0]) + 1
        out_width = ((data_size[1] - kernel_size[1] + 2 * padding_size[1]) // stride_size[1]) + 1

        output_dimensions = (out_height, out_width)
        return output_dimensions

    def predict(self, forwarded_data: torch.Tensor) -> torch.Tensor:
        """
        Takes the raw estimate from a forward pass (nr. data points, nr. classes)
        and sets the highest val in each row 1 and the rest zero.

         Parameters:
        - forwarded_data: torch.Tensor of shape (nr. data points, nr. classes)

         Returns:
        - _prediction: torc.Tensor of shape (nr. data points, nr. classes)

        """
        assert forwarded_data.shape[
                   1] == self.nr_classes, f'Forwarded data should be of shape (nr. data points, nr. classes).'
        assert len(forwarded_data.shape) == 2, f' Forwarded data should be a 2D tensor.'
        assert forwarded_data.device.type == self.device.type, f'forwarded_data is on device: {forwarded_data.device.type}, but should be on: {self.device.type}'
        for _row in forwarded_data:
            _row[_row == torch.max(_row)] = 1
            _row[_row != 1] = 0
            _row.to(self.device)
        assert forwarded_data.device.type == self.device.type, f'_prediction is on device: {forwarded_data.device.type}, but should be on: {self.device.type}'
        assert torch.all((forwarded_data == 0) | (
                    forwarded_data == 1)), f'forwarded_data should be one-hot encoded (only 1 and 0), but is: {forwarded_data}.'
        return forwarded_data

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Function performing the forward pass on 'data', i.e. sequentially
        applying each layer in the network.

         Parameters:
        - data: 4D torch tensor of shape (nr. data points, nr. channels, height, width).

         Returns:
        - data: 4D torch tensor of shape (nr. data points, nr. channels, height, width).

        """

        assert len(data.shape) == 4, f'Shape of X should be (nr. data points, nr. channels, height, width)'

        # Layer 1
        data = self.conv1(data)
        data = self.norm1(data)
        data = self.activation1(data)

        # Layer 2
        data = self.conv2(data)
        data = self.norm2(data)
        data = self.activation2(data)

        # Pool layer
        data = self.pool1(data)

        # Layer 3
        data = self.conv3(data)
        data = self.norm3(data)
        data = self.activation3(data)

        # Layer 4
        data = self.conv4(data)
        data = self.norm4(data)
        data = self.activation4(data)

        # Layer 5
        data = self.flatten(data)
        data = self.lin1(data)

        return data

    def accuracy(self, y_hat: torch.Tensor,
                 y_actual: torch.Tensor) -> float:
        """
        Function for calculating the accuracy of 'y_hat' against 'y_actual'
        in terms of: nr. agreements / nr. comparisons.

         Parameters:
        - y_hat: 2D torch tensor of shape (nr. datapoints, nr. classes) only containing 1 and 0.
        - y_actual: 2D torch tensor of shape (nr. datapoints, nr. classes) only containing 1 and 0.

         Returns:
        - _accuracy: float in range (0,1).
        """
        assert y_hat.shape == y_actual.shape, f'Inputs are not of matching shapes.'
        assert y_hat.device.type == self.device.type, f'y_hat is on device: {y_hat.device.type}, but should be on: {self.device.type}'
        assert y_actual.device.type == self.device.type, f'y_actual is on device: {y_actual.device.type}, but should be on: {self.device.type}'
        assert torch.all(
            (y_hat == 0) | (y_hat == 1)), f'y_hat should be one-hot encoded (only 1 and 0), but is: {y_hat}'
        assert torch.all((y_actual == 0) | (y_actual == 1)), f'y_actual should be one-hot encoded (only 1 and 0).'

        _nr_equals = torch.sum(y_hat * y_actual)
        _accuracy = _nr_equals / y_hat.shape[0]
        return _accuracy

    def train_network(self, train_data_batches: list[torch.Tensor],
                      train_labels_batches: list[torch.Tensor],
                      test_data_batches: list[torch.Tensor],
                      test_labels_batches: list[torch.Tensor],
                      epochs: int = 10,
                      device_name: str = "cuda") -> tuple[list[float], list[float], list[float], list[float]]:

        """
        Function for training the network, e.i. learn the optimal weights for each
        layer. Training is done using stochastic gradient descent, with Negative
        Log-Likelihood (also known as Cross Entropy) as a loss function.

         Parameters:
        - train_data_batches: list of 4D torch tensors, each of shape (nr. data points, nr. channels, height, width)
        - train_labels_batches: list of 4D torch tensors, each of shape (nr. data points, nr. channels, height, width)
        - test_data_batches: list of 4D torch tensors, each of shape (nr. data points, nr. channels, height, width)
        - test_labels_batches: list of 4D torch tensors, each of shape (nr. data points, nr. channels, height, width)
        - epochs: nr. of epochs (iteration) used to train.

         Returns:
        - _train_accuracies: list of floats w. avg. accuracies of training data at given epoch.
        - _train_losses: list of floats w. avg. loss of training data at given epoch.
        - _test_accuracies: list of floats w. avg. accuracies of test data at given epoch.
        - _test_losses: list of floats w. avg. loss of test data at given epoch.

        """
        self.device = torch.device('cpu')
        if device_name != 'cpu':
            if torch.cuda.is_available():
                self.device = torch.device(device_name)
                # Allocating model weights on GPU
                self.to(self.device)
                self.train()
                print("Current torch cuda version: ", torch.version.cuda)
                print("Cuda devices found by torch: ", torch.cuda.device_count())
                print("Current cuda device to be used by torch: ", torch.cuda.current_device())
                print("Name of cuda device: ", torch.cuda.get_device_name(0))
                t = torch.cuda.get_device_properties(0).total_memory
                print("Total memory in cuda device: ", t / 1e6, "MB")
                r = torch.cuda.memory_reserved(0)
                print("Total memory reserved in cuda device: ", r / 1e6, "MB")
                a = torch.cuda.memory_allocated(0)
                print("Total memory allocated in cuda device: ", a / 1e6, "MB")
                print("Total remaining memory in cuda device:: ", (t - r) / 1e6, "MB")
                # Allocating input on GPU
                for _batch in range(len(train_data_batches)):
                    train_data_batches[_batch] = train_data_batches[_batch].to(self.device)
                    train_labels_batches[_batch] = train_labels_batches[_batch].to(self.device)
                for _batch in range(len(test_data_batches)):
                    test_data_batches[_batch] = test_data_batches[_batch].to(self.device)
                    test_labels_batches[_batch] = test_labels_batches[_batch].to(self.device)
            else:
                self.to(self.device)
                self.train()
                print("No cuda device available to torch - defaulting to cpu.")

        _optimizer = torch.optim.Adam(params=self.parameters(),
                                      lr=0.0001,
                                      weight_decay=0.0001)
        _loss_function = torch.nn.CrossEntropyLoss()

        _train_accuracies, _train_losses = [], []
        _test_accuracies, _test_losses = [], []

        _best_test_accuracy = 0.0
        for epoch in tqdm(range(epochs)):
            _counter = 0
            _batch_train_acc, _batch_train_loss = 0.0, 0.0
            _batch_test_acc, _batch_test_loss = 0.0, 0.0
            # Training against training data
            for x_batch, y_batch in zip(train_data_batches, train_labels_batches):
                assert x_batch.device.type == self.device.type, f'x_batch is on device: {x_batch.device.type}, but should be on: {self.device.type}'
                assert y_batch.device.type == self.device.type, f'y_batch is on device: {y_batch.device.type}, but should be on: {self.device.type}'
                # Ensure that parameter gradients are 0.
                _optimizer.zero_grad()
                # Forward pass
                _y_hat = self.forward(x_batch)
                assert _y_hat.shape[0] == train_labels_batches[0].shape[0]
                assert _y_hat.shape[1] == self.nr_classes
                # Loss calculation + backprop + optimization
                _loss = _loss_function(_y_hat, y_batch)
                _loss.backward()
                _optimizer.step()
                # Saving losses and accuracies
                with torch.no_grad():
                    if self.device.type != 'cpu':
                        _batch_train_acc += self.accuracy(self.predict(_y_hat), y_batch).item()
                    else:
                        _batch_train_acc += self.accuracy(self.predict(_y_hat), y_batch)
                    _batch_train_loss += _loss.item()
                    _counter += 1
            _train_accuracies.append(_batch_train_acc / _counter)
            _train_losses.append(_batch_train_loss / _counter)
            # Testing model on test data
            for x_batch, y_batch in zip(test_data_batches, test_labels_batches):
                assert x_batch.device.type == self.device.type, f'x_batch is on device: {x_batch.device.type}, but should be on: {self.device.type}'
                assert y_batch.device.type == self.device.type, f'y_batch is on device: {y_batch.device.type}, but should be on: {self.device.type}'
                # Forward pass
                _y_hat = self.forward(x_batch)
                assert _y_hat.shape[0] == train_labels_batches[0].shape[0]
                assert _y_hat.shape[1] == self.nr_classes
                # Loss calculation
                _loss = _loss_function(_y_hat, y_batch)
                # Saving losses and accuracies
                with torch.no_grad():
                    if self.device.type != 'cpu':
                        _batch_test_acc += self.accuracy(self.predict(_y_hat), y_batch).item()
                    else:
                        _batch_test_acc += self.accuracy(self.predict(_y_hat), y_batch)
                    _batch_test_loss += _loss.item()

            # Saving instance of model that yields the highest test acc during training
            if _batch_test_acc / _counter > _best_test_accuracy:
                _best_test_accuracy = _batch_test_acc / _counter
                torch.save(obj=self.state_dict(), f="Model/model.pt")
                print("Model saved at epoch: ", epoch + 1, ", with test acc: ", _best_test_accuracy.item() * 100.0, "%")

            # Saving current loss and current accuracy
            _test_accuracies.append(_batch_test_acc / _counter)
            _test_losses.append(_batch_test_loss / _counter)

        # Removing from GPU and Allocating input on CPU
        if self.device.type == "cuda":
            self.device = torch.device('cpu')
            self.to(self.device)
            for _batch in range(len(train_data_batches)):
                train_data_batches[_batch] = train_data_batches[_batch].to(self.device)
                train_labels_batches[_batch] = train_labels_batches[_batch].to(self.device)
            for _batch in range(len(test_data_batches)):
                test_data_batches[_batch] = test_data_batches[_batch].to(self.device)
                test_labels_batches[_batch] = test_labels_batches[_batch].to(self.device)
        return _train_accuracies, _train_losses, _test_accuracies, _test_losses

