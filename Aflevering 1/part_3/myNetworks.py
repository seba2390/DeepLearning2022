import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from myDataSet import *
from tqdm import tqdm


class NeuralNet(torch.nn.Module):
    # see https://www.deeplearning.ai/ai-notes/initialization/index.html for weight initialization
    def __init__(self,
                 nr_classes: int,
                 nr_features: int,
                 seed: int = 0) -> None:
        super(NeuralNet, self).__init__()

        # Setting torch seed for reproducibility
        torch.manual_seed(0)

        self.nr_features = nr_features
        self.nr_classes = nr_classes
        self.device = None

        # layer 1: linear layer
        self.lin1 = torch.nn.Linear(in_features=self.nr_features,
                                    out_features=40,
                                    bias=True)
        self.activation1 = torch.nn.ReLU()

        # layer 2: linear layer
        self.lin2 = torch.nn.Linear(in_features=self.lin1.out_features,
                                    out_features=60,
                                    bias=True)
        self.activation2 = torch.nn.ReLU()

        # layer 3: linear layer
        self.lin3 = torch.nn.Linear(in_features=self.lin2.out_features,
                                    out_features=self.nr_classes,
                                    bias=True)

        # Recursively initializing weights of linear layers in specific manner
        self.apply(self.init_weights)

    @staticmethod
    def init_weights(module: torch.nn.modules,
                     mode: str = "kaiming_uniform") -> None:
        """
        Void type static function for initializing
        weights and biases of linear layers in network in
        chosen manner.

         Parameters:
        - module: torch.nn.modules
        - mode: string representation of chosen initialization mode.
        """
        if isinstance(module, torch.nn.Linear):

            if mode == "xavier_uniform":
                """ Understanding the difficulty of training deep 
                    feedforward neural networks - Glorot, X. & Bengio, Y. (2010) """
                torch.nn.init.xavier_uniform_(module.weight, gain=torch.nn.init.calculate_gain('relu'))
                if module.bias is not None:
                    torch.nn.init.xavier_uniform_(module.bias, gain=torch.nn.init.calculate_gain('relu'))
            if mode == "xavier_normal":
                """ Understanding the difficulty of training deep 
                    feedforward neural networks - Glorot, X. & Bengio, Y. (2010) """
                torch.nn.init.xavier_normal_(module.weight, gain=torch.nn.init.calculate_gain('relu'))
                if module.bias is not None:
                    torch.nn.init.xavier_normal_(module.bias, gain=torch.nn.init.calculate_gain('relu'))
            if mode == "kaiming_uniform":
                """ Delving deep into rectifiers: Surpassing
                    human-level performance on ImageNet classification - He, K. et al. (2015) """
                torch.nn.init.kaiming_uniform_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    torch.nn.init.kaiming_uniform_(module.weight, mode='fan_out', nonlinearity='relu')
            if mode == "kaiming_normal":
                """ Delving deep into rectifiers: Surpassing
                    human-level performance on ImageNet classification - He, K. et al. (2015) """
                torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')


    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Function performing the forward pass on 'data', i.e. sequentially
        applying each layer in the network.

         Parameters:
        - data: 4D torch tensor of shape (nr. data points, nr. channels, height, width).

         Returns:
        - data: 4D torch tensor of shape (nr. data points, nr. channels, height, width).

        """

        # Layer 1
        data = self.lin1(data)
        data = self.activation1(data)

        # Layer 2
        data = self.lin2(data)
        data = self.activation2(data)

        # layer 3
        data = self.lin3(data)

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

    def train_network(self, train_dataloader, validation_dataloader,
                      epochs: int = 10,
                      no_improvement_max: int = 10,
                      device_name: str = "cuda") -> tuple[list[float], list[float], list[float], list[float]]:

        """
        Function for training the network, e.i. learn the optimal weights for each
        layer. Training is done using stochastic gradient descent, with Negative
        Log-Likelihood (also known as Cross Entropy) as a loss function.

         Parameters:
        - train_data_batches: list of 4D torch tensors, each of shape (nr. data points, nr. channels, height, width)
        - train_labels_batches: list of 4D torch tensors, each of shape (nr. data points, nr. channels, height, width)
        - validation_data_batches: list of 4D torch tensors, each of shape (nr. data points, nr. channels, height, width)
        - validation_labels_batches: list of 4D torch tensors, each of shape (nr. data points, nr. channels, height, width)
        - epochs: nr. of epochs (iteration) used to train.

         Returns:
        - _train_accuracies: list of floats w. avg. accuracies of training data at given epoch.
        - _train_losses: list of floats w. avg. loss of training data at given epoch.
        - _validation_accuracies: list of floats w. avg. accuracies of validation data at given epoch.
        - _validation_losses: list of floats w. avg. loss of validation data at given epoch.

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
            else:
                self.to(self.device)
                self.train()
                print("No cuda device available to torch - defaulting to cpu.")

        # Allocating input on relevant device
        for _, (data, labels) in enumerate(train_dataloader):
            data = data.to(self.device)
            labels = labels.to(self.device)
        for _, (data, labels) in enumerate(validation_dataloader):
            data = data.to(self.device)
            labels = labels.to(self.device)

        _optimizer = torch.optim.Adam(params=self.parameters(),
                                      lr=0.001,
                                      weight_decay=0.0001)

        # _optimizer = torch.optim.Adagrad(params=self.parameters(),
        #                                 lr=0.01)

        # _optimizer = torch.optim.SGD(params=self.parameters(),
        #                             lr=0.01,
        #                             momentum=0.9)

        _loss_function = torch.nn.CrossEntropyLoss()

        _train_accuracies, _train_losses = [], []
        _validation_accuracies, _validation_losses = [], []
        _best_validation_accuracy, _best_epoch = 0.0, 0
        _no_improvement_counter = 0

        # Initial training loss and acc
        _counter = 0
        _batch_train_acc, _batch_train_loss = 0.0, 0.0
        for _, (x_batch, y_batch) in enumerate(train_dataloader):
            assert x_batch.device.type == self.device.type, f'x_batch is on device: {x_batch.device.type}, but should be on: {self.device.type}'
            assert y_batch.device.type == self.device.type, f'y_batch is on device: {y_batch.device.type}, but should be on: {self.device.type}'
            with torch.no_grad():
                # Forward pass
                _y_hat = self.forward(x_batch)
                # Loss calculation
                _loss = _loss_function(_y_hat, y_batch)
                # Saving losses and accuracies
                _batch_train_acc += self.accuracy(self.predict(_y_hat), y_batch)
                _batch_train_loss += _loss.item()
            _counter += 1
        _train_accuracies.append(_batch_train_acc / _counter)
        _train_losses.append(_batch_train_loss / _counter)

        # Initial validation loss and acc
        _counter = 0
        _batch_validation_acc, _batch_validation_loss = 0.0, 0.0
        for _, (x_batch, y_batch) in enumerate(validation_dataloader):
            assert x_batch.device.type == self.device.type, f'x_batch is on device: {x_batch.device.type}, but should be on: {self.device.type}'
            assert y_batch.device.type == self.device.type, f'y_batch is on device: {y_batch.device.type}, but should be on: {self.device.type}'
            with torch.no_grad():
                # Forward pass
                _y_hat = self.forward(x_batch)
                # Loss calculation
                _loss = _loss_function(_y_hat, y_batch)
                # Saving losses and accuracies
                _batch_validation_acc += self.accuracy(self.predict(_y_hat), y_batch)
                _batch_validation_loss += _loss.item()
            _counter += 1
        _validation_accuracies.append(_batch_validation_acc / _counter)
        _validation_losses.append(_batch_validation_loss / _counter)

        # Actual training
        for epoch in tqdm(range(epochs)):

            # Training against training data
            for _, (x_batch, y_batch) in enumerate(train_dataloader):
                assert x_batch.device.type == self.device.type, f'x_batch is on device: {x_batch.device.type}, but should be on: {self.device.type}'
                assert y_batch.device.type == self.device.type, f'y_batch is on device: {y_batch.device.type}, but should be on: {self.device.type}'
                # Ensure that parameter gradients are 0.
                _optimizer.zero_grad()
                # Forward pass
                _y_hat = self.forward(x_batch)
                # Loss calculation + backprop + optimization
                _loss = _loss_function(_y_hat, y_batch)
                _loss.backward()
                _optimizer.step()

            # Calculating and saving training data accuracy and loss
            _counter = 0
            _batch_train_acc, _batch_train_loss = 0.0, 0.0
            for _, (x_batch, y_batch) in enumerate(train_dataloader):
                assert x_batch.device.type == self.device.type, f'x_batch is on device: {x_batch.device.type}, but should be on: {self.device.type}'
                assert y_batch.device.type == self.device.type, f'y_batch is on device: {y_batch.device.type}, but should be on: {self.device.type}'
                with torch.no_grad():
                    # Forward pass
                    _y_hat = self.forward(x_batch)
                    # Loss calculation
                    _loss = _loss_function(_y_hat, y_batch)
                    # Saving losses and accuracies
                    _batch_train_acc += self.accuracy(self.predict(_y_hat), y_batch)
                    _batch_train_loss += _loss.item()
                _counter += 1
            _train_accuracies.append(_batch_train_acc / _counter)
            _train_losses.append(_batch_train_loss / _counter)

            # Calculating and saving validation data accuracy and loss
            _counter = 0
            _batch_validation_acc, _batch_validation_loss = 0.0, 0.0
            for _, (x_batch, y_batch) in enumerate(validation_dataloader):
                assert x_batch.device.type == self.device.type, f'x_batch is on device: {x_batch.device.type}, but should be on: {self.device.type}'
                assert y_batch.device.type == self.device.type, f'y_batch is on device: {y_batch.device.type}, but should be on: {self.device.type}'
                with torch.no_grad():
                    # Forward pass
                    _y_hat = self.forward(x_batch)
                    # Loss calculation
                    _loss = _loss_function(_y_hat, y_batch)
                    # Saving losses and accuracies
                    _batch_validation_acc += self.accuracy(self.predict(_y_hat), y_batch)
                    _batch_validation_loss += _loss.item()
                _counter += 1
            _validation_accuracies.append(_batch_validation_acc / _counter)
            _validation_losses.append(_batch_validation_loss / _counter)

            # Saving instance of model that yields the highest validation acc during training
            if _batch_validation_acc / _counter > _best_validation_accuracy:
                _best_validation_accuracy = _batch_validation_acc / _counter
                torch.save(obj=self.state_dict(), f="Model/best_val_acc.pt")
                _best_epoch = epoch + 1
                _no_improvement_counter = 0
            else:
                _no_improvement_counter += 1

            if _no_improvement_counter >= 40:
                print("Early stopping due to no validation acc. improvement in: ", 40, "epochs.")
                break

        print("Highest validation acc. model saved at epoch: ", _best_epoch, ", with validation acc: ",
              _best_validation_accuracy.item() * 100.0, "%")

        # Saving model on last iteration
        torch.save(obj=self.state_dict(), f="Model/final_epoch.pt")

        # Removing from GPU and Allocating input on CPU
        if self.device.type != "cpu":
            self.device = torch.device('cpu')
            self.to(self.device)
            for _, (data, labels) in enumerate(train_dataloader):
                data = data.to(self.device)
                labels = labels.to(self.device)
            for _, (data, labels) in enumerate(validation_dataloader):
                data = data.to(self.device)
                labels = labels.to(self.device)
        return _train_accuracies, _train_losses, _validation_accuracies, _validation_losses
