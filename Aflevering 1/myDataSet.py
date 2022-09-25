import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.io import ImageReadMode


class MyCustomImageDataset(Dataset):
    def __init__(self, annotations_file_dir: str,
                 image_directory_path: str,
                 transform=None,
                 target_transform=None):

        self.img_labels = np.loadtxt(annotations_file_dir,
                                     delimiter=",",
                                     dtype=str,
                                     skiprows=1,
                                     usecols=[1])

        self.img_titles = np.loadtxt(annotations_file_dir,
                                     delimiter=",",
                                     dtype=str,
                                     skiprows=1,
                                     usecols=[2])

        self.image_directory_path = image_directory_path
        self.transform = transform
        self.target_transform = target_transform  # Unused
        self.img_dim = (520, 520)

        self.X = torch.stack([self.__getitem__(i)[0] for i in range(len(self.img_labels))])
        self.Y = self.img_labels

    def __crop__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Function that assures each picture is of equal size by cropping.

         Parameters:
        - image: 2D torch tensor of shape (height, width)

         Returns:
        - image: cropped 2D tensor of shape (cropped height, cropped width)

        """
        assert len(image.shape) == 3, f"image is not 3d tensor, but has shape {image.shape}"
        assert self.img_dim[0] <= image.shape[1] and self.img_dim[1] <= image.shape[
            2], f'image is of shape {image.shape}'
        height_diff = image.shape[1] - self.img_dim[0]
        width_diff = image.shape[2] - self.img_dim[1]

        new_start_h, new_start_w = None, None
        if height_diff % 2 == 0:
            new_start_h = int(height_diff / 2)
        else:
            new_start_h = height_diff // 2
        if width_diff % 2 == 0:
            new_start_w = int(width_diff / 2)
        else:
            new_start_w = width_diff // 2
        return image[:, new_start_h:new_start_h + self.img_dim[0], new_start_w:new_start_w + self.img_dim[1]]

    def __len__(self):
        """
        Function that calculates the number of loaded image objects.

         Returns:
        - length: int representation of nr. of image labels.
        """
        length = len(self.img_labels)
        return length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, object]:
        """
        Function that returns a cropped version of image nr. 'idx'
        at specified 'image_directory_path' in memory.

         Parameters:
        - idx: integer representation of requested file nr.

         Returns:
        - image, label: tuple containing a 2D torch tensor and corresponding label.

        """
        image = read_image(self.image_directory_path + "/" + self.img_titles[idx], mode=ImageReadMode.UNCHANGED)
        label = self.img_labels[idx]
        if self.transform is not None:
            image = self.transform(image)
            return image, label

        else:  # Defaulting to cropping.
            image = self.__crop__(image)
            return image, label


class DataPrep:
    def __init__(self, dataset: MyCustomImageDataset,
                 batch_size: int = 20,
                 data_fraction: float = 0.1,
                 test_fraction: float = 0.2) -> None:
        self.X, self.Y = dataset.X.to(torch.float32), list(dataset.Y)
        self.batch_size = batch_size
        self.test_fraction = test_fraction
        self.data_fraction = data_fraction

        self.label_dict = None
        self.train_X, self.train_Y = None, None
        self.test_X, self.test_Y = None, None

    def prepare(self):
        # One-hot encoding labels
        self.Y = self.one_hot(self.Y)

        # Normalizing pixels values
        self.X = self.normalize_pixels(self.X)

        # Batching data in batches
        self.X, self.Y = self.batch_data(data=self.X,
                                         labels=self.Y,
                                         batch_size=self.batch_size,
                                         randomize=True)

        # Picking out portion size according to 'data_fraction'
        print("Initial nr. of batches: ", len(self.X))
        self.X, self.Y = self.X[:int(self.data_fraction * len(self.X))], self.Y[:int(self.data_fraction * len(self.Y))]
        print("Final nr. of batches: ", len(self.X))

        # Splitting into test and training set according to 'test_fraction'
        self.test_X, self.test_Y = self.X[:int(self.test_fraction * len(self.X))], self.Y[:int(
            self.test_fraction * len(self.Y))]
        self.train_X, self.train_Y = self.X[len(self.test_X):], self.Y[len(self.test_Y):]

    @staticmethod
    def batch_data(data: torch.Tensor,
                   labels: torch.Tensor,
                   batch_size: int,
                   randomize: bool = False) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Takes the input data, shuffles it randomly and repacks it in batches of
        'batch_size'. If len(labels) % batch_size != 0, the final number of
        batches is rounded down.

         Parameters:
        - data: A torch Tensor of shape (nr. data points, nr. channels, height, width).
        - labels: Either tuple, list or 1d numpy array containing the labels for the data.
        - batch_size: The number of data points contained within each batch.
        - randomize: A boolean determining whether the data is randomly shuffled.

         Returns:
        - batches: A tuple of (X_batches, Y_batches)
        """
        assert type(data) is torch.Tensor and len(data.shape) == 4
        assert type(labels) is torch.Tensor and len(labels.shape) == 2

        _indices = [i for i in range(len(labels))]
        if randomize:
            _indices = torch.randperm(len(_indices))
            _indices = _indices.tolist()

        _X_batches, _Y_batches = [], []
        for _batch in range(len(labels) // batch_size):
            _i = _indices[batch_size * _batch: batch_size * (_batch + 1)]
            _X_batches.append(data[_i, ::])
            _Y_batches.append(labels[_i, ::])

        batches = _X_batches, _Y_batches
        return batches

    def one_hot(self, labels: tuple[object] | list[object]) -> torch.Tensor:
        """
        Creates a one-hot encoding of given labels.

         Parameters:
        - labels: tuple or list containing labels.

         Returns:
        - one_hot_Y: 2D torch Tensor of shape (nr. data points, nr. classes).
        """
        assert type(labels) is tuple or type(labels) is list, f'Unrecognized labels type: should be tuple or list.'
        # Creating map from label to integer
        self.label_dict = {}
        for class_idx, label in enumerate(set(labels)):
            self.label_dict[label] = class_idx
        # Calculating Number of data points x Nr of classes one-hot encoding of Y
        nr_labels = len(list(labels))
        nr_unique_labels = len(list(self.label_dict.keys()))
        one_hot_Y = torch.zeros(size=(nr_labels, nr_unique_labels))
        for i, label in enumerate(labels):
            one_hot_Y[i][self.label_dict[label]] = 1
        return one_hot_Y

    @staticmethod
    def normalize_pixels(data: torch.Tensor) -> torch.Tensor:
        """
        Void type function normalizing every value: (0,255) -> (0,1), in input tensor.

         Parameters:
        - data: 4D torch tensor of data of shape (nr. datapoints, nr. channels, height, width)

        """
        _PIXEL_MAX = 255
        for data_point in range(data.shape[0]):
            data[data_point] *= 1.0 / _PIXEL_MAX
        return data
