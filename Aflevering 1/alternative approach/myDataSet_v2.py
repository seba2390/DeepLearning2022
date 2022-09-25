import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.io import ImageReadMode


class MyCustomImageDataset2(Dataset):
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

        self.label_dict = None
        self.Y = self.one_hot(self.img_labels.tolist())
        self.X = torch.stack([self.__getitem__(i)[0].to(torch.float32) for i in range(len(self.img_labels))])

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
        label_Y = torch.zeros(size=(len(self.label_dict.keys()),))
        label_Y[self.label_dict[label]] = 1
        label = label_Y
        if self.transform is not None:
            image = self.transform(image)
            return image.to(torch.float32), label

        else:  # Defaulting to cropping.
            image = self.__crop__(image)
            return image.to(torch.float32), label

    def one_hot(self, labels) -> torch.Tensor:
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
