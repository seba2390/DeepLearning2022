import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image


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
        self.target_transform = target_transform
        self.img_dim = (520, 520)

        self.X = torch.stack([self.__getitem__(i)[0] for i in range(len(self.img_labels))])
        self.Y = self.img_labels

    def __crop__(self, image: torch.Tensor) -> torch.Tensor:
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
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = read_image(self.image_directory_path + "/" + self.img_titles[idx])
        image = self.__crop__(image)
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
