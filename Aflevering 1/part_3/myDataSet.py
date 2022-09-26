import numpy as np
import torch
from torch.utils.data import Dataset


class MyCustomDataset(Dataset):
    def __init__(self, filename: str,
                 skip_rows: int = 0,
                 label_col: int = 0,
                 feature_cols: tuple[int, ...] = (1, 2),
                 label_map: dict = None) -> None:

        self.labels = np.loadtxt(filename,
                                 dtype=float,
                                 skiprows=skip_rows,
                                 usecols=label_col)

        self.features = np.loadtxt(filename,
                                   dtype=float,
                                   skiprows=skip_rows,
                                   usecols=feature_cols)

        if label_map is None:
            self.label_dict = {}
            _label_counter = 0
            for label in self.labels:
                if label not in list(self.label_dict.keys()):
                    self.label_dict[label] = _label_counter
                    _label_counter += 1
        else:
            self.label_dict = label_map

    def __len__(self):
        """
        Function that calculates the number of loaded image objects.

         Returns:
        - length: int representation of nr. of image labels.
        """
        length = len(self.labels)
        return length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Function that returns a cropped version of image nr. 'idx'
        at specified 'filename' path, in memory.

         Parameters:
        - idx: integer representation of requested file nr.

         Returns:
        - image, label: tuple containing a 2D torch tensor and corresponding label.

        """
        X = torch.from_numpy(self.features[idx]).to(torch.float32)
        Y = self.one_hot(str(int(self.labels[idx])))
        return X, Y

    def one_hot(self, label: str) -> torch.Tensor:
        """
        Creates a one-hot encoding of a label.

         Parameters:
        - label: string repr. of single label

         Returns:
        - one_hot_Y: one-hot encoded 1D torch Tensor of shape (nr. classes, ).
        """
        one_hot_Y = torch.zeros(size=(len(self.label_dict.keys()),))
        one_hot_Y[self.label_dict[label]] = 1
        return one_hot_Y
