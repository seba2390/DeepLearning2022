import torch
import torchvision
import matplotlib.pyplot as plt
from myDataSet import *
from myNetworks import *


def test_model(trained_model: NeuralNet | NeuralNet2,
               data: DataPrep,
               nr_batches: int) -> None:
    """
    Void type function that tests a trained model on 'nr_batches' randomly
    selected batches by comparing model prediction with actual label.

     Parameters:
    - trained_model: the trained Neural Net.
    - data: that data to test model on.
    - nr_batches: the nr. of batches to test on.

    """

    # Generating random indices
    rand_indices = []
    while len(rand_indices) < nr_batches:
        _rand_int = torch.randint(low=0, high=len(data.Y), size=(1,)).item()
        if _rand_int not in rand_indices:
            rand_indices.append(_rand_int)
    # Predicting w. model and checking against true labels
    batch_counter = 0
    correct_counter = 0
    with torch.no_grad():
        for _index in rand_indices:
            mod = trained_model.forward(data.X[_index])
            y_hat = trained_model.predict(mod)
            y_real = data.Y[_index]
            print("\n ---  Random Batch: ", batch_counter + 1, " ---")
            for _data_point in range(y_hat.shape[0]):
                pred = list(data.label_dict.keys())[torch.where(y_hat[_data_point] == 1)[0]]
                actual = list(data.label_dict.keys())[torch.where(y_real[_data_point] == 1)[0]]
                print("  ##| Prediction: ", pred, " |--| Actual: ", actual, " |##")
                if pred == actual:
                    correct_counter += 1
            batch_counter += 1
    print("\n #####| ", correct_counter, "/", batch_counter * data.batch_size, " predicted correctly |#####")


def show_image_from_path(filepath: str) -> None:
    """
    Void type function that plots image at path 'filepath' using
    matplotlib.

     Parameters:
    - filepath: string name of filepath relative to current script

    """
    img_tensor = torchvision.io.read_image(filepath, mode=torchvision.io.ImageReadMode.UNCHANGED)
    img_tensor = torchvision.transforms.functional.to_pil_image(img_tensor)
    plt.imshow(img_tensor)
    plt.show()


def show_image_from_idx(index: int,
                        dataset: MyCustomImageDataset,
                        crop: bool = False,
                        resize: bool = False):
    """
       Void type function that plots image nr. index from dataset
       using matplotlib.

        Parameters:
       - index: integer representation of nr. of image.
       - filepath: string name of filepath relative to current script.
       - crop: boolean that determines whether to crop img. or not.

       """
    path = dataset.image_directory_path + "/" + dataset.img_titles[index]
    img_tensor = torchvision.io.read_image(path, mode=torchvision.io.ImageReadMode.UNCHANGED)
    if crop:
        img_tensor = dataset.__crop__(image=img_tensor)
    if resize:
        transform = torchvision.transforms.Resize(dataset.img_dim)
        img_tensor = transform(img_tensor)
    img_tensor = torchvision.transforms.functional.to_pil_image(img_tensor)
    plt.imshow(img_tensor)
    plt.show()