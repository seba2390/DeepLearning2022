import torch


def test_model(trained_model,
               data,
               nr_batches: int = 0,
               verbose: bool = False) -> None:
    """
    Void type function that tests a trained model on 'nr_batches' randomly
    selected batches by comparing model prediction with actual label.

     Parameters:
    - trained_model: the trained Neural Net.
    - data: that data to test model on.
    - nr_batches: the nr. of batches to test on.
    - verbose: boolean determining whether to print progress.

    """
    available_nr_batches = data.__len__()
    assert nr_batches <= available_nr_batches, f'Requested nr. batches is larger than nr. batches available in data.'

    # Predicting w. model and checking against true labels
    batch_counter = 0
    correct_counter = 0
    with torch.no_grad():
        for _index, (x, y) in enumerate(data):
            y_hat = trained_model.predict(trained_model.forward(x))
            y_real = y
            if verbose:
                print("\n ---  Random Batch: ", batch_counter + 1, " ---")
            for _data_point in range(y_hat.shape[0]):
                pred = y_hat[_data_point]
                actual = y[_data_point]
                if verbose:
                    print("  ##| Prediction: ", pred, " |--| Actual: ", actual, " |##")
                if torch.sum(pred == actual) / len(pred) == 1:
                    correct_counter += 1
            batch_counter += 1
    print("\n #####| ", correct_counter, "/", batch_counter * data.batch_size, " items in ", available_nr_batches,
          " test batches predicted correctly ~ acc: ", round(correct_counter / (batch_counter * data.batch_size), 4),
          " |#####")
