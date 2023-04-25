import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from numpy.random import default_rng
from lstm_training_test1 import LSTMModel, Dataset_builder, train_LSTM_model
from lstm_cross_val import k_fold_cross_val
from sklearn.model_selection import train_test_split


def main():
    USE_GPU = True

    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    print(device)

    logging = True
    logfile_name = "LSTM_gridsearch1.txt"

    if logging:
        logfile = open(logfile_name, "w")

    # Constants
    window_size = 1000 # Used in pre-processing
    input_dim = 90
    layer_dim = 1
    output_dim = 7
    n_folds = 5

    # Hyperparameters for grid search
    batch_sizes = [5, 10, 20, 50, 100, 200] # Used for training
    learning_rates = [0.001, 0.0001, 0.00001]
    no_epochs = [100, 200] # Training epochs
    hidden_dims = [200]

    if logging:
        logfile.write(f"LSTM grid search\n")
        logfile.write(f"Window size: {window_size}, input dimension: {input_dim}, layer dimension: {layer_dim}, output dimension: {output_dim}\n")
        logfile.write("\n")

    # Read in data
    with open("data_test3.pk1", "rb") as file:
        data = pickle.load(file)
    x_train = data[0]
    x_test = data[1]
    y_train = data[2]
    y_test = data[3]

    # Split training data into train and val data for grid search
    x_train_train, x_train_val, y_train_train, y_train_val = train_test_split(x_train, y_train, test_size=0.20, random_state=1000)

    # Convert to torch tensors, move to GPU and reshape x into sequential data (3D)
    x_train_tensor = torch.Tensor(x_train_train)
    x_test_tensor = torch.Tensor(x_train_val).to(device=device)
    y_train_tensor = torch.Tensor(y_train_train)
    y_test_tensor = torch.Tensor(y_train_val).to(device=device)

    x_train_tensor = torch.reshape(x_train_tensor, (x_train_tensor.shape[0], window_size, -1))
    x_test_tensor = torch.reshape(x_test_tensor, (x_test_tensor.shape[0], window_size, -1))

    train_dataset = Dataset_builder(x_train_tensor, y_train_tensor)

    for batch_size in batch_sizes:
        for learning_rate in learning_rates:
            for n_epochs in no_epochs:
                for hidden_dim in hidden_dims:
                    print(f"Batch size: {batch_size}, learning rate: {learning_rate}, no. epochs: {n_epochs}, hidden dimension: {hidden_dim}")
                    # Create data loader for batch gradient descent
                    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=2)

                    # Instantiate LSTM model and loss function
                    model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim, device)
                    loss_function = nn.CrossEntropyLoss()
                    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

                    # Train model
                    accuracy, cm = train_LSTM_model(model, device, n_epochs, train_loader, optimiser, loss_function, x_test_tensor, y_test_tensor)

                    if logging:
                        logfile.write(f"Batch size: {batch_size}, learning rate: {learning_rate}, no. epochs: {n_epochs}, hidden dimension: {hidden_dim} - accuracy = {accuracy}\n")
                            
    logfile.close()


if __name__ == "__main__":
    main()