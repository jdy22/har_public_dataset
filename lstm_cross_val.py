import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from numpy.random import default_rng
from lstm_training_test1 import LSTMModel, Dataset_builder, train_LSTM_model


def k_fold_split(n_splits, n_instances, random_generator):
    """ Split n_instances into n mutually exclusive splits at random.
    
    Args:
        n_splits (int): Number of splits
        n_instances (int): Number of instances to split
        random_generator (np.random generator): A random generator

    Returns:
        list: a list (length n_splits). Each element in the list should contain a 
            numpy array giving the indices of the instances in that split.
    """
    # Generate a random permutation of indices from 0 to n_instances
    shuffled_indices = random_generator.permutation(n_instances)

    # Split shuffled indices into almost equal sized splits
    split_indices = np.array_split(shuffled_indices, n_splits)

    return split_indices


def train_test_k_fold(n_folds, n_instances, random_generator=default_rng()):
    """ Generate train and test indices at each fold.
    
    Args:
        n_folds (int): Number of folds
        n_instances (int): Total number of instances
        random_generator (np.random generator): A random generator

    Returns:
        list: a list of length n_folds. Each element in the list is a list (or tuple) 
            with two elements: a numpy array containing the train indices, and another 
            numpy array containing the test indices.
    """
    # Split the dataset into k splits
    split_indices = k_fold_split(n_folds, n_instances, random_generator)

    folds = []
    for k in range(n_folds):
        # Pick k as test
        test_indices = split_indices[k]

        # Combine remaining splits as train
        train_indices = np.hstack(split_indices[:k] + split_indices[k+1:])

        folds.append([train_indices, test_indices])

    return folds


def k_fold_cross_val(n_folds, x_train, y_train, device, window_size, input_dim, hidden_dim, layer_dim, output_dim, learning_rate, batch_size, n_epochs):
    # Start cross-validation
    k_confusion_matrices = []
    k_accuracies = []

    for (train_indices, val_indices) in train_test_k_fold(n_folds, len(x_train)):
        print("Starting new fold...")
        train_data = x_train[train_indices,]
        train_labels = y_train[train_indices,]
        val_data = x_train[val_indices,]
        val_labels = y_train[val_indices,]

        # Convert to torch tensors, move to GPU and reshape x into sequential data (3D)
        x_train_tensor = torch.Tensor(train_data)
        x_test_tensor = torch.Tensor(val_data).to(device=device)
        y_train_tensor = torch.Tensor(train_labels)
        y_test_tensor = torch.Tensor(val_labels).to(device=device)

        x_train_tensor = torch.reshape(x_train_tensor, (x_train_tensor.shape[0], window_size, -1))
        x_test_tensor = torch.reshape(x_test_tensor, (x_test_tensor.shape[0], window_size, -1))

        # Create model, loss function, optimiser
        model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)
        loss_function = nn.CrossEntropyLoss()
        optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Create dataloader
        train_dataset = Dataset_builder(x_train_tensor, y_train_tensor)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=2)

        # Train model
        accuracy, cm = train_LSTM_model(model, device, n_epochs, train_loader, optimiser, loss_function, x_test_tensor, y_test_tensor)

        k_confusion_matrices.append(cm)
        k_accuracies.append(accuracy)

    # Calculate average metrics over all folds
    print("Cross-validation completed")
    average_accuracy = sum(k_accuracies)/n_folds
    average_cm = sum(k_confusion_matrices)/n_folds
    print(f"Average accuracy = {average_accuracy}")
    print("Average confusion matrix:")
    print(average_cm)

    return average_accuracy, average_cm


def main():
    USE_GPU = True

    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    print(device)

    logging = True
    logfile_name = "LSTM_crossval2.txt"

    if logging:
        logfile = open(logfile_name, "w")

    # Constants
    window_size = 1000 # Used in pre-processing
    input_dim = 90
    layer_dim = 1
    output_dim = 7
    n_folds = 5

    # Hyperparameters
    batch_size = 10 # Used for training
    learning_rate = 0.0001
    n_epochs = 100 # Training epochs
    hidden_dim = 300

    if logging:
        logfile.write(f"LSTM {n_folds}-fold cross validation\n")
        logfile.write(f"Window size: {window_size}, batch size: {batch_size}, learning rate: {learning_rate}, epochs: {n_epochs}\n")
        logfile.write(f"Input dimension: {input_dim}, hidden dimension: {hidden_dim}, layer dimension: {layer_dim}, output dimension: {output_dim}\n")
        logfile.write("\n")

    # Read in data
    with open("data_test3.pk1", "rb") as file:
        data = pickle.load(file)
    x_train = data[0]
    x_test = data[1]
    y_train = data[2]
    y_test = data[3]

    average_accuracy, average_cm = k_fold_cross_val(n_folds, x_train, y_train, device, window_size, input_dim, hidden_dim, layer_dim, output_dim, learning_rate, batch_size, n_epochs)

    if logging:
        logfile.write(f"Average accuracy = {average_accuracy}\n")
        logfile.write(f"\nAverage confusion matrix:\n")
        for i in range(len(average_cm)):
            logfile.write(str(average_cm[i])+"\n")
            
        logfile.close()


if __name__ == "__main__":
    main()