import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# LSTM model class
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, device):
        super(LSTMModel, self).__init__()
        # Number of hidden units
        self.hidden_dim = hidden_dim
        # Number of hidden layers
        self.layer_dim = layer_dim

        # batch_first=True means that input tensor will be of shape (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        # Output layer (linear combination of last outputs)
        self.fc = nn.Linear(hidden_dim, output_dim)

        self.device = device

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        h0 = h0.to(device=self.device)
        c0 = c0.to(device=self.device)

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # out.size() --> batch_size, seq_dim, hidden_dim
        # out[:, -1, :] --> batch_size, hidden_dim --> extract outputs from last layer

        out = self.fc(out[:, -1, :]) 
        # out.size() --> batch_size, output_dim

        # Apply softmax activation to output
        activation = nn.Softmax(dim=1)
        out = activation(out)

        return out


# Dataset builder class
class Dataset_builder(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.len = x.shape[0]        
    # Getting the data
    def __getitem__(self, index):    
        return self.x[index], self.y[index]    
    # Getting length of the data
    def __len__(self):
        return self.len


# Function to train model
def train_LSTM_model(model, device, n_epochs, train_loader, optimiser, loss_function, x_test_tensor, y_test_tensor, logging=False, logfile=None):
    model = model.to(device=device)
    for n_epoch in range(n_epochs):
        for i, (inputs, labels) in enumerate(train_loader):
            # if i%10 == 0:
            #     print(f"{i} batches processed")
            inputs = inputs.to(device=device)
            labels = labels.to(device=device)
            optimiser.zero_grad()
            outputs = model(inputs)
            labels = torch.argmax(labels, dim=1)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimiser.step()
            
        with torch.no_grad():
            predictions = model(x_test_tensor)
            labels = torch.argmax(y_test_tensor, dim=1)
            test_loss = loss_function(predictions, labels)
            accuracy = torch.count_nonzero(torch.argmax(predictions, dim=1)==labels)/len(predictions)
            print(f"Model loss after {n_epoch+1} epochs = {test_loss}, accuracy = {accuracy}")
            if logging and logfile is not None:
                logfile.write(f"Model loss after {n_epoch+1} epochs = {test_loss}, accuracy = {accuracy}\n")

    cm_labels = ["bed", "fall", "walk", "pickup", "run", "sitdown", "standup"]
    cm = confusion_matrix(labels.cpu(), torch.argmax(predictions, dim=1).cpu(), normalize="true")
    print("Confusion matrix:")
    print(cm)

    return accuracy, cm


def main():
    USE_GPU = True

    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    print(device)

    logging = True
    logfile_name = "LSTM_final.txt"

    if logging:
        logfile = open(logfile_name, "w")

    # Constants/parameters
    window_size = 1000 # Used in pre-processing
    batch_size = 5 # Used for training
    learning_rate = 0.0001
    n_epochs = 200 # Training epochs
    input_dim = 90
    hidden_dim = 200
    layer_dim = 1
    output_dim = 7

    if logging:
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

    # Convert to torch tensors, move to GPU and reshape x into sequential data (3D)
    x_train_tensor = torch.Tensor(x_train)
    x_test_tensor = torch.Tensor(x_test).to(device=device)
    y_train_tensor = torch.Tensor(y_train)
    y_test_tensor = torch.Tensor(y_test).to(device=device)

    x_train_tensor = torch.reshape(x_train_tensor, (x_train_tensor.shape[0], window_size, -1))
    x_test_tensor = torch.reshape(x_test_tensor, (x_test_tensor.shape[0], window_size, -1))

    # Instantiate LSTM model and loss function
    model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim, device)
    loss_function = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Create data loader for batch gradient descent
    train_dataset = Dataset_builder(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=2)

    # Train model
    accuracy, cm = train_LSTM_model(model, device, n_epochs, train_loader, optimiser, loss_function, x_test_tensor, y_test_tensor, logging, logfile)

    if logging:
        logfile.write(f"\nFinal accuracy = {accuracy}\n")
        logfile.write(f"\nFinal confusion matrix:\n")
        for i in range(len(cm)):
            logfile.write(str(cm[i]))
            
        logfile.close()


if __name__ == "__main__":
    main()

