import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import ParameterGrid

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
import torch.nn as nn

import os
import csv
import yaml

from torchinfo import summary


class DeepNeuralNetwork(nn.Module):
    """ Class that defines the neural network. This class is customizable in the sense that it takes input_size, output_size and
     hidden_neurons and activation function as input so basically one can customize the whole network based on these parameters.

    Inputs:
        input_size                 - the number of input features you want to pass to the neural network.
                                     Example: 5


        output_size                - the size of the prediction, note that this network is designed to have a linear layer in the end. This
                                      is a concious choice based on the problem statement
                                      Example: 1

        hidden_neurons             - this is an array, stating the neurons in each hidden layer of the neural network.
                                     Example: [30, 50, 100, 80, 40, 10], so this states that the first hidden layer has 30 neurons, second has 50 and so on.

        activation                 - this states the non linearity we want to introduce between each layer, as stated before the final activation is always linear.
                                     this is just between layers.
                                     Example: 'relu', 'sigmoid'

    Outputs: It will initialize a CustomDeepNeuralNetwork object for you

    """
    def __init__(self, input_size, output_size, hidden_neurons, activation, dropout_prob = 0.2):
        super().__init__()
        layers = [input_size] + hidden_neurons + [output_size]
        self.layers = []

        for i in range(len(layers)- 1):
            layer = nn.Linear(layers[i], layers[i+1])     # Creating the hidden layers from the array given as input
            layer_activation = self.get_activation(activation)
            dropout = nn.Dropout(dropout_prob)
            self.layers.extend([layer, layer_activation, dropout])

        # Remove the last dropout layer and activation so that final layer is linear
        self.layers.pop()
        self.layers.pop()

        self.network = nn.Sequential(*self.layers)

        # Initialize weights with mean=0 and std=0.1
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0, std=0.1)
                nn.init.constant_(layer.bias, 0)  # Initialize biases to zero

    def forward(self, x):
        return self.network(x)
    
    # Defining all the activation function available for this class, one can add more if needed
    def get_activation(self, name):
        if name == 'relu':
            return nn.ReLU()
        elif name == 'leaky_relu':
            return nn.LeakyReLU()
        elif name == 'sigmoid':
            return nn.Sigmoid()
        elif name == 'tanh':
            return nn.Tanh()
        elif name == 'softmax':
            return nn.Softmax(dim=1)
        elif name == 'exp':
            return  nn.ELU()
        else:
            raise ValueError(f"Invalid activation function: {name}")

# Defining a custom dataset class according to how the signals are defined
class DataPrep(Dataset):
    """
        Initializes a custom dataset with input signals and target values.

        Args:
            data (numpy.ndarray): The dataset containing input signals and target values.
        """
    def __init__(self, data):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cuda':
            self.inputs = torch.tensor(data[:, :-1], dtype=torch.float32).cuda()  # Load inputs as a tensor and move to CUDA
            self.targets = torch.tensor(data[:, -1], dtype=torch.float32).cuda()  # Load targets as a tensor and move to CUDA
        else:
            self.inputs = torch.tensor(data[:, :-1], dtype=torch.float32)  # Load inputs as a tensor and move to CUDA
            self.targets = torch.tensor(data[:, -1], dtype=torch.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        x = self.inputs[index]
        y = self.targets[index]
        return x, y

def train_model(signals_train, criterion, hyperparameters, training_output = 2):
    """
    Trains a PyTorch model using the given data, criterion, and optimizer.

    Args:
        signals_train (numpy.ndarray): Training data with input signals and target values.
        criterion: The loss criterion to compute the training loss.
        hyperparameters: library of selected hyperparameter values
        training_output (int): default set to output training metrics at every epoch. Set to 2 if 
                                loss plot is desired during the middle of training. Set to 3 if no 
                                training metric output is desired.
    """
    hidden_neurons = hyperparameters['neurons_per_layer']
    activation = hyperparameters['activation']
    learning_rate = hyperparameters['learning_rate']
    batch_size = hyperparameters['batch_size']
    num_epochs = hyperparameters['num_epochs']

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if training_output != 3:
        print('Running on:', device)
    if device == "cpu":
        train_loader = DataLoader(DataPrep(signals_train), batch_size, shuffle=True, num_workers = -1)
    else:
        train_loader = DataLoader(DataPrep(signals_train), batch_size, shuffle=True, pin_memory=True)

    input_size = np.shape(signals_train)[1]-1
    output_size = 1
    model = DeepNeuralNetwork(input_size, output_size, hidden_neurons, activation)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()  # Set the model in training mode
    model.to(device)

    epoch_loss = []
    for epoch in range(num_epochs):
        mini_loss = []
        for inputs, targets in train_loader:
            optimizer.zero_grad()  # Clear the gradients
            outputs = model(inputs)  # Forward pass
            outputs = outputs.squeeze()
            loss = criterion(outputs, targets)  # Compute the loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update the weights
            mini_loss.append(loss.item())
                
        epoch_loss.append(np.mean(mini_loss))
        if training_output == 1:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss[epoch]}")
            
    # Plot the loss
    if training_output == 2:
        plt.plot(epoch_loss)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.show()    

    train_features = signals_train[:, :-1] # Prepare test and train data for evaluation   
    train_tensor = torch.tensor(train_features, dtype=torch.float32)  # Convert the train data to PyTorch tensors
    model.eval()  # Set the model in evaluation mode
    model.cpu()  # Move the model to CPU

    # Make predictions on the train dataset
    with torch.no_grad():
        train_predictions = model(train_tensor)

    # Convert the predictions and ground truth to NumPy arrays
    train_predictions = train_predictions.numpy()
    train_targets = signals_train[:, -1]

    # Calculate the R-squared value for test and train data
    r2_train = r2_score(train_targets, train_predictions)
    if training_output != 3:
        print(summary(model, verbose = 1))
        print("R-squared value (train):", r2_train)
        print("Finished Training")
    return model, r2_train

def test_model(model, signals_test, testing_output = 1):
    """
    evaluate model performance
    
    """     
    # Prepare test data for evaluation
    test_features = signals_test[:, :-1]
    test_targets = signals_test[:, -1]  # Assuming the last column represents the target values

    # Convert the test data to PyTorch tensors
    test_tensor = torch.tensor(test_features, dtype=torch.float32)

    # Make predictions on the test dataset
    with torch.no_grad():
        test_predictions = model(test_tensor)

    # Convert the predictions to a NumPy array
    test_predictions = test_predictions.numpy()
    
    # Calculate the R-squared value for test data
    r2_test = r2_score(test_targets, test_predictions)
    if testing_output == 1:
        print("R-squared value (test):", r2_test)

    return test_predictions, r2_test


def grid_search(param_grid, signals_train, signals_test, results = 'Results'):
    
     # Generate all possible combinations of hyperparameters
    param_combinations = list(ParameterGrid(param_grid))

    # Perform grid search
    test_r2 = []
    for params in range(len(param_combinations)):    
        criterion = nn.MSELoss()
    
        # Train and evaluate the model
        model, _ = train_model(signals_train, criterion, param_combinations[params], training_output = 3)
        _, r2_test = test_model(model, signals_test, 3)
        test_r2.append(r2_test)
        print(f"Test R2 score for hyperparameter combination {params+1}/{len(param_combinations)}: {r2_test}")
    best_score_index = np.argmax(test_r2)

    # Save optimum hyperparameters to pyaml configurable file
    yaml_file_path = results + "/best_hypers.yaml"
    with open(yaml_file_path, "w") as yaml_file:
        yaml.dump(param_combinations[best_score_index], yaml_file, default_flow_style=False)

    # save hyperparameter search to file with test scores
    csv_file_path = results + "/hyperparameter_grid.csv"
    for i, entry in enumerate(param_combinations):
        entry["test_score"] = test_r2[i]

    # Save data to CSV file
    if not(os.path.exists(results)):
        os.mkdir(results)
    with open(csv_file_path, "w", newline="") as csvfile:
        fieldnames = ['num_epochs', 'activation', 'neurons_per_layer', 'learning_rate', 'batch_size','test_score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()  # Write header
        for entry in param_combinations:
            writer.writerow(entry)

    print(f"Data saved to {csv_file_path}")
    print("optimum hyperparameters found to be:")
    print(param_combinations[best_score_index])
    print(f'With an r2 test score of {test_r2[best_score_index]}')


