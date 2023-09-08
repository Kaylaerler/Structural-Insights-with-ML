import csv
import yaml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split
import pickle

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
import torch.nn as nn
from torchinfo import summary

import preprocess_data

def create_features(signals):
    """  
    Inputs:
        signals - displacement, velocity, acceleration, force columns  

    Outputs:
        
    """
    t   = signals[:,0]
    ux  = signals[:,1]
    vx  = signals[:,2]
    ax  = signals[:,3]
    signvx = np.sign(vx)
    Y   = signals[:,-1]
    
    signals_out = np.vstack((ux, vx, ax, signvx, Y)).T
    return signals_out


class DeepNeuralNetwork(nn.Module):
    """ Class that defines the neural network. This class is customizable in the sense that it takes input_size, output_size and
     hidden_neurons and activation function as input so basically one can customize the whole network based on these parameters.

    Args:
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
    def __init__(self, input_size, output_size, hidden_neurons, hidden_layers, activation, dropout_prob = 0.2):
        super().__init__()
        layers = np.append(input_size, hidden_neurons*np.ones((hidden_layers,1)))
        layers = np.append(layers, output_size).astype(int)
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

class DataPrep(Dataset):
    """
    Class to prepare data for PyTorch model
    
    Args:
        data (numpy.ndarray): Training data with input signals and target values.
    
    Outputs: It will initialize a DataPrep object for you
    """
    def __init__(self, X, y):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cuda':
            self.inputs = torch.tensor(X, dtype=torch.float32).cuda()  # Load inputs as a tensor and move to CUDA
            self.targets = torch.tensor(y, dtype=torch.float32).cuda()  # Load targets as a tensor and move to CUDA
        else:
            self.inputs = torch.tensor(X, dtype=torch.float32)  # Load inputs as a tensor and move to CUDA
            self.targets = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        x = self.inputs[index]
        y = self.targets[index]
        return x, y
    
def test_model(model, X_test, y_test = None):
    """
    evaluate model performance on test data and return testing predictions and r2 score

    Args:
        model: trained model
        signals_test (numpy.ndarray): Testing data with input signals and target values.
        testing_output (int): default set to output testing metrics. Set to 2 if...
        denorm (bool): default set to True. Set to False if no denormalization is desired.
        norm_params (tuple): default set to None. Set to tuple of normalization parameters if denorm is desired.

    Returns:
        test_predictions (numpy.ndarray): Testing predictions
        r2_test (float): R-squared value for test data   
    """     
    test_tensor = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        test_predictions = model(test_tensor)
    test_predictions = test_predictions.numpy()
    if y_test is None:
       r2_test = None
    else:
        r2_test = r2_score(y_test, test_predictions)
    return test_predictions, r2_test

def train_model(signals_library, hyperparameters, path_names, training_output = 2):
    """
    Trains a PyTorch model using the given data, criterion, and optimizer.

    Args:
        signals_train (numpy.ndarray): Training data with input signals and target values.
        criterion: The loss criterion to compute the training loss.
        hyperparameters: library of selected hyperparameter values
        training_output (int): default set to output training metrics at every epoch. Set to 2 if 
                                loss plot is desired during the middle of training. Set to 3 if no 
                                training metric output is desired.


    Returns:
        model: trained model
        r2_train (float): R-squared value for train data
    """
    hidden_neurons = hyperparameters['hidden_neurons']
    hidden_layers = hyperparameters['hidden_layers']
    activation = hyperparameters['activation']
    learning_rate = hyperparameters['learning_rate']
    batch_size = hyperparameters['batch_size']
    num_epochs = hyperparameters['num_epochs']
    dropout_prob = hyperparameters['dropout_prob']


    # Prepare data for training
    signals_numpy = preprocess_data.library_to_numpy(signals_library)
    signals = create_features(signals_numpy)
    X_norm, X_norm_params = preprocess_data.z_score_normalize(signals[:,:-1])
    y_norm, y_norm_params = preprocess_data.z_score_normalize(signals[:,-1])
    norm_params = X_norm_params, y_norm_params
    X_train, X_test, y_train, y_test = train_test_split(X_norm, y_norm, test_size=0.33, random_state=42)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if training_output != 3:
        print('Running on:', device)
    if device == "cpu":
        train_loader = DataLoader(DataPrep(X_train, y_train), batch_size, shuffle=True, num_workers = -1)
    else:
        train_loader = DataLoader(DataPrep(X_train, y_train), batch_size, shuffle=True, pin_memory=True)

    input_size = np.shape(signals)[1]-1
    output_size = 1
    model = DeepNeuralNetwork(input_size, output_size, hidden_neurons, hidden_layers, activation, dropout_prob)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    model.train()  # Set the model in training mode
    model.to(device)

    epoch_loss = []
    for epoch in range(num_epochs):
        mini_loss = []
        for inputs, target in train_loader:
            optimizer.zero_grad()  # Clear the gradients
            outputs = model(inputs)  # Forward pass
            outputs = outputs.squeeze()
            loss = criterion(outputs, target)  # Compute the loss
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

    _, train_r2 = test_model(model, X_train, y_train)
    _, test_r2  = test_model(model, X_test,  y_test)
 
    if training_output != 3:
        print(summary(model))
        print("R-squared value (train):", train_r2)
        print("R-squared value (test):",  test_r2)
        print("Finished Training")
        torch.save(model.state_dict(), path_names['saved_model'])
        yaml.dump(hyperparameters, open(path_names['saved_hypers'], 'w'))
        with open(path_names['norm_params'], 'wb') as fp:
            pickle.dump(norm_params, fp)
    return model, train_r2, test_r2


def grid_search(param_grid, signals_library, path_names):
    """
    Perform grid search on hyperparameters and save optimum hyperparameters to pyaml configurable file

    Args:
        param_grid (dict): dictionary of hyperparameters to search over
        signals_library:

    Returns:
        None
    """
    
     # Generate all possible combinations of hyperparameters
    param_combinations = list(ParameterGrid(param_grid))

    # Perform grid search
    test_r2 = []
    for params in range(len(param_combinations)):    
        _, _, r2_test = train_model(signals_library, param_combinations[params], path_names, training_output = 3)
        test_r2.append(r2_test)
        print(f"Test R2 score for hyperparameter combination {params+1}/{len(param_combinations)}: {r2_test}")
    best_score_index = np.argmax(test_r2)

    # Save trained best model and hyperparameters
    tuned_parameters = param_combinations[best_score_index]
    _, _, _ = train_model(signals_library, tuned_parameters, path_names)

    # save hyperparameter search to file with test scores
    for i, entry in enumerate(param_combinations):
        entry["test_score"] = test_r2[i]
    with open(path_names['grid_results'], "w", newline="") as csvfile: # Save data to CSV file
        fieldnames = list(param_grid.keys())
        fieldnames.append("test_score")
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()  # Write header
        for entry in param_combinations:
            writer.writerow(entry)

    print(f"Data saved to {path_names['grid_results']}")
    print("optimum hyperparameters found to be:")
    print(param_combinations[best_score_index])
    print(f'With an r2 test score of {test_r2[best_score_index]}')

def initiate_saved_model(hyper_path, model_path, signals):
    """
    Load saved model and hyperparameters
    
    Args:
        hyper_path (str): path to saved hyperparameters
        model_path (str): path to saved model
        signals (numpy.ndarray): Training data with input signals and target values. Used to determine input size of model.
        
    Returns:
        model: trained model
    """
    hyperparameters = yaml.load(open(hyper_path, 'r'), Loader=yaml.SafeLoader)
    input_size = np.shape(signals)[1]-1
    output_size = 1
    hidden_neurons = hyperparameters['hidden_neurons']
    hidden_layers = hyperparameters['hidden_layers']
    activation = hyperparameters['activation']
    model = DeepNeuralNetwork(input_size, output_size, hidden_neurons, hidden_layers, activation)
    model.load_state_dict(torch.load(model_path))
    return model

def predict(signals, norm_params, model):
    """
    """
    X_norm_params, y_norm_params = norm_params
    signals = create_features(signals)
    X_norm, _ = preprocess_data.z_score_normalize(signals[:,:-1], X_norm_params)
    prediction_norm, _ = test_model(model, X_norm)
    prediction = preprocess_data.z_score_normalize_inverse(prediction_norm, y_norm_params)
    return prediction