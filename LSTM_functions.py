"""
Module Name: LSTM_functions
Author: Kayla Erler
Version: 1.0.0
Date: 4/29/2024
License: GNU

Description:
-----------
LSTM_functions is a collection of useful functions and utilities for tasks related to forming a deep neural network 
                from signals that were constructed using the preprocess_data function developed by the same author.
                The deep neural network is constructed using PyTorch and implements mini-batch gradient descent with
                the Adam optimizer. The network is customizable in the sense that the user can specify the number of
                hidden layers, the number of neurons in each hidden layer, the activation function, the learning rate,
                the batch size, the number of epochs, and the dropout probability. The network is designed to have a
                linear output layer. The network is trained using the mean squared error loss function. The network
                can be trained using a random grid search over the hyperparameters specified by the user. The network
                can also be trained using a saved model and hyperparameters. The network can be used to predict the
                output of a set of input signals.

Usage:
------
import LSTM_functions 

# Example Usage
result = DNN_functions.some_function(arg1, arg2)

"""
# modules developed for this project
import preprocess_data

# Open source modules available in python
import csv
import yaml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split
import pickle
import os

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
import torch.nn as nn
from torchinfo import summary
from torchmetrics.regression import R2Score

def create_features(signals):
    """  
    Creates features for the DNN model from the input signals. The features are displacement, velocity, acceleration, signum of velocity, and target value.
    
    Args:
        signals (numpy.ndarray): Input signals to create features for the DNN model (see preprocess_data for the signal format)

    Outputs:
        signals_out (numpy.ndarray): Selected input signals for the DNN model
        
    """
    
    ux  = signals[:,1] # displacement
    vx  = signals[:,2] # velocity
    ax  = signals[:,3] # accleration
    signvx = np.sign(vx) # signum of velocity (1 if positive, -1 if negative)
    f   = signals[:,-1]  # horizontal force recording
    signals_out = np.vstack((ux, vx, ax, signvx, f)).T
    
    return signals_out

def dictionary_to_tensor(signals_dictionary, n_features = 4, n_outputs = 1):
    """
    Converts the dictionary of signals to a 3D tensor array where each test run is stacked on the previous.
    This procedure is followed instead of using a 3D array to avoid issues with the varying number of samples 
    in each test run.

    Args:
        signals_dictionary (dict): Dictionary of all input signals

    Returns:
        X_tensor (torch.tensor): 3D tensor of input signals
        of the form X[i,j,k]
        where, i corresponds to the time history values in a run
               j corresponds to a specific run
               k corresponds to a feature.

        Y_tensor (torch.tensor): 3D tensor of output signalstensor_X (torch.tensor): 3D tensor of input signals
        of the form X[i,j,k]
        where, i corresponds to the time history values in a run
               j corresponds to a specific run
               k corresponds to a predicted output
         
    """
    # find longest run to create a zero padded tensor
    i = 0
    for ii in range(len(signals_dictionary)):
        i = max((signals_dictionary[ii]['data'].shape[0],i))

    j = len(signals_dictionary)
    kx = n_features
    ky = n_outputs

    # initiate pytorch tensor for input signals
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tensor_X = torch.zeros((i, j, kx), dtype = torch.float32).to(device)
    tensor_Y = torch.zeros((i, j, ky), dtype = torch.float32).to(device)
    
    for i in range(len(signals_dictionary)):
        signals = signals_dictionary[i]['data']
        features = create_features(signals)
        tensor_X[0:features.shape[0],i,:] = torch.tensor(features[:,0:-1], dtype = torch.float32).to(device)
        tensor_Y[0:features.shape[0],i,0] = torch.tensor(features[:,-1], dtype = torch.float32).to(device)
    return tensor_X, tensor_Y

def create_feature_dictionary(signals_training, signals_validation, singals_testing, path_names):
    """
    Creates a dictionary of features for the DNN model from the input signals. 
        The features are displacement, velocity, acceleration, signum of velocity, and target value.

    Args:
        signals_dictionary (dict): Dictionary of input signals to create features for the DNN model (see preprocess_data for the signal format)
        path_names (dict): Dictionary of file paths

    Outputs:
        feature_dictionary (dict): Dictionary of selected input signals for the DNN model split into training, validation, and testing sets

    """
    # Prepare data for training
    signals_train_X, signals_train_y = dictionary_to_tensor(signals_training)
    signals_val_X,   signals_val_y   = dictionary_to_tensor(signals_validation)
    signals_test_X,  signals_test_y  = dictionary_to_tensor(singals_testing)

    X_train, X_norm_params = preprocess_data.z_score_normalize(signals_train_X)
    y_train, y_norm_params = preprocess_data.z_score_normalize(signals_train_y)
    norm_params = X_norm_params, y_norm_params
    X_val, _ = preprocess_data.z_score_normalize(signals_val_X, X_norm_params)
    y_val, _ = preprocess_data.z_score_normalize(signals_val_y, y_norm_params)
    X_test, _ = preprocess_data.z_score_normalize(signals_test_X, X_norm_params)
    y_test, _ = preprocess_data.z_score_normalize(signals_test_y, y_norm_params)
    
    feature_dictionary = {'X_train': X_train, 
                          'X_val': X_val, 
                          'X_test': X_test, 
                          'y_train': y_train, 
                          'y_val': y_val, 
                          'y_test': y_test,
                          'norm_params': norm_params}

    if path_names['save_results']:
        if not os.path.exists(path_names['results']):
            os.mkdir(path_names['results'])
        with open(path_names['norm_params'], 'wb') as fp:
            pickle.dump(norm_params, fp)
    return feature_dictionary, norm_params


#################################### NEED TO Modify this function to allow for user defined dataset ####################################
def create_user_feature_dictionary(X,y, path_names):
    """
    Alternative function to "create_feature_dictionary" that allows all other DNN functions to be usable for a user defined dataset.  
        Creates a dictionary of features for the DNN model from the input signals.
    Args:
        X (numpy.ndarray): Matrix of input features with columns representing features and rows representing examples
        y (numpy.ndarray): Target values for the DNN model with rows representing examples
    Outputs:
        feature_dictionary (dict): Dictionary of selected input signals for the DNN model split into training, validation, and testing sets
    """
    # Prepare data for training
    X_norm, X_norm_params = preprocess_data.z_score_normalize(X) # normalize input features
    y_norm, y_norm_params = preprocess_data.z_score_normalize(y) # normalize target values
    norm_params = X_norm_params, y_norm_params
    X_train_val, X_test, y_train_val, y_test = train_test_split(X_norm, y_norm, test_size=0.33, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.33, random_state=42)
    feature_dictionary = {'X_train': X_train, 
                          'X_val': X_val, 
                          'X_test': X_test, 
                          'y_train': y_train, 
                          'y_val': y_val, 
                          'y_test': y_test,
                          'norm_params': norm_params}
    

    if path_names['save_results']:
        if not os.path.exists(path_names['results']):
            os.mkdir(path_names['results'])
        with open(path_names['norm_params'], 'wb') as fp:
            pickle.dump(norm_params, fp)
    return feature_dictionary
###################################################################################################################################


class LSTM(nn.Module):
    """ 
    Class to create a deep neural network using PyTorch  
    
    Args:

    Outputs: LSTM object initiated

    """
    def __init__(self):
        super().__init__()
        layers = np.append(input_size, hidden_neurons*np.ones((hidden_layers,1)))
        layers = np.append(layers, output_size).astype(int)
        self.layers = []
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for i in range(len(layers)- 1):
            layer = nn.Linear(layers[i], layers[i+1]).to(device)     # Creating the hidden layers from the array given as input
            layer_activation = self.get_activation(activation).to(device)
            dropout = nn.Dropout(dropout_prob).to(device)
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
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        x = self.X[index]
        y = self.y[index]
        return x, y
    
def customized_loss(output, target):
    mse = torch.nn.MSELoss()
    output1 = output[:, :, 0:1]
    target1 = target[:, :, 0:1]
    return 0.5 * torch.maximum(mse(output1, target1))
    
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
    model.eval()    
    with torch.no_grad():
        test_predictions = model(X_test)
    if y_test is None:
       r2_test = None
    else:
        r2score = R2Score()
        r2_test = r2score(y_test, test_predictions)
    return test_predictions, r2_test

def train_model(features, hyperparameters, path_names, training_output = 2):
    """
    Trains a PyTorch model using the given data, criterion, and optimizer.

    Args:
        features (dict): dictionary containing numpy arrays for training, testing and validation data
        criterion      : The loss criterion to compute the training loss.
        hyperparameters: dictionary of selected hyperparameter values
        training_output (int): set to 1 to output training metrics at every epoch. 
                                Set to 2 if loss plot is desired with scoring metrics only provided at end. 
                                Set to 3 if no training metric output is desired, only validation.


    Returns:
        model: trained model
        r2_train (float): R-squared value for train data
        r2_val   (float): R-squared value for validation data
    """
    hidden_neurons = hyperparameters['hidden_neurons']
    hidden_layers = hyperparameters['hidden_layers']
    activation = hyperparameters['activation']
    learning_rate = hyperparameters['learning_rate']
    batch_size = hyperparameters['batch_size']
    num_epochs = hyperparameters['num_epochs']
    print_epoch = int(num_epochs/10)
    dropout_prob = hyperparameters['dropout_prob']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if training_output != 3:
        print('Running on:', device)
    if device == "cpu":
        train_loader = DataLoader(DataPrep(features['X_train'], features['y_train']), batch_size, shuffle=True, num_workers = -1)
    else:
        train_loader = DataLoader(DataPrep(features['X_train'], features['y_train']), batch_size, shuffle=True)
        
    input_size  = np.shape(features['X_train'])[1]
    output_size = 1
    model = LSTM(input_size, output_size, hidden_neurons, hidden_layers, activation, dropout_prob).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    model.train()  # Set the model in training mode

    epoch_train_loss = []
    epoch_val_loss = []
    for epoch in range(num_epochs):
        batches_run = 0
        train_loss = 0
        model.train()
        for inputs, target in train_loader:
            optimizer.zero_grad()  # Clear the gradients
            outputs = model(inputs)  # Forward pass
            outputs = outputs.squeeze()
            loss = criterion(outputs, target)  # Compute the loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update the weights
            train_loss += loss.item()
            batches_run += 1

        train_loss /= batches_run # average total loss by number of batches
        model.eval()
        with torch.no_grad():
            outputs = model(X_val_tensor)
            outputs = outputs.squeeze()
            val_loss = criterion(outputs, y_val_tensor)  # Compute the loss
                
        epoch_train_loss.append(train_loss)
        epoch_val_loss.append(val_loss.item())

        if training_output == 1:
            if epoch % print_epoch == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_train_loss[epoch]}")
            
    # Plot the loss
    if training_output == 2 or training_output == 1:
        plt.plot(epoch_train_loss, label = 'Training Loss')
        plt.plot(epoch_val_loss, label = 'Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.show()    

    _, train_r2 = test_model(model, features['X_train'], features['y_train'])
    _, val_r2   = test_model(model, features['X_val'],   features['y_val'])
    _, test_r2  = test_model(model, features['X_test'],  features['y_test'])
 
    if training_output != 3:
        print(summary(model))
        print("R-squared value (train):", train_r2)
        print("R-squared value (test):",  test_r2)
        print("Finished Training")
        if path_names['save_results']:
            torch.save(model.state_dict(), path_names['saved_model'])
            # add test score key to hyperparameter dictionary
            hyperparameters["test_score"] = float(test_r2)
            yaml.dump(hyperparameters, open(path_names['saved_hypers'], 'w'))
    return model, train_r2, val_r2, test_r2


def random_search(features, param_grid, path_names, search_space_ratio):
    """
    Perform random grid search on hyperparameters and save optimum hyperparameters to pyaml configurable file
    as well as output to a csv with all hyperparameter combinations and their respective test scores.

    Args:
        param_grid (dict)         : dictionary of hyperparameters to search over
        features (dict)           : dictionary of features for train, val, and testing
        path_names (dict)         : dictionary of file paths
        search_space_ratio (float): ratio of hyperparameter combinations to search over (0 to 1)

    Returns:
        best_model (model)        : trained model with optimum hyperparameters
    """
    
    # Generate all possible combinations of hyperparameters
    param_combinations = list(ParameterGrid(param_grid))

    # Randomly select a subset of hyperparameter combinations to search over
    random_param_index = np.random.choice(len(param_combinations), int(len(param_combinations)*search_space_ratio), replace=False)
    print(f'{len(random_param_index)} hyper combination out of {len(param_combinations)} will be searched over')
    random_param_combinations = [param_combinations[i] for i in random_param_index]

    # Perform grid search over the randomly selected hyperparameter combinations
    val_r2 = []
    for params in range(len(random_param_combinations)):    
        _, _, r2_val, _ = train_model(features, random_param_combinations[params], path_names, training_output = 3)
        val_r2.append(r2_val)
        print(f"Validation R2 score for hyperparameter combination {params+1}/{len(random_param_combinations)}: {r2_val}")
    best_score_index = np.argmax(val_r2)

    print('-------------------------------------------')
    print("best hyperparameters found to be:")
    print(random_param_combinations[best_score_index])
    print('-------------------------------------------')

    # Save trained best model and hyperparameters
    tuned_parameters = random_param_combinations[best_score_index]
    best_model, _, _, _ = train_model(features, tuned_parameters, path_names)

    # add validation score to hyperparameter dictionary 
    for i in range(len(random_param_combinations)):
        random_param_combinations[i]["val_score"] = float(val_r2[i])

    # if "test_score" is a key in random_param_combinations remove it
    for i in range(len(random_param_combinations)):
        if "test_score" in random_param_combinations[i].keys():
            random_param_combinations[i].pop("test_score")

    # save hyperparameter search to file with test scores
    with open(path_names['grid_results'], "w", newline="") as csvfile: # Save data to CSV file
        fieldnames = list(param_grid.keys())
        fieldnames.append("val_score")
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()  # Write header
        for entry in random_param_combinations:
            writer.writerow(entry)

    print(f"Data saved to {path_names['grid_results']}")
    return best_model

def initiate_pretrained_network(features, hyper_path, model_path):
    """
    Load saved model and hyperparameters
    
    Args:
        hyper_path (str): path to saved hyperparameters
        model_path (str): path to saved model
        features (dict) : dictionary of features for train, val and testing
        
    Returns:
        model: trained network
    """
    hyperparameters = yaml.load(open(hyper_path, 'r'), Loader=yaml.SafeLoader)
    input_size = np.shape(features['X_test'])[1]
    output_size = 1
    hidden_neurons = hyperparameters['hidden_neurons']
    hidden_layers  = hyperparameters['hidden_layers']
    activation     = hyperparameters['activation']
    model = LSTM(input_size, output_size, hidden_neurons, hidden_layers, activation)
    model.load_state_dict(torch.load(model_path))
    print("Loaded saved model from file: ", model_path)
    return model

def predict(signals, norm_params, model):
    """ 
    Predicts the output of the model for a given set of input signals

    Args:
        signals (numpy.ndarray): Input signals to predict output for
        norm_params (tuple): Normalization parameters for input and output signals
        model: trained network

    Returns:
        prediction (numpy.ndarray): Predicted output

    """
    X_norm_params, y_norm_params = norm_params
    signals = create_features(signals)
    X_norm, _ = preprocess_data.z_score_normalize(signals[:,:-1], X_norm_params)
    prediction_norm, _ = test_model(model, X_norm)
    prediction = preprocess_data.z_score_normalize_inverse(prediction_norm, y_norm_params)
    return prediction