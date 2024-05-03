## Setup:
if running on local machine, otherwise, necessary installations are performed in Jupyter notebook.
<br>
*Note: run all these from root of the project*

Installation (must use python 3.9):
```
python3 -m venv venv
```

```
pip install -r requirements.txt
```

And you're done!

## Usage:
The notebooks deploying code functions for the project have the naming convention of 'Case', number and type of algorithm. The underlying files being run by these notebooks are either the general function files used for pre and post processing or algorithm specific function files. The notebooks represent an example case of these algorithms for real world data and are not the only or necessarily the best methods for performing predictions with the data set. Please see the Jupyter notebooks for more details on implementation of the developed algorithms.

### Jupyter Notebooks and their corresponding python function files 
- General functions
    * preprocess_data.py: Functions for filtering, extracting and saving data for efficient use with Jupyter notebooks or any python code.
    * ShortreedModel.py: Creates a prediction using the empirically developed model developed in Shortreed et al 2001.
    * post_process.py: Functions used to develop visualization of the results from any of the algorithms compared with the Shortreed model

- Case 0. PreprocessingVisualization.ipynb: View different options that have gone into designing the filters for the function in preprocess.py 

- Case 1. LinearRegression.ipynb: Linear regression with parametric study for nonlinear terms 
    * LR_functions: Functions specifically taylored to the linear regresion algorithm implementation for this dataset

- Case 2. DNN.ipynb: Deep neural network regression with hyperparameter random grid search
    * DNN_functions: Functions  developed for the implementation of a deep neural network with mini-batch gradient descent using an Adam optimizer and randomized grid search to select 'best' hyperparameters.
    
 ### Data Folders 
- ET_Runs: holds the data for the testing runs. The folders lying within contain text files for runs related to a specific protocol. Note protocols vary widely and no general assumptions can be made about the range of data within a given protocol. The csv files correspond to the protocols and contain information from test logs.

- preprocessed_data: If this folder does not exist, it will be created by executing the necessary function with preprocess_data.py. Otherwise, it will hold a python variable defined as a dictionary (similar to a structure for MATLAB) of all signals available from the ET_runs data set.
