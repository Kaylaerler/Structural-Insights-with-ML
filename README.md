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
The primary notebooks that will be used in the project all start with Case and the number for the case. All other files are either underlying function files for the notebooks. Note that the complexity of the preprocessing and data curation has posed a more significant challenge than building the models themselves and is considered integral to accuracy in the results. 


### Jupyter Notebooks and their corresponding python function files 
- Case 0. PreprocessingVisualization.ipynb: This notebook allows the user to view the different options that have gone into designing the filters for the function in preprocess.py. 

- Case 1. LinearRegression.ipynb: Current functioning linear regression file 
    * preprocess.py: Functions for filtering and extracting data
    * ShortreedModel.py: Original empirical model
    * post_process.py: Functions for outputting plots and demonstrating model performance

- Case 2. DNN.ipynb: Deep neural network regression
    * preprocess.py: Functions for filtering and extracting data
    * ShortreedModel.py: Original empirical model
    * post_process.py: Functions for outputting plots and demonstrating model performance
    

    
 ### Data Folders
- ET_Runs: holds the data for the testing runs. The folders lying within contain text files for runs related to a specific protocol. Note protocols vary widely and no general assumptions can be made about the range of data within a given protocol. The csv files correspond to the protocols and contain information from test logs.

- preprocessed_data: If this folder does not exist, MLCase1LinearRegression.ipynb will create it and use it to store the preprocessed and split data
