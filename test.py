import numpy as np
import preprocess_data
import pickle

saved_data_directory = 'signals_data.pkl'
signals = preprocess_data.ETdata_to_numpy(saved_data_directory)

data = pickle.load(saved_data_directory)
print(data[0])

testing_set, train_val_set, training_set, val_set, norm_params, index = preprocess_data.test_train_split(data, index = None, normalize = True, split = 0.3)