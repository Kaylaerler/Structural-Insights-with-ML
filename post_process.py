"""
Module Name: post_process
Author: Kayla Erler
Version: 1.0.0
Date: 10/15/2023
License: GNU

Description:
-----------
post_process is a collection of useful functions and utilities for tasks related to visualization after running machine learning algorithms to evaluate model fit.

Usage:
------
import post_process 

# Example Usage
result = post_process.some_function(arg1, arg2)

"""
# modules created for this project
import ShortreedModel
import preprocess_data

# Open source modules available in python
import numpy as np
import matplotlib.pyplot as plt
import sklearn as skl

def test_scores(model, model_type, saved_data_directory = 'preprocessed_data', norm_params = None, selected_feature_indices=None, alpha = 0):
    """
    test_scores is a function that takes in a model and returns the MSE and MAE for each run in the testing data set.
    These testing scores are normalized on a per run basis to understand the equivalent performance of the model accross
    runs with varrying amplitudes. 

    Args:
        model : (tuple or torch object) contains the model parameters and bias for linear regression or the torch object for the DNN
        model_type : (str) either "LR" or "DNN" to indicate which model is being tested
        saved_data_directory : (str) directory where the preprocessed data is saved
        norm_params : (tuple) contains the mean and standard deviation used to normalize the data
        selected_feature_indices : (list) contains the indices of the features that were selected by the feature selection algorithm
        alpha : (float) exponential term from equation 3

    Returns:
        empirical_output : (dict) contains the empirical prediction and MSE and MAE for each run
        model_output : (dict) contains the model prediction and MSE and MAE for each run
    """         
    empirical_output = {}
    model_output= {}
    _, _, _, signals_test, _ = preprocess_data.load_data_set(saved_data_directory)
    for i in range(len(signals_test)):     # cycle through all data folders
        signals = signals_test[i]['data']
        empirical_prediction = ShortreedModel.predict(signals)
        if model_type == "LR":
            import LR_functions
            # use model paramaters that are in denormalized units to predict
            bias, params = model
            X, _, _ = LR_functions.create_features(alpha, signals)
            X_exp = X[:,selected_feature_indices]
            params = params.reshape(len(params), 1)
            prediction = np.matmul(X_exp, params) + bias
        else:
            import DNN_functions
            prediction = DNN_functions.predict(signals, norm_params, model)

        # z-score normalize on a per run basis to have better comparison amongst runs for fit
        target, t_norm_params = preprocess_data.z_score_normalize(signals[:,-1].reshape(-1,1))
        predict_run_norm, _   = preprocess_data.z_score_normalize(prediction, t_norm_params)
        emp_run_norm, _       = preprocess_data.z_score_normalize(empirical_prediction, t_norm_params)
        
        # scoring metrics
        MSE           = skl.metrics.mean_squared_error(target, predict_run_norm)
        MSE_empirical = skl.metrics.mean_squared_error(target, emp_run_norm)
        MAE           = skl.metrics.median_absolute_error(target, predict_run_norm)
        MAE_empirical = skl.metrics.median_absolute_error(target, emp_run_norm)
        Ax_Median     = np.median(np.abs(signals[:,3]))
        Vx_Median     = np.median(np.abs(signals[:,2]))
        model_output[i] = {"prediction": prediction, 
                              'MSE': MSE,
                              'MAE': MAE,
                              'Ax_median': Ax_Median,
                              'Vx_median': Vx_Median}

        empirical_output[i] = {"prediction": empirical_prediction,
                               "MSE": MSE_empirical,
                               "MAE": MAE_empirical}

    return empirical_output, model_output

def model_per_run_scoring(empirical_output, model_output, model_name):
    """
    model_per_run_scoring is a function that takes in MSE and MAE for each run in the testing data set and plots these scores
    with median velocity and acceleration normalized so that the relative scale of each is indicated per run.
        
    Args:
        empirical_output : (dict) contains the empirical prediction and MSE and MAE for each run
        model_output : (dict) contains the model prediction and MSE and MAE for each run
        model_name : (str) name of the model being tested

    Returns:
        plot of MSE and MAE for each run with median velocity and acceleration normalized
    """
    MSE = []
    Vx_median = []
    Ax_median = []
    MSE_empirical = []
    for i in range(len(empirical_output)):
        MSE.append(model_output[i]['MSE'])
        Vx_median.append(model_output[i]['Vx_median'])
        Ax_median.append(model_output[i]['Ax_median'])
        MSE_empirical.append(empirical_output[i]['MSE']) 


    main_font = 17
    run_number = np.arange(0,len(MSE))
    norm_Vx, _ = preprocess_data.z_score_normalize(np.array(Vx_median))
    norm_Ax, _ = preprocess_data.z_score_normalize(np.array(Ax_median))
    f, ax   = plt.subplots(nrows=1,ncols=1, figsize=(20,5))
    axs = ax
    axs.scatter(run_number, norm_Ax, marker = 'x', color = 'k', label = 'Normalized Median Acceleration')
    axs.scatter(run_number, norm_Vx, marker = 'o', color = 'r', label = 'Normalized Median Velocity')
    axs.plot(run_number, MSE, label = model_name, color = 'k')
    axs.plot(run_number, MSE_empirical, label = 'Empirical Model', color = 'g')
    axs.set_xlabel('Run Number', fontsize = main_font)
    axs.set_ylabel('MSE', fontsize = main_font)
    #axs.set_title('Model Per Run Scoring', fontsize = title_font)
    axs.legend(fontsize = main_font)
    plt.setp(ax, xticks = run_number[0:87:2], xticklabels = run_number[0:87:2])
    plt.xticks(fontsize = main_font)
    plt.yticks(fontsize = main_font)
    axs.set_ylim([-1,5])
    axs.set_xlim([0,87])
    axs.grid(which = 'both', axis = 'both')
    axs.text(0.40 , 0.6, model_name+' Average MSE= %.3f'%(np.mean(MSE)), horizontalalignment='center', verticalalignment='center', transform=axs.transAxes, color = 'k', fontsize = main_font)
    axs.text(0.40, 0.45, 'Empirical Average MSE= %.3f'%(np.mean(MSE_empirical)), horizontalalignment='center', verticalalignment='center', transform=axs.transAxes, color = 'g', fontsize = main_font)

def plot_prediction(empirical_prediction, model_prediction, signals, model_name, run_name, metric = False):
    """
    plot_prediction is a function that takes in the time, signals, empirical prediction, and model prediction and plots
    the target and predictions for a given run.
    
    Args:
        empirical_prediction : (np.array) empirical prediction for the run
        model_prediction : (np.array) model prediction for the run
        signals : (np.array) time and signals for the run
        model_name : (str) name of the model being tested
        run_name : (str) name of the run being tested

    Returns:
        plot of target and predictions for a given run
    """
    if metric:
        disp = 25.4*(signals[:,1] - np.mean(signals[:,1]))
        target = 8.896*signals[:,-1]
        raw_target = 8.896*signals[:,-3]
        empirical = 8.896*empirical_prediction
        model = 8.896*model_prediction
    else:
        disp = signals[:,1] - np.mean(signals[:,1])
        target = signals[:,-1]
        raw_target = signals[:,-3]
        empirical = empirical_prediction
        model = model_prediction
    # target and prediction plot
    target_color = 'b'
    raw_color = 'c'
    empirical_color = 'g'
    model_color = 'k'
    _, ax = plt.subplots(nrows=1,ncols=2, figsize=(15,5))
    axs = ax[0]
    axs.plot(disp, target, label = 'Target', color = target_color)
    axs.plot(disp, empirical, label = 'Empirical Model', color = empirical_color)
    axs.plot(disp, model, label = model_name, color = model_color)
    if metric:
        axs.set_ylabel('Force [kN]', fontsize = 14)
        axs.set_xlabel('Displacement [mm]', fontsize = 14)
    else:
        axs.set_ylabel('Force [tons]', fontsize = 14)
        axs.set_xlabel('Displacement [in]', fontsize = 14)
    
    axs.set_title('Example Prediction:' + run_name, fontsize=14)
    axs.legend()
    axs.grid()

    axs = ax[1]
    axs.plot(signals[:,0], raw_target, label = 'Raw Data', color = raw_color)
    axs.plot(signals[:,0], target, label = 'Target', color = target_color)
    axs.plot(signals[:,0], empirical, label = 'Empirical Model', color = empirical_color)
    axs.plot(signals[:,0], model, label = model_name, color = model_color)
    if metric:
        axs.set_ylabel('Force [kN]', fontsize = 14)
    else:
        axs.set_ylabel('Force [tons]', fontsize = 14)
    axs.set_xlabel('Time [s]', fontsize = 14)
    axs.set_title('Example Prediction:' + run_name, fontsize=14)
    axs.legend()
    axs.grid()

def plot_signals(test_number, preprocessed_data_directory = 'preprocessed_data', metric = False):
    """ 
    plot_signals is a function that takes in the test number and plots the signals for that test.

    Args:
        test_number : (int) number of the test to plot
        preprocessed_data_directory : (str) directory where the preprocessed data is saved
        metric : (bool) if true, the units of the signals will be converted to metric units

    Returns:
        plot of the signals for the given test
    """
    plot_signals = [1,2,3,4,5,9]
    _, _, _, _, signals_dictionary =  preprocess_data.load_data_set(preprocessed_data_directory)
    _, ax   = plt.subplots(nrows=len(plot_signals),ncols=1, figsize=(10,10))
    signals = signals_dictionary[test_number]['data']
    labels  = signals_dictionary[test_number]['labels']
    i = 0
    factor = 1
    for sig in plot_signals:
        if metric:
            if 'in' in labels[sig]['units']:
                ylabel = str(labels[sig]['units']).replace('in','mm')
                factor = 25.4
            elif 'ton' in labels[sig]['units']:
                ylabel = str(labels[sig]['units']).replace('tons','kN ')
                factor = 8.896
            else:
                ylabel = labels[sig]['units']
        else:
            ylabel = labels[sig]['units']
        axs = ax[i]
        if sig == plot_signals[-1]:
            axs.plot(signals[:,0],factor*signals[:,7], label = 'Raw Signal')
        axs.plot(signals[:,0],factor*signals[:,sig], label = labels[sig]['name'])
        axs.set_xlabel(labels[0]['name']+labels[0]['units'])
        axs.set_ylabel(ylabel)
        axs.legend()
        axs.grid()
        i += 1
        