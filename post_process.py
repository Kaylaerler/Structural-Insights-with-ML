import numpy as np
import preprocess_data
import matplotlib.pyplot as plt
import sklearn as skl

# modules created for this project
import DNN_functions
import ShortreedModel
import LR_functions


def test_scores(model, model_type, saved_data_directory = 'preprocessed_data', norm_params = None, selected_feature_indices=None, alpha = 0):
    """
    test_scores is a function that takes in a model and returns the MSE and MAE for each run in the testing data set.
    These testing scores are normalized on a per run basis to understand the equivalent performance of the model accross
    runs with varrying amplitudes. 

    Inputs:
    ----------
    model : torch.nn.Module or tuple
    
    """         
    empirical_output = {}
    model_output= {}
    signals_library = preprocess_data.load_data_set(saved_data_directory)
    for i in range(len(signals_library)):     # cycle through all data folders
        signals = signals_library[i]['data']
        empirical_prediction = ShortreedModel.predict(signals)
        if model_type == "LR":
            # use model paramaters that are in denormalized units to predict
            bias, params = model
            X, _, _ = LR_functions.create_features(alpha, signals)
            X_exp = X[:,selected_feature_indices]
            params = params.reshape(len(params), 1)
            prediction = np.matmul(X_exp, params) + bias
        else:
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
    ----------
    MSE : list
        Mean Squared Error for each run
    Vx_Median : list
        Median Horizontal Velocity for each run
    Ax_Median : list
        Median Horizontal Acceleration for each run
    MSE_empirical : list
        Mean Squared Error for each run for the empirical model
    model_name : str
        Name of the model being tested

    Returns:
    ----------
    plot of MSE and MAE per run with median velocity and acceleration normalized
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

    run_number = np.arange(0,len(MSE))
    norm_Vx, _ = preprocess_data.z_score_normalize(np.array(Vx_median))
    norm_Ax, _ = preprocess_data.z_score_normalize(np.array(Ax_median))
    f, ax   = plt.subplots(nrows=1,ncols=1, figsize=(20,5))
    axs = ax
    axs.scatter(run_number, norm_Ax, marker = 'x', color = 'k', label = 'Normalized Median Acceleration')
    axs.scatter(run_number, norm_Vx, marker = 'o', color = 'r', label = 'Normalized Median Velocity')
    axs.plot(run_number, MSE, label = model_name, color = 'k')
    axs.plot(run_number, MSE_empirical, label = 'Empirical Model', color = 'g')
    axs.set_xlabel('Run Number')
    axs.set_ylabel('MSE')
    axs.set_title('Model Per Run Scoring', fontsize = 10)
    axs.legend()
    plt.setp(ax, xticks = run_number[0:-1:2], xticklabels = run_number[0:-1:2])
    axs.grid(which = 'both', axis = 'both')
    axs.text(0.51, 0.5, model_name+' Average MSE= %.3f'%(np.mean(MSE)), horizontalalignment='center', verticalalignment='center', transform=axs.transAxes, color = 'k', fontsize = 10)
    axs.text(0.5, 0.45, 'Empirical Average MSE= %.3f'%(np.mean(MSE_empirical)), horizontalalignment='center', verticalalignment='center', transform=axs.transAxes, color = 'g', fontsize = 10)

def plot_prediction(empirical_prediction, model_prediction, signals, model_name, run_name):
    """
    plot_prediction is a function that takes in the time, signals, empirical prediction, and model prediction and plots
    the target and predictions for a given run.
    
    Args:
    ----------
    time : np.array
        time vector
    signals : np.array
        signals from the run
    empirical_prediction : np.array
        prediction from the empirical model
    prediction : np.array
        prediction from the model
    run_name : str
        name of the run
    model_name : str
        name of the model being tested
        
    Returns:
    ----------
    plot of target and predictions for a given run
    """
    # target and prediction plot
    _, ax = plt.subplots(nrows=1,ncols=2, figsize=(15,5))
    axs = ax[0]
    axs.plot(signals[:,1], signals[:,-1], label = 'Target')
    axs.plot(signals[:,1], empirical_prediction, label = 'Empirical Model')
    axs.plot(signals[:,1], model_prediction, label = model_name)
    axs.set_ylabel('Force [tons]', fontsize = 14)
    axs.set_xlabel('Displacement [in]', fontsize = 14)
    axs.set_title('Example Prediction:' + run_name, fontsize=14)
    axs.legend()
    axs.grid()

    axs = ax[1]
    axs.plot(signals[:,0], signals[:,-1], label = 'Target')
    axs.plot(signals[:,0], empirical_prediction, label = 'Empirical Model')
    axs.plot(signals[:,0], model_prediction, label = model_name)
    axs.set_ylabel('Force [tons]', fontsize = 14)
    axs.set_xlabel('Timne [s]', fontsize = 14)
    axs.set_title('Example Prediction:' + run_name, fontsize=14)
    axs.legend()
    axs.grid()

def plot_signals(test_number, preprocessed_data_directory = 'preprocessed_data'):
    """
    """
    plot_signals = [1,2,3,4,5,8]
    signals_library =  preprocess_data.load_data_set(preprocessed_data_directory)
    _, ax   = plt.subplots(nrows=len(plot_signals),ncols=1, figsize=(10,10))
    signals = signals_library[test_number]['data']
    labels  = signals_library[test_number]['labels']
    i = 0
    for sig in plot_signals:
        axs = ax[i]
        axs.plot(signals[:,0],signals[:,sig], label = labels[sig]['name'])
        axs.set_xlabel(labels[0]['name']+labels[0]['units'])
        axs.set_ylabel(labels[sig]['units'])
        axs.legend()
        axs.grid()
        i += 1

def lgmodel_denorm_params(bias, model_params, norm_params, y_norm_params, selected_feature_indices):
    """
    Returns the model coefficients in real units if the model has been z-score normalized. 

    (y - μy)/σy = a0(x0 - μ0) / σ0 + a1(x1 - μ1) / σ1 ... + b
        where 
            μ = mean of x
            σ = standard deviation of x
            a = parameter model solution of normalized features
            b = bias of normalized feature solution

    y = X.*a*σy./σ - σy [a.*μ/σ - b] + μy

    
    therefore:
    b_real = - σy [a.*μ/σ - b] + μy
    a_real = σy*a./σ 
    *note 
        .* signifies dot product
        *  signifies element-wise multiplication of a vector
        when no subscript exists in formula, vector notation is implied

    Inputs
    ----------
    bias : np.array
        b
    model_params : np.array
        a
    norm_params : tuple
        μ & σ
        The mean and std deviation used when normalizing

    Returns
    -------
        Tuple:
            bias in tons
            model parameters based on real coefficient units
    """
    u, sd = norm_params
    u = u[selected_feature_indices]
    sd = sd[selected_feature_indices]
    uy, sdy = y_norm_params
    bias_real = -sdy*(np.dot(model_params/sd,u)-bias)+uy
    a_real = model_params*sdy/sd
    return bias_real, a_real