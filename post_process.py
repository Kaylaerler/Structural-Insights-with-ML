import numpy as np
import os 
import preprocess_data
import matplotlib.pyplot as plt
import sklearn as skl
import torch
import ShortreedModel


def test_scores(regression_type, model, model_params, signals_type = 1, norm_params = None, selected_feature_indices=None, alpha = 0):
    MSE = []
    MSE_empirical = []
    MAE = []
    MAE_empirical = []
    Ax_Median = []
    Vx_Median = [] 
    data = {}
    Mpath      = os.getcwd()             # get current working directory                                             
    Mdata_fold = '/ET_Runs/'             # folder containing data 
    data_folds = ['dataW/','dataK/','dataH2/','dataH1/','data3/','data2/','data1/'] # subfolders containing data from testing routines
    k          = 0                       # instantiate index for examples
    for i in range(len(data_folds)):     # cycle through all data folders
        dpath      = Mpath + Mdata_fold + data_folds[i] # initial path to get directory list of files contained within 
        dir_list   = os.listdir(dpath)   # files within testing routine
        for j in range(len(dir_list)):   # cycle through all files in the folders 
            if "txt" in dir_list[j]:
                file         = dir_list[j] # select desired file
                f_ext        = dpath+file  # create full extension for chosen file
                signals, _   = preprocess_data.signal_preprocessing(f_ext,dpath, file, signals_type = signals_type) # obtain preprocessed signals from the file extension needed for linear regression
                features_SRMD, dt = preprocess_data.signal_preprocessing(f_ext, dpath, file, signals_type = 2)
                friction          = ShortreedModel.mach_frict(features_SRMD[:,1], features_SRMD[:,5], features_SRMD[:,4], 114)
                empirical_prediction = ShortreedModel.Horizontal_Forces(friction, features_SRMD[:,1], features_SRMD[:,2], 114)

                if regression_type == "sklearn":
                    bias, params = model_params
                    X_exp_norm, _, _, x_norm_params, _ = preprocess_data.create_features(alpha, signals)
                    X_exp = preprocess_data.z_score_normalize_inverse(X_exp_norm,x_norm_params)
                    X_exp = X_exp[:,selected_feature_indices]
                    params = params.reshape(len(params), 1)
                    prediction = np.matmul(X_exp, params) + bias
                else:
                    u, sd = norm_params
                    if hasattr(u, "__len__"):
                        y_norm_params = (u[-1],sd[-1])
                    else:
                        y_norm_params = (u,sd)
                    signals_norm, _ = preprocess_data.z_score_normalize(signals, norm_params)
                    prediction_norm = model(torch.autograd.Variable(torch.from_numpy(signals_norm[:,0:-1])).to(torch.float32)).cpu().data.numpy()
                    prediction = preprocess_data.z_score_normalize_inverse(prediction_norm,y_norm_params)
                
                data[k] = prediction, empirical_prediction, signals, data_folds[i]+file
                k = k+1 

                # scoring metrics

                # z-score normalize on a per run basis to have better comparison amongst runs for fit
                target, t_norm_params = preprocess_data.z_score_normalize(signals[:,-1].reshape(-1,1))
                predict_run_norm, _   = preprocess_data.z_score_normalize(prediction, t_norm_params)
                emp_run_norm, _       = preprocess_data.z_score_normalize(empirical_prediction, t_norm_params)

                MSE.append(skl.metrics.mean_squared_error(target, predict_run_norm))
                MSE_empirical.append(skl.metrics.mean_squared_error(target, emp_run_norm))
                MAE.append(skl.metrics.median_absolute_error(target, predict_run_norm))
                MAE_empirical.append(skl.metrics.median_absolute_error(target, emp_run_norm))
                Ax_Median.append(np.median(np.abs(signals[:,2])))
                Vx_Median.append(np.median(np.abs(signals[:,1])))

    return MSE, MAE, MSE_empirical, MAE_empirical, data, Ax_Median, Vx_Median

def plot_signals(signals, time):
    f, ax   = plt.subplots(nrows=signals.shape[1],ncols=1, figsize=(10,10))
    labels  = ['Horizontal Displacement', 'Horizontal Velocity', 'Horizontal Acceleration', 'Outrigger Normal Force', 'Horizontal Velocity Signum', 'Horizontal Force']
    ylabels = ['Disp [in]', 'Velocity [in/s]', 'Acceleration [g]', 'Force [tons]', 'Velocity sign []', 'Force [tons]']

    for i in range(np.shape(signals)[1]):
        axs = ax[i]
        axs.plot(time,signals[:,i],label = labels[i])
        axs.set_xlabel('Time [s]')
        axs.set_ylabel(ylabels[i])
        axs.legend()
        axs.grid()

def plot_prediction(time, signals,empirical_prediction,prediction, run_name, model_name):
    # target and prediction plot
    plt.figure(figsize=(6,4))
    plt.plot(time, signals[:,-1], label = 'Target')
    plt.plot(time, empirical_prediction, label = 'Empirical Model')
    plt.plot(time, prediction, label = model_name)
    plt.ylabel('Force [tons]', fontsize = 14)
    plt.xlabel('Time [sec]', fontsize = 14)
    plt.title('Example Prediction:' + run_name, fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.grid()

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
   
def model_per_run_scoring(MSE, Vx_Median, Ax_Median, MSE_empirical):
    run_number = np.arange(0,len(MSE))
    Vx_median = np.array(Vx_Median)
    Ax_median = np.array(Ax_Median)
    norm_Vx, _ = preprocess_data.z_score_normalize(Vx_median)
    norm_Ax, _ = preprocess_data.z_score_normalize(Ax_median)
    f, ax   = plt.subplots(nrows=1,ncols=1, figsize=(20,5))
    axs = ax
    axs.scatter(run_number, norm_Ax, marker = 'x', color = 'k', label = 'Normalized Median Acceleration')
    axs.scatter(run_number, norm_Vx, marker = 'o', color = 'r', label = 'Normalized Median Velocity')
    axs.plot(run_number, MSE, label = 'Linear Regression Model', color = 'k')
    axs.plot(run_number, MSE_empirical, label = 'Empirical Model', color = 'g')
    axs.set_xlabel('Run Number')
    axs.set_ylabel('MSE')
    axs.set_title('Model Per Run Scoring', fontsize = 10)
    axs.legend()
    plt.setp(ax, xticks = run_number[0:-1:2], xticklabels = run_number[0:-1:2])
    axs.grid(which = 'both', axis = 'both')
    axs.text(0.51, 0.5, 'LR Average MSE= %.3f'%(np.mean(MSE)), horizontalalignment='center', verticalalignment='center', transform=axs.transAxes, color = 'k', fontsize = 10)
    axs.text(0.5, 0.45, 'Empirical Average MSE= %.3f'%(np.mean(MSE_empirical)), horizontalalignment='center', verticalalignment='center', transform=axs.transAxes, color = 'g', fontsize = 10)

