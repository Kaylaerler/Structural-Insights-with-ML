# References and Licensing
"""
Code functions written by Kayla Erler updated 5/22/2023

Functions defined for data from the seismic response modification device (SRMD) empty table runs. More details on the testing facility can be found at: https://se.ucsd.edu/facilities/laboratory-listing/srmd

Refferences for the papers used to facilitate developement of models:
[Shortreed et al. (2001)](https://royalsocietypublishing.org/doi/10.1098/rsta.2001.0875) "Characterization and testing of the Caltrans Seismic Response Modification Device Test System". Phil. #Trans. R. Soc. A.359: 1829–1850

[Ozcelick et al. (2008)](http://jaguar.ucsd.edu/publications/refereed_journals/Ozcelik_Luco_Conte_Trombetti_Restrepo_EESD_2008.pdf) "Experimental Characterization, modeling and identification of the NEES-UCSD shake table mechanical systetm". Eathquake Engineering and Structural Dynamics, vol. 37, pp. 243-264, 2008

Citation and Licensing:
Kayla Erler 

[Rathje et al. (2017)](https://doi.org/10.1061/(ASCE)NH.1527-6996.0000246) "DesignSafe: New Cyberinfrastructure for Natural Hazards Engineering". ASCE: Natural Hazards Review / Volume 18 Issue 3 - August 2017

This software is distributed under the [GNU General Public License](https://www.gnu.org/licenses/gpl-3.0.html).
"""

environment = 'local' # 'designsafe' or 'local'



import numpy as np
import os
if environment == 'local':
    import ShortreedModel
else:
    import gdrive.MyDrive.structural.ShortreedModel as ShortreedModel
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy import signal
from scipy.fft import fft
from scipy.signal import find_peaks

def signal_preprocessing(directory, dpath, file, signals_type = 1):
    
    ####   extract data  
    data_open    = open(directory) # open data and store data
    data         = np.loadtxt(directory,skiprows=70,unpack=True) # convert to workable data
    n_channels,c = data.shape # extract # of channels and length of data 
    skiprow      = 6+n_channels*2
    data         = np.loadtxt(directory,skiprows=skiprow,unpack=True) # convert to workable data

    # get data from test logs
    test_ext   = dpath[0:-1]+'.csv' # create extension for accompanying csv containing test specifications
    run        = file[0:-4]         # identify column for csv file
    test_data  = pd.read_csv(test_ext,index_col=0)
    test_data.index = test_data.index.str.strip()  

    ####   create channel label strings: ' CHAN_# '
    info_head=[]
    for i in range(0,n_channels):
        info_head.append("CHAN_"+str(i+1))#create the names of the channels
        
    ####   import data as a dataframe to extract channel descriptions and units
    with open(directory) as file:
        lines = [next(file) for x in range(skiprow)]
    data_open.close()
    chan_label = []
    chan_units = []
    r = np.arange(6,len(lines),2)    
    for i in r:
        a = lines[i][25:-1]
        a = a.strip()
        b = lines[i+1][25:-1]
        b = b.strip()
        chan_label.append(a)
        chan_units.append(b)
    
    df = pd.DataFrame(list(zip(chan_label, chan_units)),index = info_head, columns =['Description', 'Units'])
    
    dt             = lines[4][25:]
    dt             = float(dt.strip())
    d_ind_ref      = np.where(df["Description"] == "Long Reference")[0]
    d_ind_feedback = np.where(df["Description"] == "Long Feedback")[0]
    f_ind          = np.where(df["Description"] == "Long Force fbk")[0]
    Out_NE_ind     = np.where(df["Description"] == "O-NE Force fbk")[0]
    Out_SE_ind     = np.where(df["Description"] == "O-SE Force fbk")[0]
    Out_NW_ind     = np.where(df["Description"] == "O-NW Force fbk")[0]
    Out_SW_ind     = np.where(df["Description"] == "O-SW Force fbk")[0]
    V_NE_ind       = np.where(df["Description"] == "V-NE Force fbk")[0]
    V_SE_ind       = np.where(df["Description"] == "V-SE Force fbk")[0]
    V_NW_ind       = np.where(df["Description"] == "V-NW Force fbk")[0]
    V_SW_ind       = np.where(df["Description"] == "V-SW Force fbk")[0]
    Compression_ind= np.where(df["Description"] == "Compression Force fbk")[0]
    first_point = 0
    last_point = -1
    disp_ref       = np.array(data[:][d_ind_ref])[0,first_point:last_point]
    disp_feedback  = np.array(data[:][d_ind_feedback])[0,first_point:last_point]
    force          = np.array(data[:][f_ind])[0,first_point:last_point]
    Out_NE         = np.expand_dims(np.array(data[:][Out_NE_ind])[0,first_point:last_point],1)
    Out_SE         = np.expand_dims(np.array(data[:][Out_SE_ind])[0,first_point:last_point],1)
    Out_NW         = np.expand_dims(np.array(data[:][Out_NW_ind])[0,first_point:last_point],1)
    Out_SW         = np.expand_dims(np.array(data[:][Out_SW_ind])[0,first_point:last_point],1)
    V_NE           = np.expand_dims(np.array(data[:][V_NE_ind])[0,first_point:last_point],1)
    V_SE           = np.expand_dims(np.array(data[:][V_SE_ind])[0,first_point:last_point],1)
    V_NW           = np.expand_dims(np.array(data[:][V_NW_ind])[0,first_point:last_point],1)
    V_SW           = np.expand_dims(np.array(data[:][V_SW_ind])[0,first_point:last_point],1)
    Compression    = np.array(data[:][Compression_ind])[0,first_point:last_point] 

    Fs          = 1/dt         # sampling frequency [Hz]
    disp_FFT    = abs(np.fft.fftshift(fft(disp_ref)))/len(disp_ref)*2     # centered fft displacement feedback signal
    f           = np.linspace(-Fs/2,Fs/2,len(disp_FFT)+1)
    f           = f[0:len(disp_FFT)]
    # get peak of FFT
    indices     = find_peaks(disp_FFT)[0]
    max_peak    = max(max(np.where(disp_FFT==np.max(disp_FFT[indices]))))

    #to avoid returning 0 for the cutoff frequency
    z = 0
    while abs(f[max_peak]) < 0.0005 or z < 100:  
        z+= 1
        indices = np.delete(indices,np.where(indices==max_peak))
        max_peak= max(max(np.where(disp_FFT==np.max(disp_FFT[indices]))))
    
    f_peak      = np.abs(f[max_peak])
    fc          = f_peak*30     # cutoff frequency   [Hz]
    
    # specify minimum for regular tests and triangular that have a tendency to provide low frequency for filtering
    fc_min = 0.1
    if 'shape' in test_data[run]:   # shape of test run (cosine, sine or triangle)
        shape = test_data[run]['shape']
        if shape == 'triangle':
            fc_min = 1 
    if fc < fc_min:
        fc = fc_min

    # apply sgolay filters to derive velocity and acceleration
    window_length   = 65
    velocity        = savgol_filter(disp_feedback, window_length = window_length, polyorder = 2, deriv = 1, delta = dt) #[in/s]
    g               = 386.4  #[in/s^2]
    ax_filt_sgolay  = savgol_filter(disp_feedback, window_length = window_length, polyorder = 2, deriv = 2, delta = dt)/g #[g]
    
    # Use firwin with a Kaiser window to create a lowpass FIR filter.
    wp              = fc/(Fs/2)    # Normalized passband edge frequency w.r.t. Nyquist rate
    b_filt          = signal.firwin(numtaps=200, cutoff=wp, window='hamming', pass_zero="lowpass")
    ax_filt         = signal.filtfilt(b_filt, 1, ax_filt_sgolay)
    force_filt      = signal.filtfilt(b_filt,1, force)

    # Remove begining and end of signals as they are prone to spurious regions 
    pts = len(disp_feedback)
    mid_point = round(pts/2)
    first_point     = 65
    last_point      = -65
    displacement    = np.expand_dims(disp_feedback[first_point:last_point],1)
    velocity        = np.expand_dims(velocity[first_point:last_point],1)
    acceleration    = np.expand_dims(ax_filt[first_point:last_point],1)
    force           = np.expand_dims(force[first_point:last_point],1)
    Compression     = np.expand_dims(Compression[first_point:last_point],1)
    OutriggerForces = np.concatenate((Out_NE,Out_SE,Out_NW,Out_SW),axis = 1)
    OutriggerMeanForce = np.mean(OutriggerForces[first_point:last_point,:], axis = 1, keepdims = True)
    Act_Total_Force = np.sum((V_NE,V_SE,V_NW,V_SW),axis = 0)
    vx = velocity
    min_velocity = max(0.001*np.max(vx),0.002)
    vx[np.abs(vx)<min_velocity] = 0
    Act_Force       = Act_Total_Force[first_point:last_point]
    OutriggerForce  = np.expand_dims(np.sum(np.abs(OutriggerForces[first_point:last_point,:]), axis = 1),1)

    # vector truncation
    centralized_points = 2 # number of points from each side of the vectors to be removed for truncation
    disp_matrix = np.zeros((len(displacement),centralized_points*2+1))
    vel_matrix  = np.zeros((len(displacement),centralized_points+1))
    for i in range(centralized_points*2+1):
        if i == 0:
            disp_matrix[:, 0] = displacement.flatten()
        else:
            disp_matrix[i:, i] = displacement[0:-i].flatten()
    for i in range(centralized_points+1):
        vel_matrix[(centralized_points-1+i):,i] = velocity[0:(-centralized_points+1-i)].flatten()
    
    if signals_type == 1:
        # used for linear regression models
        features = np.concatenate((displacement, velocity, acceleration, Act_Force, OutriggerForce, force),axis = 1)
    elif signals_type == 2:
        # used for DNN model
        features = np.concatenate((displacement,velocity, acceleration, np.sign(vx), force),axis = 1)
    elif signals_type ==3:
        # used for PINN model
        features = np.concatenate((disp_matrix,velocity,acceleration,Act_Force,OutriggerForce,force),axis = 1)
        features = features[centralized_points:-centralized_points, :]
    elif signals_type == 4:
        # used for empirical model
        features = np.concatenate((displacement,velocity,acceleration,force,Compression,OutriggerMeanForce),axis = 1)
    else:
        # used for multiple displacements and multiple velocity in order to predict acceleration
        features = np.concatenate((disp_matrix,vel_matrix,acceleration,Act_Force,OutriggerForce,force),axis = 1)
        features = features[centralized_points:-centralized_points, :]
    return features, dt

def z_score_normalize(X, norm_params = None, treat_disp_different = False):
    """
    Performs z-score normalization on X. 

    f(x) = (x - μ) / σ
        where 
            μ = mean of x
            σ = standard deviation of x

    Parameters
    ----------
    X : np.array
        The data to z-score normalize
    u (optional) : np.array
        The mean to use when normalizing
    sd (optional) : np.array
        The standard deviation to use when normalizing (if zero standard deviation replace with 1 to return a zero z-score)

    Returns
    -------
        Tuple:
            Transformed dataset with mean 0 and stdev 1
            Computed statistics (mean and stdev) for the dataset to undo z-scoring.
    """
    if treat_disp_different:
        disp = X[:,0]
        sd_disp = np.std(disp)
        disp_norm = disp/sd_disp

    if norm_params is None:
        u = np.mean(a=X, axis=0)
        sd = np.std(a=X, axis=0)
    else:
        u, sd = norm_params
        
    X_norm = (X - u) / sd
    if treat_disp_different:
        X_norm[:,0] = disp_norm
        u[0] = 0
    return X_norm, (u, sd)

def z_score_normalize_inverse(X, norm_params):
    """
    Performs inverse z-score normalization on X. 

    Parameters
    ----------
    X : np.array
        The data to inverse z-score normalize
    u : np.array
        The mean used when normalizing
    sd : np.array
        The standard deviation used when normalizing

    Returns
    -------
        Tuple:
            Inverse of transformed dataset with mean 0 and stdev 1
    """
    u, sd = norm_params
    return X*sd+u

def test_train_split(signals_library, index = None, normalize = True, split = 0.3):
    """signals contains data for displacement, velocity, acceleration and horizontal force in the respective column number
    It is brought in as a library and returned as (optionally) randomized 2D matrices for the desired set """

    #reordered_signals_library, sorted_indices = reorder_library(signals_library, index)
    reordered_signals_library = signals_library
    signals = reordered_signals_library[0]
    for i in range(0,len(reordered_signals_library)-1):
        signals = np.append(signals, reordered_signals_library[i], axis = 0)

    if not hasattr(index, "__len__"):
        index = np.arange(0,np.shape(signals)[0])
        np.random.shuffle(index)
        
    signals_out = signals[index]
        
    if normalize:
        signals_out, norm_params = z_score_normalize(signals_out, norm_params = None, treat_disp_different =False)
    else:
        u = 0
        sd = 1
        norm_params = u, sd

    data_points      = np.shape(signals_out)[0]
    testing_points   = round(data_points*split)
    train_val_points = data_points - testing_points
    training_points  = round(train_val_points*(1-split))

    if np.ndim(signals_out) < 2:
        testing_set      = signals_out[0:testing_points]
        train_val_set    = signals_out[testing_points:]
        training_set     = train_val_set[0:training_points]
        val_set          = train_val_set[training_points:]
    else:
        testing_set      = signals_out[0:testing_points, :]
        train_val_set    = signals_out[testing_points:,:]
        training_set     = train_val_set[0:training_points,:]
        val_set          = train_val_set[training_points:,:]
    return testing_set, train_val_set, training_set, val_set, norm_params, index

def create_features(alpha, signals, norm_params = None, y_norm_params = None):
    """ function takes the exponential terms of the model and creates the input features from the input signals and stacks the data into a numpy array, 
    X for the input features and Y for the target. 

    Inputs:
        alpha   - exponential term from equation 3
        signals - displacement, velocity, acceleration, force columns  

    Outputs:
        X        - 2D feature matrix with rows corresponding to instance and columns corresponding to feature
        Y        - 1D target array with number of rows equal to that of X 
    """

    ux = np.array([signals[:,0]])
    vx = np.array([signals[:,1]])
    vx[np.abs(vx)<0.001] = 0 # set small values of velocity to zero
    ax  = np.array([signals[:,2]])
    Act = np.array([signals[:,3]])
    Out = np.array([signals[:,4]])
    Y  = np.array(signals[:,5])
    
    a1 = ax
    a2 = np.abs(vx)**(alpha)*np.sign(vx)
    a3 = Out
    a4 = Act
    a5 = np.sign(vx)
    a6 = ux

    feature_names = ['Inertia','Nonlinear_Friction','Outrigger_Friction','Actuator_Friction', 'Constant Friction', 'Spring Force']
    X = np.concatenate((a1,a2,a3,a4,a5,a6),axis = 0).T
    X_norm, norm_params = z_score_normalize(X, norm_params = norm_params)
    Y_norm, y_norm_params = z_score_normalize(Y, norm_params = y_norm_params)
    return X_norm, Y_norm, feature_names, norm_params, y_norm_params

def load_data_set(norm_params_directory, model_type, data_stored = True, store_data = True, normalize_data_set = True):
    """ function utilizes the preprocess_data.py file for filtering, normalizing, and seperating data functions.
    The Shortreed model file is called to define predicted forces using the previously developed model for SRMD
    post processing. All files are either saved or loaded to storage space using a numpy file type. These file types are binary
    and highly efficient for purposes in which the preprocessed data is read with python code.

    Inputs:
        data_stored                - flag to designate whether the previously preprocessed data should be loaded (default True)
                                     if False, or the data has not previously been stored, this function will pull data from
                                     the ET_Run folder, preprocess data, store to files and return it for use

        store_data                 - flag to allow the user to preprocess data without storing

        normalize_data_set          - flag to designate whether data normalization with z-scoring is desired (default True)

        preprocessed_data_directory - name of folder to save preprocessed data sets to

    Outputs:
        signals_test      - numpy 2D array containing disp, vel, acc, and force data for testing (default 30% of full set)
        signals_train_val - numpy 2D array containing disp, vel, acc, and force data for training and validation (default 70% of full set)
        signals_train     - numpy 2D array containing disp, vel, acc, and force data for training (default 70% of train_val set)
        signals_val       - numpy 2D array containing disp, vel, acc, and force data for validation (default 30% of train_val set)
        norm_params       - tuple containing mean and std needed to inverse normalize
        force_SRMD_test   - numpy 1D array containing predicted forces using Shortreed empirical method
    """
    preprocessed_data_directory = 'preprocessed_data_' + model_type         
    if model_type == 'LR':
        signals_type = 1
    elif model_type == 'DNN':
        signals_type = 2
    elif model_type == 'PINN':
        signals_type = 3
    else:
        signals_type = 5    
                               
    if os.path.exists(preprocessed_data_directory) and data_stored:
        print("Loading Stored Data")
        signals_test      = np.load(preprocessed_data_directory+'/signals_test.npy')
        signals_train_val = np.load(preprocessed_data_directory+'/signals_train_val.npy')
        signals_train     = np.load(preprocessed_data_directory+'/signals_train.npy')
        signals_val       = np.load(preprocessed_data_directory+'/signals_val.npy')
        norm_params       = np.load(preprocessed_data_directory+'/norm_params.npy')
        force_SRMD_test   = np.load(preprocessed_data_directory+'/force_SRMD_test.npy')
    else:
        signals    = {}                      # initialize feature library
        force_SRMD = {}                      # initialize SRMD empirical equation predictions
        if environment == 'local':
            Mpath      = os.getcwd()
        else:
            Mpath = "/content/gdrive/MyDrive/structural"
        Mdata_fold = '/ET_Runs/'             # folder containing data
        data_folds = ['dataW/','dataK/','dataH2/','dataH1/','data3/','data2/','data1/'] # subfolders containing data from testing routines
        k          = 0                       # instantiate index for examples
        weight     = 114                     # weight defined per prescribed process in original post-processing procedure
        for i in range(len(data_folds)):     # cycle through all data folders
            dpath      = Mpath + Mdata_fold + data_folds[i] # initial path to get directory list of files contained within
            dir_list   = os.listdir(dpath)   # files within testing routine
            for j in range(len(dir_list)):   # cycle through all files in the folders
                if "txt" in dir_list[j]:
                    file            = dir_list[j] # select desired file
                    f_ext           = dpath+file  # create full extension for chosen file
                    signals[k], _   = signal_preprocessing(f_ext,dpath, file, signals_type = signals_type) # obtain preprocessed signals from the file extension needed for linear regression
                    signals_SRMD, _ = signal_preprocessing(f_ext,dpath, file, signals_type = 4)  # obtain features needed for Shortreed model
                    friction        = ShortreedModel.mach_frict(signals_SRMD[:,1], signals_SRMD[:,5], signals_SRMD[:,4],weight) # define Shortreed model friction force amplitude
                    force_SRMD[k]   = ShortreedModel.Horizontal_Forces(friction, signals_SRMD[:,1], signals_SRMD[:,2], weight) # defined Shortreed model predicted horizontal force
                    k = k+1

        # create data set split, randomize data runs and transform libraries into 2D matrices
        signals_test, signals_train_val, signals_train, signals_val, norm_params, index = test_train_split(signals, index = None, normalize = normalize_data_set)
        force_SRMD_test, _, _, _, _, _                                                  = test_train_split(force_SRMD, index = index, normalize = False)

        print(store_data)
    if store_data:
        #save variables to file to avoid computation time related to reloading
        if not(os.path.exists(preprocessed_data_directory)):
            os.mkdir(preprocessed_data_directory)

        sep = '/'
        stripped = norm_params_directory.split(sep, 1)[0]
        if not(os.path.exists(stripped)):
            os.mkdir(stripped)
                
        np.save(preprocessed_data_directory+'/signals_test.npy',signals_test)
        np.save(preprocessed_data_directory+'/signals_train_val.npy',signals_train_val)
        np.save(preprocessed_data_directory+'/signals_train.npy',signals_train)
        np.save(preprocessed_data_directory+'/signals_val.npy',signals_val)
        np.save(preprocessed_data_directory+'/norm_params.npy',norm_params)
        np.save(preprocessed_data_directory+'/force_SRMD_test.npy',force_SRMD_test)
        np.save(norm_params_directory, norm_params)

    return signals_test, signals_train_val, signals_train, signals_val, norm_params, force_SRMD_test