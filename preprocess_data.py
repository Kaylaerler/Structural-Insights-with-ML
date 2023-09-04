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
from scipy.signal import savgol_filter
from scipy import signal
from scipy.fft import fft
from scipy.signal import find_peaks
import pickle

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
    disp_ref       = np.array(data[:][np.where(df["Description"] == "Long Reference")[0]])
    disp_feedback  = np.array(data[:][np.where(df["Description"] == "Long Feedback")[0]])
    force          = np.array(data[:][np.where(df["Description"] == "Long Force fbk")[0]])
    Out_NE         = np.expand_dims(np.array(data[:][np.where(df["Description"] == "O-NE Force fbk")[0]]))
    Out_SE         = np.expand_dims(np.array(data[:][np.where(df["Description"] == "O-SE Force fbk")[0]]))
    Out_NW         = np.expand_dims(np.array(data[:][np.where(df["Description"] == "O-NW Force fbk")[0]]))
    Out_SW         = np.expand_dims(np.array(data[:][np.where(df["Description"] == "O-SW Force fbk")[0]]))
    V_NE           = np.expand_dims(np.array(data[:][np.where(df["Description"] == "V-NE Force fbk")[0]]))
    V_SE           = np.expand_dims(np.array(data[:][np.where(df["Description"] == "V-SE Force fbk")[0]]))
    V_NW           = np.expand_dims(np.array(data[:][np.where(df["Description"] == "V-NW Force fbk")[0]]))
    V_SW           = np.expand_dims(np.array(data[:][np.where(df["Description"] == "V-SW Force fbk")[0]]))
    Compression    = np.array(data[:][np.where(df["Description"] == "Compression Force fbk")[0]])

    Fs          = 1/dt         # sampling frequency [Hz]
    disp_FFT    = abs(np.fft.fftshift(fft(disp_ref)))/len(disp_ref)*2     # centered fft displacement feedback signal
    f           = np.linspace(-Fs/2,Fs/2,len(disp_FFT)+1)
    f           = f[0:len(disp_FFT)]
    indices     = find_peaks(disp_FFT)[0] # get peak of FFT
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

    # Remove begining and end of signals as they are prone to spurious regions 
    first_point     = 65
    last_point      = -65
    displacement    = np.expand_dims(disp_feedback[first_point:last_point],1)
    vx              = np.expand_dims(velocity[first_point:last_point],1)
    acceleration    = np.expand_dims(ax_filt[first_point:last_point],1)
    force           = np.expand_dims(force[first_point:last_point],1)
    Compression     = np.expand_dims(Compression[first_point:last_point],1)
    OutriggerForces = np.concatenate((Out_NE,Out_SE,Out_NW,Out_SW),axis = 1)
    Act_Total_Force = np.mean((V_NE,V_SE,V_NW,V_SW),axis = 0)
    min_velocity = max(0.001*np.max(vx),0.002)
    vx[np.abs(vx)<min_velocity] = 0
    Act_Force       = Act_Total_Force[first_point:last_point]

    features = np.concatenate((displacement, vx, acceleration, Act_Force, OutriggerForces, Compression, force), axis = 1)
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

def library_to_numpy(signals_library):
    """
    """
    reordered_signals_library = signals_library
    signals_numpy = reordered_signals_library[0]
    for i in range(0,len(reordered_signals_library)-1):
        signals_numpy = np.append(signals_numpy, reordered_signals_library[i], axis = 0)
    return signals_numpy

def ETdata_to_library(saved_data_directory):
    """ 
    """                        
    signals    = {} # initialize feature library
    signals_name = {}
    if environment == 'local':
        Mpath      = os.getcwd()
    else:
        Mpath = "/content/gdrive/MyDrive/structural"
    Mdata_fold = '/ET_Runs/'             # folder containing data
    data_folds = ['dataW/','dataK/','dataH2/','dataH1/','data3/','data2/','data1/'] # subfolders containing data from testing routines
    k          = 0                       # instantiate index for examples
    for i in range(len(data_folds)):     # cycle through all data folders
        dpath      = Mpath + Mdata_fold + data_folds[i] # initial path to get directory list of files contained within
        dir_list   = os.listdir(dpath)   # files within testing routine
        for j in range(len(dir_list)):   # cycle through all files in the folders
            if "txt" in dir_list[j]:
                file            = dir_list[j] # select desired file
                f_ext           = dpath+file  # create full extension for chosen file
                signals[k], _   = signal_preprocessing(f_ext, dpath, file) # obtain preprocessed signals from the file extension needed for linear regression
                signals_name[k]  = data_folds[i]+file
                k = k+1
                
    signal_file = saved_data_directory + '/signals_data.pkl'
    signals_name_file = saved_data_directory + '/signals_name.pkl'
    with open(signal_file, 'wb') as fp:
        pickle.dump(signals, fp)
        print('dictionary saved successfully to file:', saved_data_directory)
    with open(signals_name_file, 'wb') as fp:
        pickle.dump(signals_name, fp)
    return signals

def load_data_set(preprocessed_data_directory):
    """ 
    """                                
    if os.path.exists(preprocessed_data_directory):
        print("Loading Stored Data")
        signals = np.load(preprocessed_data_directory)
    else:
        print("Extracting Data from Individual Folders")
        signals = ETdata_to_library(preprocessed_data_directory)
    return signals
