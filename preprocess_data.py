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

def signal_preprocessing(directory, dpath, file):
    
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
    signal_labels = ["Long Reference", "Long Feedback", 
                     "O-NE Force fbk", "O-SE Force fbk", "O-NW Force fbk", "O-SW Force fbk", 
                     "V-NE Force fbk", "V-SE Force fbk", "V-NW Force fbk", "V-SW Force fbk",
                     "Compression Force fbk", "Long Force fbk"]
    all_signals = np.array([])
    for i in range(len(signal_labels)):
        new_column = data[:][np.where(df["Description"] == signal_labels[i])[0]]
            # Add the new column to the existing array
        if all_signals.size == 0:
            all_signals = new_column   # If it's the first column, assign it directly
        else:
            all_signals = np.vstack((all_signals, new_column))


    Fs          = 1/dt         # sampling frequency [Hz]
    disp_FFT    = abs(np.fft.fftshift(fft(all_signals[0,:])))/len(all_signals[0,:])*2     # centered fft displacement feedback signal
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
    velocity        = savgol_filter(all_signals[1,:], window_length = window_length, polyorder = 2, deriv = 1, delta = dt) #[in/s]
    min_velocity = max(0.001*np.max(velocity),0.002)
    velocity[np.abs(velocity)<min_velocity] = 0
    
    # Use firwin with a Kaiser window to create a lowpass FIR filter.
    g               = 386.4  #[in/s^2]
    ax_filt_sgolay  = savgol_filter(all_signals[1,:], window_length = window_length, polyorder = 2, deriv = 2, delta = dt)/g #[g]
    wp              = fc/(Fs/2)    # Normalized passband edge frequency w.r.t. Nyquist rate
    b_filt          = signal.firwin(numtaps=200, cutoff=wp, window='hamming', pass_zero="lowpass")
    acceleration    = signal.filtfilt(b_filt, 1, ax_filt_sgolay)

    # Concatenate signals desired for models
    OutriggerMeanForce  = np.mean(all_signals[2:6], axis = 0)
    OutriggerTotalForce = np.sum(np.abs(all_signals[2:6]), axis = 0)
    ActuatorForce       = np.sum(all_signals[7:11], axis = 0)
    time = np.arange(0,len(ActuatorForce))*dt

    #   0         1           2               3             4                      
    # time, displacement, velocity, accceleration, Outrigger total force, 
    #          5                   6                 7                 8
    # Actuator force, Outrigger mean force,  compression force, horizontal force
    signals = np.vstack((time, all_signals[1,:], velocity.T, acceleration.T, 
                         OutriggerTotalForce.T, ActuatorForce, OutriggerMeanForce.T, 
                         all_signals[-2], all_signals[-1]))
    
    labels = {}
    names  = ['Time', 'Displacement', 'Velocity', 'Acceleration', 
             'Outrigger Total Force', 'Actuator Force', 'Outrigger Mean Force', 
             'Compression Force', 'Horizontal Force']
    units = [' [s]', ' [in]', ' [in/s]',' [g]', ' [tons]', ' [tons]',' [tons]', ' [tons]',' [tons]']
    for i in range(signals.shape[0]):
        labels[i] = {'name': names[i], 'units': units[i]}
    return signals.T, labels

def ETdata_to_dictionary(saved_data_directory, save_results = True):
    """ 
    """                        
    signals_dictionary = {} # initialize signals dictionary
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
                file       = dir_list[j] # select desired file
                f_ext      = dpath+file  # create full extension for chosen file
                # obtain preprocessed signals from the file extension needed for linear regression
                signals, labels = signal_preprocessing(f_ext, dpath, file)
                signals_dictionary[k] = {'data' : signals, 'test' : data_folds[i]+file, 'labels': labels} 
                k = k+1    
    signal_file = saved_data_directory + '/signals_data.pkl'
    if save_results:
        with open(signal_file, 'wb') as fp:
            pickle.dump(signals_dictionary, fp)
            print('dictionary saved successfully to file:', saved_data_directory)
    return signals_dictionary

def dictionary_to_numpy(signals_dictionary):
    """
    """
    signals_numpy = signals_dictionary[0]['data']
    for i in range(1,len(signals_dictionary)):
        signals_numpy = np.append(signals_numpy, signals_dictionary[i]['data'], axis = 0)
    return signals_numpy

def load_data_set(preprocessed_data_directory, load = True):
    """ 
    """   
    if not os.path.exists(preprocessed_data_directory):
        os.mkdir(preprocessed_data_directory)
    if os.path.exists(preprocessed_data_directory +'/signals_data.pkl') and load:
        print("Loading Stored Data")
        signals_dictionary = np.load(preprocessed_data_directory+'/signals_data.pkl', allow_pickle = True)
    else:
        print("Extracting Data from Individual Folders")
        signals_dictionary = ETdata_to_dictionary(preprocessed_data_directory)
    return signals_dictionary

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