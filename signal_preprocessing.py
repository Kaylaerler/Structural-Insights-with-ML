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