import numpy as np
import preprocess_data

# Open source libraries available in python
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score 
import importlib
importlib.reload(preprocess_data) #used to reload file if it has been modified during notebook use
from sklearn.feature_selection import SelectKBest, f_regression

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
    X_norm, norm_params = preprocess_data.z_score_normalize(X, norm_params = norm_params)
    Y_norm, y_norm_params = preprocess_data.z_score_normalize(Y, norm_params = y_norm_params)
    return X_norm, Y_norm, feature_names, norm_params, y_norm_params

def train(alpha, signals_train, signals_test, number_features = 0):
    """"The sklearn implementation of Linear Regression fits a model with the input features and outputs 
        the training and testing metrics. Additional feature selection can be implemented if the user 
        believes not all features are significant to the model or that they are overly correlated. 
        Feature selection with sklearn will try all combinations of features to arrive at the best MSE
        using the number of features selected. 
    
    Inputs:
        alpha - exponential term (eq 3)
        signals_train - displacement, velocity, acceleration, force columns (training set)
        signals_test  - displacement, velocity, acceleration, force columns (testing set)
        number_features - if feature selection is desired, the number of features for the model to select the 
                          best fit from can be inputed here. (default 0 which will ignore feature selection)
        
    Outputs:
        MSE_test   - testing MSE for fitted model
        R2_test    - testing R^2 for fitted model
        MSE_train  - training MSE for fitted model
        R2_train   - training R^2 for fitted model
        simple_reg - fitted model
        selected_feature_indices - indices for selected features if feature selection is desired
                                   default reserves all model features
        """
    X_train, y_train, feature_names, x_norm_params, y_norm_params = preprocess_data.create_features(alpha,signals_train)
    X_test,  y_test,  _, _, _       = create_features(alpha,signals_test, x_norm_params, y_norm_params)
    selected_feature_indices        = np.arange(np.shape(X_train)[1]) # default all features

    if not(number_features == 0):
        # Feature selection using SelectKBest and f_regression
        selector = SelectKBest(f_regression, k=number_features)
        X_train  = selector.fit_transform(X_train, y_train)
        X_test   = selector.transform(X_test)

        # Get the selected feature indices
        selected_feature_indices = selector.get_support(indices=True)

        # Get the names of the selected features
        selected_feature_names = [feature_names[i] for i in selected_feature_indices]

        # Print the selected feature names
        print("Selected features:", selected_feature_names)

    #Linear Regression Model 
    simple_reg   = LinearRegression(fit_intercept = True, n_jobs = -1, positive = True)    # initiate model object
    simple_reg.fit(X_train,y_train)             # fit the estimator
    Y_test_pred  = simple_reg.predict(X_test)   # model prediction on testing data
    Y_train_pred=simple_reg.predict(X_train)    # model prediction on training data
    
    # Obtain testing and training metrics
    MSE_test = mean_squared_error(y_test,Y_test_pred)
    R2_test  = r2_score(y_test,Y_test_pred)
    MSE_train = mean_squared_error(y_train,Y_train_pred)
    R2_train  = r2_score(y_train,Y_train_pred)
    
    return MSE_test, R2_test, MSE_train, R2_train, simple_reg, selected_feature_indices, x_norm_params, y_norm_params 


def fit_exponential(num_pts, signals_train, signals_test):
    # parametric analysis to fit exponential terms
    alpha_range = np.linspace(0.2,1,num = num_pts)
    MSE = np.empty(np.shape(alpha_range))
    for i in range(0, len(alpha_range)):
        MSE[i], _, _, _, _, _, _, _ = train(alpha_range[i],signals_train, signals_test)

    alpha_optimal = alpha_range[MSE == np.min(MSE)]
    alpha_optimal = alpha_optimal[0]
    print(f'Optimal parameters found with all features: alpha = {alpha_optimal}')
    print('Providing an MSE = %.3f on testing data'%(np.min(MSE)))
    print('-----------------------------------')

    return alpha_optimal