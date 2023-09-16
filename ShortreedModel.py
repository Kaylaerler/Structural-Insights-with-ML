"""
Library Name: ShortreedModel
Author: Kayla Erler
Version: 1.0.0
Date: 10/15/2023
License: GNU

Description:
-----------
ShortreedModel is a collection of useful functions to create the prediction for horizontal forces of the SRMD machine 
                based on the empirical prediction developed by Shortreed et al. (2001).

                [Shortreed et al. (2001)](https://royalsocietypublishing.org/doi/10.1098/rsta.2001.0875) 
                "Characterization and testing of the Caltrans Seismic Response Modification Device Test System". 
                Phil. #Trans. R. Soc. A.359: 1829–1850
Usage:
------
import ShortreedModel 

# Example Usage
result = Shortreed.some_function(arg1, arg2)

"""

# Open source modules available in python
import numpy as np


# default is set to accept imperial units only
def mach_frict(vel, OutriggerMeanForce, CompForce, ActualTareWt = 114, LiftPressure = 1300, dof = 0): 
    """mach_frict is a function translated directly from the Caltrans SRMD postprocessing protocol (original code in MatLab). It calculates the correction for friction forces only. 
    
    Args: 
        vel                - velocity array for the specified degree of freedom. The default is in the X-Direction
        OutriggerMeanForce - array of the mean over the 4 outriggers force readings
        CompForce          - array of total compression force mearsured over time defined as the Compression Force fbk channel
        LiftPressure       - Default set to 1300, but varies from test to test (not all information was available, so it is held constant)
        dof                - specifies the dof for the test run 0:5 denoting X, Y, Z, Roll, Pitch and Yaw (default 0=X/horizontal)
        ActualTareWt       - Effective weight of the empty table. (Default value set to that in the original protocol)
        
    Returns:
        FrictionValue - Scalar value used to compute friction forces
    """
    
    OCompZMean = np.mean(CompForce)
    maxv=70
    v = np.arange(0,maxv,0.1)
    vone=np.ones((len(v),1))
    max_vel = np.max(vel)
    nv= int(np.floor((max_vel/maxv)*(len(v)-1))+1)

    ka = np.zeros((6,1))
    b = np.zeros((len(v),6))
    c = np.zeros((len(v),6))
    d = np.zeros((len(v),6))


    # COEFFICIENT VALUES in X, Y, Z, Roll, Pitch and Yaw dofs:
    # using 2nd empirical model, developed by JSS 7/10/00
    ka1= [0.2,     0.5,     0, 0, 0, 0]

    b3 = [0.080,      0.080,     0, 0, 0, 0]
    b4 = [0.720,      0.500,     0, 0, 0, 0]
    b6 = [0.045,      0.050,     0, 0, 0, 0]
    b7 = [0.850,      0.900,     0, 0, 0, 0]
    b8 = [0.0675,     0.080,     0, 0, 0, 0]
    b9 = [0.00063,    0.00045,   0, 0, 0, 0]

    c3 = [0.0007,     0.0007,      0, 0, 0, 0]
    c4 = [0.760,      0.760,       0, 0, 0, 0]
    c6 = [0.00075,    0.00075,     0, 0, 0, 0]
    c7 = [0.880,      0.880,       0, 0, 0, 0]
    c8 = [0.0008,     0.0009,      0, 0, 0, 0]
    c9 = [0.0000114,  0.0000114,   0, 0, 0, 0]

    d1 = [14,         14,          0, 0, 0, 0]   # about 14 times the friction betwn outriggers & verts


    for i in range(6):
        for n in range(len(v)):
            ka[i]= vone[n]*ka1[i];
            b[n,i]= ((b4[i]**v[n])*b3[i] - ((b7[i]**v[n])*b6[i]) + (b9[i]*v[n])+ (vone[n]*b8[i])) 
            c[n,i]= ((c4[i]**v[n])*c3[i] - ((c7[i]**v[n])*c6[i]) + (c9[i]*v[n])+ (vone[n]*c8[i])) 
            d[n,i]= c[n,i]*d1[i]

    LiftArea = 4.9*2  # after adding 4 extra outriggers on Jan 1,2002
    LiftN    = 8
    FL = (LiftPressure*LiftArea*LiftN)*0.5/1000 # from "lb" to "ton" , 1000(lb)=0.5(ton)
    FO = np.mean(OutriggerMeanForce) *4         #  In case of cal. with four outrigger value
    FV =       FO + ActualTareWt + OCompZMean 
    FrictionValue = ka[dof]+b[nv,dof]*FL + c[nv,dof]*FV + d[nv,dof]*FO
    FrictionValue = np.floor(FrictionValue*100)/100

    return FrictionValue
    
def Horizontal_Forces(FrictionValue, velocity, acceleration, weight):
    """" Function takes the inputs of the model and returns the predicted forces based on 
    friction and inertia forces. Friction calculated using mach_friction, weight comes from log files or 
    user defined.
    
    Args:
        FrictionValue (float): value of friction calculated from mach_friction
        velocity (array): array of velocity from log file
        acceleration (array): array of acceleration from log file
        weight (float): weight of the table
        
    Returns:
        horizontal_force (array): array of predicted horizontal forces
    """
    friction_force = FrictionValue*np.sign(velocity)
    inertia = weight*acceleration
    horizontal_force = friction_force+inertia
    return horizontal_force

def predict(signals, weight = 114):
    """
    Function takes the input of the signals array and returns the predicted horizontal forces based on
    the empirical model developed by Shortreed et al. (2001). The default weight is set to the weight of the
    SRMD table.

    Args:
        signals (array): array of signals from the log file
        weight (float, optional): weight of the table. Defaults to 114.

    Returns:
        horizontal_force_prediction (array): array of predicted horizontal forces

    """
    vel = signals[:,2]
    acceleration = signals[:,3]
    OutriggerMeanForce = signals[:,6]
    CompForce = signals[:,7]
    FrictionValue = mach_frict(vel, OutriggerMeanForce, CompForce, ActualTareWt = weight)
    horizontal_force_prediction = Horizontal_Forces(FrictionValue, vel, acceleration, weight)
    return horizontal_force_prediction

