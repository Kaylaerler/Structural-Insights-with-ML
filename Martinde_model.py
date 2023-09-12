# 1 Dimensional Flow with Area Variation
# Inverse problem : 1 parameter  
import torch 
import pandas as pd 
import matplotlib 
import matplotlib.pyplot as plt 
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'font.size': 13
})
writer = SummaryWriter('C:/Users/marti/Desktop/MUMAC/TFM/1 Dim Flow with Area Variation/Inverse1/logs')    
class PINN_1DFlowA(torch.nn.Module):
    def __init__(self,NL):
        
        # Inheritance from torch.nn.Module initialization
        super(PINN_1DFlowA,self).__init__()
        
        # Store the NN structure 
        self.NL=NL
        
        # Initilize layers
        for i in range(0,len(NL)-2):
            setattr(self, f'layer{i}', torch.nn.Linear(NL[i],NL[i+1]))
            # setattr(self,f'batchnorm{i}', torch.nn.BatchNorm1d(NL[i+1]))
        self.layer_y1 = torch.nn.Linear(self.NL[-2], 1)
        self.layer_y2 = torch.nn.Linear(self.NL[-2], 1)
        self.layer_y3 = torch.nn.Linear(self.NL[-2], 1)
        # Initialize activation function 
        self.activation_function=torch.nn.Sigmoid()
        
        # Some parameters for training 
        self.lr=0.01
        self.change_lr=7000 
        self.max_iter=21000
        
        # Parameters of the equation
        self.gamma=1.4
        self.A0=1 # Initial value of the area 
        self.AL=3 # Final value of the area 
        self.k=torch.tensor([1.0],requires_grad=True) # Curvature 
        
        # Boundary conditions (Dirichlet for now)
        self.rho_inf=1
        self.u_inf=1
        self.p_inf=1
        self.U0=torch.tensor([self.rho_inf,self.u_inf,self.p_inf]) 
        self.UL=torch.tensor([0.2554]) # Velocity in the extrem 
        
        # Domain parameters 
        self.Ix=[0.0,4.0]
        self.batch_size=50 # batch_size
        
    # Process to obtain the NN output
    def forward(self,x):
        for i in range(0,len(self.NL)-2):
            x = getattr(self, f'layer{i}')(x) 
            # x = getattr(self,f'batchnorm{i}')(x)     
            x = self.activation_function(x)
        # We add a different last layer for each output 
        y1=self.layer_y1(x) # Associated to rho
        y2=self.layer_y2(x) # Associated to u 
        y3=self.layer_y3(x) # Associated to p
        return y1 , y2, y3  
    
    # Distance function 
    def Smooth_distance(self,X):
        return X
        
    # Boundary function 
    def Boundary_extension(self,X):
        return self.U0[0] , self.U0[1] , self.U0[2]
    
    # Area variation 
    def Area_variation(self,X):
        # The Area follows a logistic equation
        return self.A0+(self.AL-self.A0)/(1+torch.exp(-self.k*(X-self.Ix[1]*0.5-self.Ix[0]*0.5)))

   # Function that computes the fluid variables and its derivatives
    def Fluid_Variables_Derivatives(self,X):
        
        # Compute the output of the model 
        y1,y2,y3=self.forward(X)
        
        # We compute the distance and boundary functions 
        G1, G2, G3 = self.Boundary_extension(X)
        D=self.Smooth_distance(X)
        
        # We compute flow variables 
        rho=G1+D*y1
        u=G2+D*y2
        p=G3+D*y3
        
        # We compute the derivatives of the flow variables 
        rho_x=torch.autograd.grad(rho,X,torch.ones(self.batch_size,1),
                                retain_graph=True,create_graph=True)[0]
        u_x=torch.autograd.grad(u,X,torch.ones(self.batch_size,1),
                                retain_graph=True,create_graph=True)[0]
        p_x=torch.autograd.grad(p,X,torch.ones(self.batch_size,1),
                                retain_graph=True,create_graph=True)[0]
                
        return rho , u , p , rho_x , u_x , p_x 
    # Loss function, depends on the EDO to solve
    def loss_function(self,k=None,verbose=False):
        # We define the points where we are evaluating the loss
        X=self.Ix[0]+(self.Ix[1]-self.Ix[0])*torch.rand((self.batch_size,1))
        X.requires_grad = True
        
        # Fluid flow ad derivatives
        rho , u , p , rho_x , u_x , p_x  = self.Fluid_Variables_Derivatives(X) 
        
        # Area variation
        A = self.Area_variation(X)
        A_x = torch.autograd.grad(A,X,torch.ones(self.batch_size,1),
                                retain_graph=True,create_graph=True)[0]
        
        # Internal Energy of the system 
        E = 0.5*rho*torch.pow(u,2)+p/(self.gamma-1)
        E_x = 0.5*rho_x*torch.pow(u,2)+rho*u*u_x+p_x/(self.gamma-1)

        # Flows 
        F1x = A*(rho_x*u+rho*u_x)
        F2x = A*(rho_x*torch.pow(u,2)+2*rho*u*u_x + p_x)
        F3x = A*(u_x*(E+p)+u*(E_x+p_x))
        
        # Source terms
        S1 = A_x*rho*u
        S2 = A_x*rho*torch.pow(u,2)
        S3 = A_x*u*(E+p)
        
        # Conservation of mass
        L1 = 1/self.batch_size*sum((F1x+S1)**2)
        writer.add_scalar('L1', L1, global_step=k)
        
        # Conservation of momentum 
        L2 = 1/self.batch_size*sum((F2x+S2)**2)
        writer.add_scalar('L2', L2, global_step=k)
        
        # Conservation of energy 
        L3 = 1/self.batch_size*sum((F3x+S3)**2)
        writer.add_scalar('L3', L3, global_step=k)
        
        # Force the velocity on the extrem 
        _,y2,_=self.forward(torch.tensor([self.Ix[1]]))
        _,G2,_=self.Boundary_extension(torch.tensor([self.Ix[1]]))
        D=self.Smooth_distance(torch.tensor([self.Ix[1]]))
        uL=G2+y2*D
        LU=abs(uL-self.UL)
        writer.add_scalar('LU', LU, global_step=k)
        
        if verbose:
            print(L1,L2,L3,LU)
            print(uL) 
            
        return L1+L2+L3+LU
        
    # Train the NN with some dataset
    def fit(self): 
        # We define the optimizer (Adam or SGD)
        optimizer=torch.optim.Adam(list(self.parameters())+[self.k],lr=self.lr)
        scheduler=StepLR(optimizer,step_size=self.change_lr,gamma=0.1) 
        for k in range(0,self.max_iter):
            # We reset the gradient of parameters
            optimizer.zero_grad()
            loss=self.loss_function(k)
            # Calculate backpropagation 
            loss.backward() 
            writer.add_scalar('Global Loss', loss.item(), global_step=k)
            # We do one step of optimization 
            optimizer.step() 
            scheduler.step() 
            if (k/(self.max_iter/20)==k//(self.max_iter/20)):
                print(f'Iteration : {k}. Loss function : {loss.item()}.')            
    
    # Obtain the results after training 
    def Solution(self):
        # Grid 
        x_coordinates=torch.linspace(self.Ix[0],self.Ix[1],100)
        X=x_coordinates.reshape(100,1)
        
        # NN output 
        y1,y2,y3=self.forward(X)
        
        # We compute the distance and boundary functions 
        G1, G2, G3 = self.Boundary_extension(X)
        D=self.Smooth_distance(X)

        # We compute flow variables 
        rho=G1+D*y1
        u=G2+D*y2
        p=G3+D*y3
        
        return X , rho , u , p
    
# Instanciate the class 
N=PINN_1DFlowA([1,20,20,20,3])

# We fit the model with the training data 
N.train() # training mode 
N.fit() # we actually perfom the fitting of the model    
N.k=1.8666
# Results 
folder='Inverse1/'
N.eval()
[X,rho,u,p]=N.Solution()
plt.figure()
plt.xlabel(r'$x$')
with torch.no_grad():
    plt.plot(X,u,label=r'$u$')
    plt.plot(X,rho,label=r'$\rho$')
    plt.plot(X,p,label=r'$p$')
    plt.plot(X,N.Area_variation(X),label=r'$A(x,\lambda=1.8666)$')
    plt.plot(X,rho*u*N.Area_variation(X),label=r'$\dot{m}$')
plt.legend()
plt.tight_layout()
plt.savefig(folder+'Variables distribution inverse.pgf',dpi=1000)
plt.close()