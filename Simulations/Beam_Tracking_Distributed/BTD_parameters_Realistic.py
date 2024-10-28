"""
This file contains the parameters for the simulations with Beam Tracking Dsistrbuted millimiter wave beam tracking.

"""
import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732

##################################
### Initial System Parameters  ###
##################################

M = 6         # RAU number 
Dt = 0.0001   # The BTD paper time interval is 0.0001

m = 2 * M    # state dimension
n = M        # observation dimension

####### Create state transition matrix 

F_CV_0 = torch.tensor([[1,  Dt],
                          [0,  1,]])

F_CA_0 = torch.tensor([[1, Dt, 0.5*Dt**2],
                          [0,  1,     Dt   ],
                          [0,  0,     1    ]])

F_CV = F_CV_0 
for i in range(M-1) :
    F_CV = torch.block_diag(F_CV , F_CV_0)

F_CA = F_CA_0
for i in range(M-1) :
    F_CA = torch.block_diag(F_CA , F_CA_0)
    
F = F_CV


####### Create state selection matrix for observation model

H_0 = torch.tensor([1.0,0.0])
H = H_0
for i in range(M-1) :
    H = torch.block_diag(H , H_0)



##################################
### Initial state and variance ###
##################################
angular_v_init = 1.0
m1x_0 = torch.tensor([[0.0],[angular_v_init]]*M) # initial state mean
m2x_0 = torch.eye(m,m)    # initial state Covariance , denoted P in the paper 
m2x_0 = m2x_0*1e-5


###############################################
### process noise Q and observation noise R ###
###############################################

####### Process noise 

# Diagonal white noise structure 
Q_structure = torch.eye(m)


### Constant Velocity Models
# Discrete White noise structure (Direct discrete white noise model)

Q_structure_CV_DWN_0 = torch.tensor( [[(Dt**4)/4, (Dt**3)/2],
                                    [(Dt**3)/2,  Dt**2   ]])

Q_structure_CV_DWN = Q_structure_CV_DWN_0
for i in range(M-1) :
    Q_structure_CV_DWN = torch.block_diag(Q_structure_CV_DWN , Q_structure_CV_DWN_0)

# Discretized Continuous White noise structure

Q_structure_CV_CWN_0 = torch.tensor([[1/3*Dt**3, 1/2*Dt**2],
                                   [1/2*Dt**2,     Dt   ]])

Q_structure_CV_CWN = Q_structure_CV_CWN_0
for i in range(M-1) :
    Q_structure_CV_CWN = torch.block_diag(Q_structure_CV_CWN , Q_structure_CV_CWN_0)


####### Observation noise 

R_structure = torch.eye(n)


    

##################################
### State evolution function f ###
##################################
# f full information

def f (x, F = F) :
    return torch.matmul(F , x)

def batched_f (x, F = F) :
    F_batched = F.to(x.device).view(1,F.shape[0],F.shape[1]).expand(x.shape[0],-1,-1)
    return torch.bmm(F_batched, x)

#####################################
### State observation function h  ###
#####################################
# h full information

# h should produce a measurement vector using only the Spatial Angle componenets of x .
# In the paper , they are located at the odd-indexed positions , but python indexing starts from 0 .
# So , we pick the even-indexed elements instead , using block diagonal matrix H = diag(H_0,...,H_0) 
# with H_0 = [1.0,0.0] as produced above 

def h(x,H=H) :
    y = torch.matmul(H,x)
    return torch.tan(y/2)

  
def batched_h(x,H=H) :
    H_batched = H.to(x.device).view(1,H.shape[0],H.shape[1]).expand(x.shape[0],-1,-1)
    y = torch.bmm(H_batched, x)
    return torch.tan(y/2)
    