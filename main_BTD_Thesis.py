import torch
from datetime import datetime
from Simulations.Extended_sysmdl import SystemModel
import Simulations.config as config
from Simulations.utils import DataGen
from Simulations.Beam_Tracking_Distributed.BTD_parameters_Realistic_2 \
import  batched_f ,f, batched_h , h,Q_structure_CV_CWN , R_structure, m , n , m1x_0 , m2x_0 , Dt
from Filters.UKF_filterpy_test import UKFTest as filterpyUKFTest
from Pipelines.Pipeline_EKF import Pipeline_EKF
from KNet.KalmanNet_nn import KalmanNetNN
import matplotlib.pyplot as plt
import numpy as np

# Taken from Roger Labbe's book "Kalman and Bayesian Filters in Python"
def plot_residual_limits(Ps, stds=1.):
   """ 
   plots standard deviation given in matrix Ps as a yellow shaded region. 
   One std by default, use stds for a different choice (e.g. stds=3 for 3 standard
   deviations.
   """
   std = np.sqrt(Ps) * stds

   plt.plot(-std, color='k', ls=':', lw=2)
   plt.plot(std, color='k', ls=':', lw=2)
   plt.fill_between(range(len(std)), -std, std,
                  facecolor='#ffff00', alpha=0.3)


################
### Get Time ###
################
today = datetime.today()
now = datetime.now()
strToday = today.strftime("%m.%d.%y")
strNow = now.strftime("%H:%M:%S")
strTime = strToday + "_" + strNow
print("Current Time =", strTime)
path_results = 'KNet/'

####################
### Design Model ###
####################
args = config.general_settings()

### dataset parameters ##################################################
args.N_E = 2000
args.N_CV = 200
args.N_T = 20
# init condition
args.randomInit_train = False
args.randomInit_cv = False
args.randomInit_test = False
if args.randomInit_train or args.randomInit_cv or args.randomInit_test:
   # you can modify initial variance
   args.variance = 1
   args.distribution = 'normal' # 'uniform' or 'normal'
   m2_0 = args.variance * torch.eye(m)
else: 
   # deterministic initial condition
   m2_0 = 0 * torch.eye(m) 
# sequence length
args.T = 100
args.T_test = 100
args.randomLength = False
if args.randomLength:# you can modify T_max and T_min 
   args.T_max = 1000
   args.T_min = 100
   # set T and T_test to T_max for convenience of batch calculation
   args.T = args.T_max 
   args.T_test = args.T_max
else:
   train_lengthMask = None
   cv_lengthMask = None
   test_lengthMask = None
# noise
# process noise intensity
q2 = torch.tensor([5e-1])
# observation noise intensity
r2 = torch.tensor([1e-4]) 
print("1/q2 [dB]: ", 10 * torch.log10(1/q2[0]))
print("1/r2 [dB]: ", 10 * torch.log10(1/r2[0]))


### True model ##################################################
Q = q2 * Q_structure_CV_CWN
R = r2 * R_structure
print("Dt is: ",Dt)
print("Covariance matrix Q is : \n" , Q) # Noise Covariance matrix for a single user 
print("Covariance matrix R is : \n" , R) # Noise Covariance matrix for a single user


### training parameters ##################################################
args.use_cuda = True # use GPU or not
# number of optimization steps
args.n_steps = 600
# batch size 
args.n_batch = 30
# learning rate  
args.lr = 1e-4
# weight decay
args.wd = 1e-2
if args.use_cuda:
   if torch.cuda.is_available():
      device = torch.device('cuda')
      print("Using GPU")
   else:
      raise Exception("No GPU found, please set args.use_cuda = False")
else:
   device = torch.device('cpu')
   print("Using CPU")


### Initialize model parameters #################################

### Generator model #############################################
# Batched versions of the system model. At this context, batched means it operate on "all users at once".
sys_model_gen = SystemModel(batched_f, Q , batched_h, R, args.T, args.T_test,m,n)
sys_model_gen.InitSequence(m1x_0, m2x_0)
sys_model_gen.Init_batched_sequence(sys_model_gen.m1x_0.view(1,sys_model_gen.m,1).expand(args.N_T,-1,-1), \
    sys_model_gen.m2x_0.view(1,sys_model_gen.m,sys_model_gen.m).expand(args.N_T,-1,-1) )        

### Same model for filterpy UKF #############################################
# Filterpy's UKF doesn't operate in the smae batch mode as KalmanNet, so we use the single-user versions for the model functions
# and evaluate UKF at every user separately
sys_model_filterpy_UKF = SystemModel(f, Q , h, R, args.T, args.T_test,m,n)
sys_model_filterpy_UKF.InitSequence(m1x_0, m2x_0)


###################################
### Data Loader (Generate Data) ###
###################################
dataFolderName = 'Simulations/Beam_Tracking_Distributed/'
dataFileName = 'Case3/ThesisData4_paper_parameters_q2-5e-1_r2-1e-4'
print("Start Data Gen")
DataGen(args, sys_model_gen, dataFolderName + dataFileName)
print("Data Load")
if args.randomLength:
   [train_input, train_target, cv_input, cv_target, test_input, test_target,train_init, cv_init, test_init, train_lengthMask,cv_lengthMask,test_lengthMask] = torch.load(dataFolderName + dataFileName, map_location=device)
else:
   [train_input, train_target, cv_input, cv_target, test_input, test_target,_,_,_] = torch.load(dataFolderName + dataFileName, map_location=device)

print("trainset size:",train_target.size())
print("cvset size:",cv_target.size())
print("testset size:",test_target.size())
print("test_input has size : \n" , test_input.size())
print("test target has size : \n" , test_target.size() )



########################################
### Evaluate Unscented Kalman Filter ###
########################################
lamda = torch.tensor([1e-3])
lamda = lamda.to("cuda")
print("lamda is :",lamda)
loss_on_all_states = True
print("######################### Start filterpy UKF with Full Information")
[MSE_UKF_linear_arr, MSE_UKF_linear_avg, MSE_UKF_dB_avg,filterpy_UKF_KG_array ,filterpy_UKF_out , filterpy_UKF_cov_mat,filterpy_UKF_x_prior] =\
  filterpyUKFTest(sys_model_filterpy_UKF,test_input,test_target,Dt,l=lamda,allStates=loss_on_all_states)



####################################
### Train and Evaluate KalmanNet ###
####################################
print("######################### Start KalmanNet with Full Information")
# ## KalmanNet with full info ##########################################################################################
# # Build Neural Network
print("KalmanNet with full model info")
KalmanNet_model = KalmanNetNN()
# KalmanNet_model = torch.load("./Knet/best-model.pt")
KalmanNet_model.NNBuild(sys_model_gen, args)
print("Number of trainable parameters for KalmanNet:",sum(p.numel() for p in KalmanNet_model.parameters() if p.requires_grad))
# Train Neural Network
KalmanNet_Pipeline = Pipeline_EKF(strTime, "KNet", "KalmanNet_BTD")
KalmanNet_Pipeline.setssModel(sys_model_gen)
KalmanNet_Pipeline.setModel(KalmanNet_model)
KalmanNet_Pipeline.setTrainingParams(args)
## select loss computation
mask_option_Train = False
print("Mask on state during Training ? : ", mask_option_Train)
mask_option_Test = False
print("Mask on state during Testing ? : ", mask_option_Test)
# Train and Test KalmanNet
[MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = \
   KalmanNet_Pipeline.NNTrain(sys_model_gen, cv_input, cv_target, train_input, train_target, path_results ,MaskOnState= mask_option_Train)
[MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,knet_out,RunTime] = \
   KalmanNet_Pipeline.NNTest(sys_model_gen, test_input, test_target, path_results , MaskOnState= mask_option_Test)

KalmanNet_Pipeline.save()


#################################### Plots

test_target = test_target.cpu()
test_input = test_input.cpu()
filterpy_UKF_out = filterpy_UKF_out.cpu()
filterpy_UKF_cov_mat = filterpy_UKF_cov_mat.cpu()
filterpy_UKF_x_prior = filterpy_UKF_x_prior.cpu()
filterpy_UKF_KG_array = filterpy_UKF_KG_array.cpu()
knet_out = knet_out.detach().cpu()

for i in range(0,args.N_T) :
      k = 0
      j = 0
      Rau = 1 
      while j < m :
         plt.figure(k)
         plt.title("Spatial Angle estimation (RAU # %i)" %Rau)
         plt.plot(test_target[i,j,:],"g")
         # plt.plot(test_input[i,j//2,:],"k")
         plt.plot(2*torch.arctan(test_input[i,j//2,:]),"k") #plot the measurments in units of spatial angle
         plt.plot(filterpy_UKF_x_prior[i,j,:],"c")
         plt.plot(filterpy_UKF_out[i,j,:],"b")
         plt.plot(knet_out[i,j,:],"r")
         plt.xlabel("Time Slot Index")
         plt.ylabel("Spatial Angle (rad)")
         plt.legend(["Target","Measurement","UKF prior","filterpy UKF","KalmanNet"])
         # plt.legend(["Target","Measurement","UKF prior","filterpy UKF"])
         k+=1
         plt.figure(k)
         plt.title("Spatial Angle estimation residuals (RAU # %i)" %Rau)
         plt.plot(test_target[i,j,:] - filterpy_UKF_out[i,j,:])
         plot_residual_limits(filterpy_UKF_cov_mat[i,j,j,:])
         plt.xlabel("Time Slot Index")
         plt.ylabel("m")
         k+=1
         j+=1 
         plt.figure(k)
         plt.title("Spatial Angular Velocity estimation (RAU # %i)" %Rau )
         plt.plot(test_target[i,j,:],"g")
         plt.plot(filterpy_UKF_out[i,j,:],"b")
         plt.plot(knet_out[i,j,:],"r")
         plt.xlabel("Time Slot Index")
         plt.ylabel("Angular Velocity (rad/s)")
         plt.legend(["Target","filterpy UKF","KalmanNet"])
         # plt.legend(["Target","filterpy UKF"])
         k+=1
         plt.figure(k)
         plt.title("Spatial Angular Velocity estimation residuals (RAU # %i)" %Rau)
         plt.plot(test_target[i,j,:] - filterpy_UKF_out[i,j,:])
         plot_residual_limits(filterpy_UKF_cov_mat[i,j,j,:])
         plt.xlabel("Time Slot Index")
         plt.ylabel("m/s")
         k+=1
         j+=1
         Rau+=1

      plt.show()

      input()

   
