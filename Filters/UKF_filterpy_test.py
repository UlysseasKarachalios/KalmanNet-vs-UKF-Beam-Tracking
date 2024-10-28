import torch.nn as nn
import torch
import time

from filterpy.kalman import UnscentedKalmanFilter, JulierSigmaPoints , MerweScaledSigmaPoints


def UKFTest(SysModel, test_input, test_target, delta_t,l ,modelKnowledge='full', allStates=True, init_cond=None):    
    
    print("lamda or kappa in UKF is :", l)
    
    
    N_T = test_target.size()[0]

    # LOSS
    loss_fn = nn.MSELoss(reduction='mean')
    
    # MSE [Linear]
    MSE_UKF_linear_arr = torch.empty(N_T)
    points = JulierSigmaPoints(n=SysModel.m,kappa=l.item())
    # points = MerweScaledSigmaPoints(SysModel.m, alpha=.1, beta=2., kappa=-1)

    def fx(x, dt):
        return SysModel.f(torch.from_numpy(x).float()).numpy()

    def hx(x):
        # print("Fuck me the hell out ,  i am tired !!!!!!! ")
        return SysModel.h(torch.from_numpy(x).float()).numpy()

    UKF = UnscentedKalmanFilter(dim_x=SysModel.m, dim_z=SysModel.n, dt=delta_t, fx=fx, hx=hx,points=points)
    UKF.x = torch.squeeze(SysModel.m1x_0).numpy() # initial state
    UKF.P = (SysModel.m2x_0 ).numpy() # initial uncertainty+ 1e-5*torch.eye(SysModel.m)
    UKF.R = SysModel.R.numpy()
    UKF.Q = SysModel.Q.numpy()
 
    UKF_out = torch.empty([N_T, SysModel.m, SysModel.T_test])
    UKF_Sigma = torch.empty([N_T , SysModel.m , SysModel.m,SysModel.T_test])
    UKF_prior =  torch.empty([N_T, SysModel.m, SysModel.T_test])
    KG_array = torch.empty([N_T, SysModel.m,SysModel.n ,SysModel.T_test])
    
    
    test_input2 = test_input.to("cpu")
    start = time.time()
    for j in range(0, N_T):
        
        if init_cond is not None:
            UKF.x = torch.unsqueeze(init_cond[j, :], 1).numpy()
        
        for z in range(0, SysModel.T_test):
            UKF.predict()
            UKF.update(test_input2[j,:,z].numpy())       
            UKF_out[j,:,z] = torch.from_numpy(UKF.x)
            UKF_Sigma[j,:,:,z] = torch.from_numpy(UKF.P)
            UKF_prior[j,:,z] = torch.from_numpy(UKF.x_prior)
            KG_array[j,:,:,z]= torch.from_numpy(UKF.K)
        
        UKF_out2 = UKF_out[j,:,:].to("cuda") 
        if allStates:
            MSE_UKF_linear_arr[j] = loss_fn(UKF_out2, test_target[j, :, :]).item()
        else:
            loc = torch.tensor([True, False]*(SysModel.m//2))
            MSE_UKF_linear_arr[j] = loss_fn(UKF_out2[loc, :], test_target[j, :, :][loc, :]).item()
        
        # All trajectories were produced with the same initial conditions
        # Since we apply the same filter for each trajectory, it keeps the results from last trajectory
        # So we have to reset it to initial conditions for each trajectory
        UKF.x = torch.squeeze(SysModel.m1x_0).numpy()
        UKF.x_prior = torch.tensor([0.0,0.0]).numpy()
        UKF.x_post = torch.tensor([0.0,0.0]).numpy()
        UKF.P = (SysModel.m2x_0 ).numpy()
        UKF.P_prior = torch.tensor([[1.0,0.0],[0.0,1.0]]).numpy()
        UKF.P_post = torch.tensor([[1.0,0.0],[0.0,1.0]]).numpy()
        
        

    UKF_out = UKF_out.to("cuda")
    
    end = time.time()
    t = end - start

    MSE_UKF_linear_avg = torch.mean(MSE_UKF_linear_arr)
    MSE_UKF_dB_avg = 10 * torch.log10(MSE_UKF_linear_avg)
    # Standard deviation
    MSE_UKF_dB_std = torch.std(MSE_UKF_linear_arr, unbiased=True)
    MSE_UKF_dB_std = 10 * torch.log10(MSE_UKF_dB_std)

    print("UKF - MSE LOSS:", MSE_UKF_dB_avg, "[dB]")
    print("UKF - MSE STD:", MSE_UKF_dB_std, "[dB]")
    # Print Run Time
    print("Inference Time:", t)
    return [MSE_UKF_linear_arr, MSE_UKF_linear_avg, MSE_UKF_dB_avg,KG_array ,UKF_out , UKF_Sigma,UKF_prior]
