"""# **Class: Unscented Kalman Filter**
Handcrafted Unscented Kalman Filter
batched version
"""
import torch
from filterpy.kalman.sigma_points import JulierSigmaPoints
import os
import matplotlib.pyplot as plt


def Ws_method(M,lamda) :
    W = torch.zeros(4*M + 1 , 1 )
    W[0] = lamda / (2*M + lamda)
    W[1:] = 1 / (2*(2*M + lamda))
    return W 




class UnscentedKalmanFilter:

    def __init__(self, SystemModel, args , lamda , W):  
        #Device
        print("USE gpu on UKF ? :", args.use_cuda)
        if args.use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        print("Using " , self.device )
        # process model
        self.f = SystemModel.f
        self.m = SystemModel.m
        self.Q = SystemModel.Q.to(self.device)
        self.M = self.m//2
        # observation model
        self.h = SystemModel.h
        self.n = SystemModel.n
        self.R = SystemModel.R.to(self.device)
        # sequence length (use maximum length if random length case)
        self.T = SystemModel.T
        self.T_test = SystemModel.T_test
        
        self.lamda = lamda
        self.W = W.to(self.device)
        
  
  
    def State_batched_sigma_points_calculation(self , m1x , m2x ) :
        sz = m1x.size()
        batched_sigma_points = torch.zeros(sz[0], 2*self.M ,  2*(2*self.M) + 1 ).to(self.device) # [ batch_size ,  m = 2M , 4M + 1 ]
        m2x_edited = torch.mul(2*self.M + self.lamda  , m2x)
        # print("E000000000000000")
        # print("4*self.M + 1 is :",4*self.M + 1)
        # print("Printing m2 edited:")
        # print(m2x_edited)
        cholesky_dec = torch.linalg.cholesky(m2x_edited) # method returns a lower triangular matrix 
        # print("cholseky dec is of size:", cholesky_dec.size())
        # print("cholesky_dec is :",cholesky_dec)
        # input()
        sigma_points_pos = 0
        batched_sigma_points[: , : , sigma_points_pos] = torch.squeeze( m1x )
        sigma_points_pos+= sigma_points_pos + 1
        for i in range(0,2*self.M) :
            m1x_dash = m1x  +  torch.unsqueeze(cholesky_dec[: , : , i] ,2)  
            batched_sigma_points[: , : , sigma_points_pos] = torch.squeeze( m1x_dash )
            sigma_points_pos+=1
        for i in range(2*self.M, 4*self.M) :
            m1x_dash = m1x  -  torch.unsqueeze(cholesky_dec[: , : , i-2*self.M ] ,2)
            batched_sigma_points[: , : , i+1] = torch.squeeze(m1x_dash)
            sigma_points_pos+=1
        sigma_points_pos = 0
        # print("sigma point array is of size:",batched_sigma_points.size())
        # print("sigma point array is:",batched_sigma_points)    
        # print("Printing sigma points")
        # plt.figure(0)
        # plt.scatter(batched_sigma_points[0,0,:].cpu(),batched_sigma_points[0,1,:].cpu())
        # plt.figure(1)
        # plt.scatter(batched_sigma_points[1,0,:].cpu(),batched_sigma_points[1,1,:].cpu())
        # plt.show()
            
        # input()
        return batched_sigma_points 
    
  
    def Combine_batched_transformed_sigma_points(self , sigmas ) :
        sz = sigmas.size()
        m1_est = torch.zeros(sz[0] , sz[1] , 1).to(self.device)
        scaled_sigmas = torch.mul(self.W.squeeze(),sigmas)
        m1_est = scaled_sigmas.sum(2)
        m1_est = m1_est.view(scaled_sigmas.shape[0],scaled_sigmas.shape[1],1)
        return m1_est
        
    def Calculate_batched_covariance_matrix(self, transformed_sigma_points , m1  ) :
        sz = transformed_sigma_points.size()
        m2_est = torch.zeros(self.batch_size , sz[1] , sz[1]).to(self.device)
        for i in range(0 , self.batch_size) :
            for j in range(0 , 4*self.M + 1) :
                u = transformed_sigma_points[i , : , j] - (m1[i,:,:]).squeeze()
                m2_est[i , : , :] = m2_est[i , : , :] + self.W[j]*torch.outer(u,u)
        
        m2_est = m2_est.transpose(2,1)
        return m2_est
         
    def Calculate_batched_cross_covariance_matrix(self , transformed_sigma_points_x , transformed_sigma_points_y, m1x_prior , m1y) :  
        m2xy = torch.zeros(self.batch_size, 2*self.M , self.n).to(self.device)
        for i in range(0 , self.batch_size) :
            for j in range(0 , 4*self.M + 1) :
                u = transformed_sigma_points_x[i , : , j] - (m1x_prior[i,:,:]).squeeze()
                r = transformed_sigma_points_y[i , : , j] - (m1y[i,:,:]).squeeze()
                m2xy[i , : , :] = m2xy[i , : , :] + self.W[j]*torch.outer(u,r)
        
        # m2xy= m2xy.transpose(2,1)
        return m2xy
  
    # Predict
    def Predict(self):
        # Create 4M + 1 Sigma points
        
        batched_sigma_points_x = self.State_batched_sigma_points_calculation(self.m1x_posterior , self.m2x_posterior  ).to(self.device) 
        # Transform the Sigma points via the System function
        self.m1x_prior_points = self.f(batched_sigma_points_x).to(self.device)
        # Predict the 1-st moment of x 
        self.m1x_prior  = self.Combine_batched_transformed_sigma_points(self.m1x_prior_points ).to(self.device)
        # Predict the 2-nd moment of x
        self.m2x_prior = self.Calculate_batched_covariance_matrix(self.m1x_prior_points , self.m1x_prior ).to(self.device)
        # print("$$$$$$$$$$$$$$$$$$$$$$$$")
        # print("Motherfucker Q is:\n",self.Q)
        # print("m2x_prior before Q addition is :\n",self.m2x_prior)
        self.m2x_prior = self.m2x_prior + self.Q
        # print("m2x_prior after Q addition is :\n",self.m2x_prior)

        # Create 4M + 1 Sigma points
        batched_sigma_points_y = self.State_batched_sigma_points_calculation(self.m1x_prior , self.m2x_prior ).to(self.device)
        # Transdorm the Sigma points via the Observation function 
        self.m1y_points = self.h(batched_sigma_points_x).to(self.device)
        # Predict the 1-st moment of y
        self.m1y =  self.Combine_batched_transformed_sigma_points(self.m1y_points ).to(self.device)
        # Predict the 2-nd moment of y
        self.m2y = self.Calculate_batched_covariance_matrix(self.m1y_points , self.m1y ).to(self.device)
        # print("$$$$$$$$$$$$$$$$$$$$$$$$")
        # print("Motherfucker R is:\n",self.R)
        # print("m2y before R addition is :\n",self.m2y)
        self.m2y = self.m2y + self.R
        # print("m2y after R addition is :\n",self.m2y)
        # input()
        #Predict the Cross-Covariance matrix between the state and observation vectors 
        self.m2xy =  self.Calculate_batched_cross_covariance_matrix(self.m1x_prior_points , self.m1y_points , self.m1x_prior , self.m1y ).to(self.device)
        # print("m2xy is :\n",self.m2xy)
    # Compute the Kalman Gain
    def KGain(self):
        self.KG = torch.bmm(self.m2xy, torch.linalg.inv(self.m2y))

        #Save KalmanGain
        self.KG_array[:,:,:,self.i] = self.KG
        self.i += 1

    # Innovation
    def Innovation(self, y):
        self.dy = y - self.m1y

    # Compute Posterior
    def Correct(self):
        # Compute the 1-st posterior moment
        # print("KG is:\n",self.KG)
        self.m1x_posterior = self.m1x_prior + torch.bmm(self.KG, self.dy)

        # Compute the 2-nd posterior moment - Usual form
        #self.m2x_posterior = torch.bmm(self.m2y, torch.transpose(self.KG, 1, 2))
        self.m2x_posterior = self.m2x_prior - torch.bmm(self.KG, torch.bmm(self.m2y, torch.transpose(self.KG, 1, 2)) )
        
        
        # Compute the 2-nd posterior moment - Numerically Stable form
        # self.m2x_posterior = torch.bmm(self.m2y, torch.transpose(self.KG, 1, 2))
        # self.m2x_posterior = self.m2x_prior + torch.bmm(self.KG, self.m2x_posterior)
        # self.m2x_posterior = self.m2x_posterior - torch.bmm(self.m2xy ,torch.transpose(self.KG, 1, 2) )
        # self.m2x_posterior = self.m2x_posterior - torch.bmm(self.KG ,torch.transpose(self.m2xy, 1, 2) )
        #Impose symmetricity on the covariance matrix
        self.m2x_posterior = (self.m2x_posterior + self.m2x_posterior.transpose(2,1))/2 
        # I_0 = torch.eye(2*self.M , 2*self.M).to(self.device) 
        # I = I_0.repeat(self.batch_size,1,1)
        # factor = torch.tensor([0.00001]).to(self.device)
        # self.m2x_posterior += I*factor
        

    def Update(self, y):
        self.Predict()
        self.KGain()
        self.Innovation(y)
        self.Correct()

        return self.m1x_posterior, self.m2x_posterior , self.m1x_prior

    
    def Init_batched_sequence(self, m1x_0_batch, m2x_0_batch):
        self.m1x_0_batch = m1x_0_batch # [batch_size, m, 1]
        self.m2x_0_batch = m2x_0_batch # [batch_size, m, m]

    ######################
    ### Generate Batch ###
    ######################
    def GenerateBatch(self, y):
        """
        input y: batch of observations [batch_size, n, T]
        """
        y = y.to(self.device)
        self.batch_size = y.shape[0] # batch size
        T = y.shape[2] # sequence length (maximum length if randomLength=True)

        # Pre allocate KG array
        self.KG_array = torch.zeros([self.batch_size,self.m,self.n,T]).to(self.device)
        self.i = 0 # Index for KG_array alocation

        # Allocate Array for 1st and 2nd order moments (use zero padding)
        self.x = torch.zeros(self.batch_size, self.m, T).to(self.device)
        self.sigma = torch.zeros(self.batch_size, self.m, self.m, T).to(self.device)
        self.x_prior = torch.zeros(self.batch_size, self.m, T).to(self.device)
            
        # Set 1st and 2nd order moments for t=0
        self.m1x_posterior = self.m1x_0_batch.to(self.device)
        self.m2x_posterior = self.m2x_0_batch.to(self.device)
        
        self.m1x_prior =  torch.zeros(self.batch_size, self.m, T).to(self.device)
        self.m2x_prior =  torch.zeros(self.batch_size, self.m, self.m, T).to(self.device)
        
        print("initial x is :\n",self.m1x_posterior)
        print("initial cov mat is :\n",self.m2x_posterior)
        
        
        # self.m2x_posterior = (self.m2x_posterior + self.m2x_posterior.transpose(2,1))/2 #Impose symmetricity on the covariance matrix 
        # I_0 = torch.eye(2*self.M , 2*self.M).to(self.device) 
        # I = I_0.repeat(self.batch_size,1,1)
        # factor = torch.tensor([0.00001]).to(self.device)
        # self.m2x_posterior += I*factor

        # Generate in a batched manner
        for t in range(0, T):
            # print("time index t is : " , t)
            yt = torch.unsqueeze(y[:, :, t],2)
            xt,sigma,xt_prior = self.Update(yt)
            self.x[:, :, t] = torch.squeeze(xt,2)
            self.sigma[:, :, :, t] = sigma
            self.x_prior[:,:,t] = torch.squeeze(xt_prior,2)

# class UnscentedKalmanFilter:

#     def __init__(self, SystemModel, args,lamda,W):
#         # Device
#         if args.use_cuda:
#             self.device = torch.device('cuda')
#         else:
#             self.device = torch.device('cpu')
#         self.F = SystemModel.F
#         self.m = SystemModel.m
#         self.Q = SystemModel.Q.to(self.device)

#         self.H = SystemModel.H
#         self.n = SystemModel.n
#         self.R = SystemModel.R.to(self.device)

#         self.T = SystemModel.T
#         self.T_test = SystemModel.T_test
   
#     # Predict

#     def Predict(self):
#         # Predict the 1-st moment of x
#         self.m1x_prior = torch.bmm(self.batched_F, self.m1x_posterior).to(self.device)

#         # Predict the 2-nd moment of x
#         self.m2x_prior = torch.bmm(self.batched_F, self.m2x_posterior)
#         self.m2x_prior = torch.bmm(self.m2x_prior, self.batched_F_T) + self.Q

#         # Predict the 1-st moment of y
#         self.m1y = torch.bmm(self.batched_H, self.m1x_prior)

#         # Predict the 2-nd moment of y
#         self.m2y = torch.bmm(self.batched_H, self.m2x_prior)
#         self.m2y = torch.bmm(self.m2y, self.batched_H_T) + self.R

#     # Compute the Kalman Gain
#     def KGain(self):
#         self.KG = torch.bmm(self.m2x_prior, self.batched_H_T)
               
#         self.KG = torch.bmm(self.KG, torch.inverse(self.m2y))

#     # Innovation
#     def Innovation(self, y):
#         self.dy = y - self.m1y

#     # Compute Posterior
#     def Correct(self):
#         # Compute the 1-st posterior moment
#         self.m1x_posterior = self.m1x_prior + torch.bmm(self.KG, self.dy)

#         # Compute the 2-nd posterior moment
#         self.m2x_posterior = torch.bmm(self.m2y, torch.transpose(self.KG, 1, 2))
#         self.m2x_posterior = self.m2x_prior - torch.bmm(self.KG, self.m2x_posterior)

#     def Update(self, y):
#         self.Predict()
#         self.KGain()
#         self.Innovation(y)
#         self.Correct()

#         return self.m1x_posterior,self.m2x_posterior,self.m1x_prior

#     def Init_batched_sequence(self, m1x_0_batch, m2x_0_batch):

#             self.m1x_0_batch = m1x_0_batch # [batch_size, m, 1]
#             self.m2x_0_batch = m2x_0_batch # [batch_size, m, m]

#     ######################
#     ### Generate Batch ###
#     ######################
#     def GenerateBatch(self, y):
#         """
#         input y: batch of observations [batch_size, n, T]
#         """
#         y = y.to(self.device)
#         self.batch_size = y.shape[0] # batch size
#         T = y.shape[2] # sequence length (maximum length if randomLength=True)

#         # Batched F and H
#         self.batched_F = self.F.view(1,self.m,self.m).expand(self.batch_size,-1,-1).to(self.device)
#         self.batched_F_T = torch.transpose(self.batched_F, 1, 2).to(self.device)
#         self.batched_H = self.H.view(1,self.n,self.m).expand(self.batch_size,-1,-1).to(self.device)
#         self.batched_H_T = torch.transpose(self.batched_H, 1, 2).to(self.device)

#         # Pre allocate KG array
#         self.KG_array = torch.zeros([self.batch_size,self.m,self.n,T]).to(self.device)
        
        
#         # Allocate Array for 1st and 2nd order moments (use zero padding)
#         self.x = torch.zeros(self.batch_size, self.m, T).to(self.device)
#         self.sigma = torch.zeros(self.batch_size, self.m, self.m, T).to(self.device)
#         self.x_predicted = torch.zeros(self.batch_size, self.m, T).to(self.device)    
#         # Set 1st and 2nd order moments for t=0
#         self.m1x_posterior = self.m1x_0_batch.to(self.device)
#         self.m2x_posterior = self.m2x_0_batch.to(self.device)

#         # Generate in a batched manner
#         for t in range(0, T):
#             yt = torch.unsqueeze(y[:, :, t],2)
#             xt,sigmat,x_pred = self.Update(yt)
#             self.x[:, :, t] = torch.squeeze(xt,2)
#             self.sigma[:, :, :, t] = sigmat
#             self.x_predicted[:, :, t] = torch.squeeze(x_pred,2)
