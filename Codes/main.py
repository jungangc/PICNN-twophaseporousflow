'''PhyCRNet for solving spatiotemporal PDEs'''

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
import time
import os
from torch.utils.data import DataLoader, TensorDataset

from PhyConvNet import PhyConvNet
from train_utils import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.manual_seed(66)
np.random.seed(66)
torch.set_default_dtype(torch.float32)

Nx= 64
Ny =64
BHPmat = scio.loadmat('BHP_full.mat')
BHP_vec = torch.tensor(BHPmat['BHP_full'], dtype=torch.float32).cuda() 

# print(BHP_vec)
# Rate  = np.array([[100]]) # [STB/day]
# Rate_vec =      torch.tensor(Rate, dtype=torch.float32).repeat(1,300).cuda()
Rate = scio.loadmat('Qinj_full.mat')
Rate_vec = torch.tensor(Rate['Qinj_full'], dtype=torch.float32).cuda() 
# print(Rate_vec)

TRUE_PERM = scio.loadmat('/scratch/user/jungangc/PICNN-2phase/PICNN-2phaseflow-constBHP-64by64-heter-TransferLearning-50stepsby2-final/Codes/system/TRUE_PERM_64by64.mat')
Perm = torch.tensor(TRUE_PERM['TRUE_PERM'], dtype=torch.float32).cuda()
# Perm = torch.tensor(0.10*np.ones((Nx, Ny)), dtype=torch.float32).cuda()

if __name__ == '__main__':
    print(os.getcwd())
    ######### download the ground truth data ############
#     data_dir = '/content/PhyCRNet-main-1phaseflow/Datasets/data/2dBurgers/burgers_1501x2x128x128.mat'    
    data_dir_p = '/scratch/user/jungangc/PICNN-2phase/PICNN-2phaseflow-constBHP-64by64-heter-TransferLearning-50stepsby2-final/Datasets/data/twophaseflow/pressure_401x1x64x64.mat'  
    data_dir_sw = '/scratch/user/jungangc/PICNN-2phase/PICNN-2phaseflow-constBHP-64by64-heter-TransferLearning-50stepsby2-final/Datasets/data/twophaseflow/saturation_401x1x64x64.mat' 
    data_p = scio.loadmat(data_dir_p)
    data_sw = scio.loadmat(data_dir_sw)
    # uv = data['uv'] # [t,c,h,w]  
    p = data_p['Psim'] # [t,c,h,w]  
    sw = data_sw['Swsim'] # [t,c,h,w] 

    # initial conidtion
    p0 = torch.tensor(p[0:1,...], dtype=torch.float32).cuda()
    # p0 = p0/3000
    sw0 = torch.tensor(sw[0:1,...], dtype=torch.float32).cuda()

    steps_net = 50
    dt = 2
    
    # time map  np.array(range(1,301,dt))
    steps_sim = 1
    
    Tmap = torch.tensor(np.zeros((steps_net,1,64,64)), dtype=torch.float32).cuda()
    for k in range(steps_net):
        Tmap[k:k+1,...] = torch.tensor(np.ones((1,1,64,64))*(k+1)*dt/steps_sim, dtype=torch.float32).cuda()
# #     T =
    # source BHP 
    BHP = np.zeros((steps_net,1,64,64))
    source_BHP = torch.tensor(BHP, dtype=torch.float32).cuda()
    source_BHP[:,0, 49, 49]=BHP_vec[0,:steps_net*dt:dt]    # Pi =3000
    source_BHP[:,0, 12, 12] = BHP_vec[1,:steps_net*dt:dt]
    
    rate = np.zeros((steps_net,1,64,64))
    Qinj = torch.tensor(rate, dtype=torch.float32).cuda()
    Qinj[:, 0,31,31] = Rate_vec[0,:steps_net*dt:dt]   # Qmax =1500
    Qinj[:, 0,12,49] = Rate_vec[1,:steps_net*dt:dt]
    Qinj[:, 0,49,12] = Rate_vec[2,:steps_net*dt:dt]

    # P0 = p0.repeat(time_sim, 1, 1, 1)
    # SW0 = sw0.repeat(time_sim, 1, 1, 1)
    # print(BHP_vec[1,:steps_net])
    # inputs = torch.cat((Qinj, source_BHP, P0, SW0), dim=1)    # concat time map and control 
    
    
    # dataset = TensorDataset(inputs)
    # dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    
    ################# build the model #####################
    # time_batch_size = 300
    # steps = time_batch_size
    # effective_step = list(range(0, steps))
    # num_time_batch = int(time_steps / time_batch_size)
    np.ones((steps_net-1, 1))
    n_iters_adam = 30000
    lr_adam = 0.01 #1e-3 

    fig_save_path = '/scratch/user/jungangc/PICNN-2phase/PICNN-2phaseflow-constBHP-64by64-heter-TransferLearning-50stepsby2-final/Datasets/figures/'  

    # model = PhyConvNet(latent_size = 200, dt = dt, time_sim = steps_net, 
    #     input_channels = 4,  
    #     hidden_channels = [16, 32, 64, 128, 64, 32, 16], 
    #     input_kernel_size = [4, 4, 4, 4, 4, 4, 4], 
    #     input_stride = [2, 2, 2, 2, 2, 2, 2], 
    #     input_padding = [1, 1, 1, 1, 1, 1, 1],  
    #     num_layers=[4,3,1], 
    #     upscale_factor=8).cuda()


    start = time.time()
    train_loss, outputs = train(source_BHP, Qinj, Tmap, p0, sw0, n_iters_adam,
        lr_adam, dt, Rate_vec[:,:steps_net*dt:dt] , BHP_vec[:,:steps_net*dt:dt] , Perm, steps_net, steps_sim)
    end = time.time()
    
    np.save('/scratch/user/jungangc/PICNN-2phase/PICNN-2phaseflow-constBHP-64by64-heter-TransferLearning-50stepsby2-final/Codes/model/train_loss', train_loss)  
    print('The training time is: ', (end-start))



















