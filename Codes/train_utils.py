import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import scipy.io as scio
import os
import matplotlib.pyplot as plt

from PhyConvNet import PhyConvNet

from loss import loss_generator, CustomLoss

def train(source_BHP, Qinj, Tmap, p0, sw0, n_iters, learning_rate, 
          dt, Rate_vec, BHP_vec, Perm, tsteps, steps_sim):

    train_loss_list = []
    second_last_state = []
    prev_output = []

    loss_physics = loss_generator(dt)
    
    outputs = []
    outputs.append(torch.cat((p0, sw0), dim=1))
    for k in range(tsteps):
        # pre_model_save_path = '/scratch/user/jungangc/PICNN-2phase/PICNN-2phaseflow-constBHP-64by64-homo-Progressive-10steps-final/Codes/model/checkpoint1000_{}.pt'.format(k)
        model_save_path = '/scratch/user/jungangc/PICNN-2phase/PICNN-2phaseflow-constBHP-64by64-heter-TransferLearning-50stepsby2-final/Codes/model/checkpoint2000_{}.pt'.format(k)

        model = PhyConvNet(num_classes=1, in_channels=2, time_sim = tsteps).cuda()
        if k>0:
            prev_model_save_path = '/scratch/user/jungangc/PICNN-2phase/PICNN-2phaseflow-constBHP-64by64-heter-TransferLearning-50stepsby2-final/Codes/model/checkpoint2000_{}.pt'.format(k-1)
            checkpoint  = torch.load(prev_model_save_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            
        best_loss = 1e9
        # load previous model
        # optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4) 
        optimizer = optim.Adam(model.parameters(), lr=learning_rate) 
        scheduler = StepLR(optimizer, step_size=100, gamma=0.995)  
        # optimizer = optim.LBFGS(model.parameters(), lr=learning_rate) 
        # scheduler = StepLR(optimizer, step_size=100, gamma=0.98)  
        # model, optimizer, scheduler = load_checkpoint(model, optimizer, scheduler, 
        #     pre_model_save_path)
        for param_group in optimizer.param_groups:
            print(param_group['lr'])
            
        loss_criterion = 1e9

        # print(outputs[k].size())
        p_prev = outputs[k][:, 0:1, :, :]
        sw_prev = outputs[k][:, 1:2, :, :]
        
        bhp = source_BHP[k:k+1,...]
        rate = Qinj[k:k+1,...]
        epoch =0
        monitor_loss = 1e9
        while monitor_loss>0.001 and epoch<50000:

            optimizer.zero_grad()
            batch_loss = 0
            # output is a list
            xp, xs = model(p_prev, sw_prev, bhp , rate , Tmap)
            
            # get loss

            lossfunc = CustomLoss(initial_coefficient=[1.0, 1.0])
            # print(tsteps)
            loss = lossfunc.compute_loss(p_prev, sw_prev, xp, xs, loss_physics, dt, Rate_vec[:,k:k+1], BHP_vec[:,k:k+1], Perm, sim_step=steps_sim)

            loss.backward(retain_graph=True)
            # loss.backward()
            batch_loss += loss.item()
            # loss_criterion = batch_loss
            # nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)  #apply cilp gradient
            optimizer.step()
            scheduler.step()
            
            monitor_loss = batch_loss
            if epoch%30==0:
                # print loss and timestep to monitor the training
                print('[timestep: %d] [epoch: %d] loss: %.10f' % ((k+1), (epoch+1), batch_loss))
            train_loss_list.append(batch_loss)

            epoch+=1
            # save model
            if batch_loss < best_loss:
                save_checkpoint(model, optimizer, scheduler, model_save_path)
                best_loss = batch_loss
                xp_save = xp
                xs_save = xs
                
        outputs.append(torch.cat((xp_save, xs_save), dim=1))

    return train_loss_list, outputs


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def save_checkpoint(model, optimizer, scheduler, save_dir):
    '''save model and optimizer'''

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
        }, save_dir)


def load_checkpoint(model, optimizer, scheduler, save_dir):
    '''load model and optimizer'''
    
    checkpoint = torch.load(save_dir)
    model.load_state_dict(checkpoint['model_state_dict'])

    if (not optimizer is None):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    print('Pretrained model loaded!')

    return model, optimizer, scheduler


def summary_parameters(model):
    for i in model.parameters():
        print(i.shape)


def frobenius_norm(tensor):
    return np.sqrt(np.sum(tensor ** 2))
