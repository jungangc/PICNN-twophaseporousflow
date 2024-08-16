import torch
import torch.nn as nn

import numpy as np
import scipy.io as scio
import os

from system.twophaseflow_init import INITIALIZE
from system.twophaseflow_preprocess import pre_processing
from system.twophaseflow_geo_avg import twophaseflow_geo_avg
from system.twophaseflow_update_fluid_property import update_fluid_property
from system.twophaseflow_update_rock_property import update_rock_property
from system.twophaseflow_compute_mobility import compute_mobility
from system.twophaseflow_compute_source import compute_source
from system.twophaseflow_compute_accumulation import compute_accumulation
from system.twophaseflow_compute_transmisibility import compute_transmisibility


class Conv1dDerivative(nn.Module):
    def __init__(self, DerFilter, resol, kernel_size=3, name=''):
        super(Conv1dDerivative, self).__init__()

        self.resol = resol  # $\delta$*constant in the finite difference
        self.name = name
        self.input_channels = 1
        self.output_channels = 1
        self.kernel_size = kernel_size

        self.padding = int((kernel_size - 1) / 2)
        self.filter = nn.Conv1d(self.input_channels, self.output_channels, self.kernel_size, 
            1, padding=0, bias=False)
        
        # Fixed gradient operator
        self.filter.weight = nn.Parameter(torch.FloatTensor(DerFilter), requires_grad=False)  

    def forward(self, input):
        derivative = self.filter(input)
        return derivative / self.resol
    
class loss_generator(nn.Module):
    ''' Loss generator for physics loss '''

    def __init__(self, delta_t):
        ''' Construct the derivatives, X = Width, Y = Height '''
       
        super(loss_generator, self).__init__()

        # temporal derivative operator
        self.dt = Conv1dDerivative(
            DerFilter = [[[-1, 0, 1]]],
            resol = (delta_t *2),
            kernel_size = 3,
            name = 'partial_t').cuda()

    def get_phy_Loss(self, p_prev, sw_prev, xp, xs, delta_t, Qinj_schedule, BHP_schedule, Perm, tsteps):

        # Initialize the 2 phase porous flow system
        Geometry, Discretization, Fluid, Rock, Wells, Conditions, Constants, FullSolution = INITIALIZE(Qinj_schedule, BHP_schedule, Perm, tsteps)
        Discretization, Wells, FullSolution = pre_processing(Geometry, Discretization, Rock, Wells, Conditions, Constants, FullSolution)
        Rock = twophaseflow_geo_avg(Discretization, Rock, FullSolution, Constants)
#         # spatial derivatives
        # temporal derivative - p
        # print("output: ", output.size())
        p_curr = xp
        # print('p_curr', p_curr)
        p_prev = p_prev
        # p_next = output[2:, 0:1, :, :]
        # p_diff_t = (p_next-2*p_curr+p_prev)/(2*delta_t)
        p_diff_t = (p_curr-p_prev)/delta_t
        
        sw_curr = xs
        sw_prev = sw_prev
        # sw_next = output[2:, 1:2, :, :]
        # sw_diff_t = (sw_next-2*sw_curr+sw_prev)/(2*delta_t)
        sw_diff_t = (sw_curr-sw_prev)/delta_t
        
        lent = p_curr.shape[0]
        lenx = p_curr.shape[3]
        leny = p_curr.shape[2]

        # reshape current pressure 
        p_conv1d = p_curr.permute(2, 3, 1, 0)  # [height(Y), width(X), c, step]
        p_vec = p_conv1d.reshape(lenx*leny,1,lent)
        P_out_vec = p_vec.permute(2,1,0)     # [step, c, X*Y]
        
        # reshape current saturation
        sw_conv1d = sw_curr.permute(2, 3, 1, 0)  # [height(Y), width(X), c, step]
        sw_vec = sw_conv1d.reshape(lenx*leny,1,lent)
        Sw_out_vec = sw_vec.permute(2,1,0)   # [step, c, X*Y]
        
        # temporal derivative - p_t
        p_conv1d = p_diff_t.permute(2, 3, 1, 0)  # [height(Y), width(X), c, step]
        p_dt_vec = p_conv1d.reshape(lenx*leny,1,lent)
        P_dt_vec = p_dt_vec.permute(2,1,0)    # [step, c, X*Y]
        
        # temporal derivative - sw_t
        sw_conv1d = sw_diff_t.permute(2, 3, 1, 0)  # [height(Y), width(X), c, step]
        sw_dt_vec = sw_conv1d.reshape(lenx*leny,1,lent)
        Sw_dt_vec = sw_dt_vec.permute(2,1,0)     # [step, c, X*Y]

        # pressure and saturation for updating properties
        # FullSolution.Pcurrent =  P_prev_vec.squeeze().t()
        # FullSolution.Swcurrent = Sw_prev_vec.squeeze().t()
        FullSolution.Pcurrent =  P_out_vec.squeeze(dim=1).t()
        FullSolution.Swcurrent = Sw_out_vec.squeeze(dim=1).t()
        
        # print(FullSolution.Pcurrent.size())
        Fluid    = update_fluid_property(Fluid,FullSolution) 
        # print(Fluid.bo)
        Rock  = update_rock_property(Rock,FullSolution)
        # print(Rock.kro)
        Mobility = compute_mobility(Fluid,Rock)
        # print('Mobility:', Mobility.fro)
        Wells   = compute_source(Discretization, Mobility, Wells)
        # print('wells', Wells.Qo[0])
        Accumulation = compute_accumulation(Discretization, Constants, Fluid, Rock, FullSolution)
        # print(Accumulation.Acc_op[0])
        Transmisibility  = compute_transmisibility(Discretization, Rock, Mobility, FullSolution)

        Acc     = Accumulation.Acc
        # print(Transmisibility.T.size())
        Trans = Transmisibility.T - Wells.Q
        q = Wells.q
        
        Xp = P_out_vec.squeeze(dim=1).t()
        Xs = Sw_out_vec.squeeze(dim=1).t()
        X = torch.vstack((Xp, Xs))

        dXpdt = P_dt_vec.squeeze(dim=1).t()
        dXsdt = Sw_dt_vec.squeeze(dim=1).t()
        dXdt = torch.vstack((dXpdt, dXsdt))
        
        residual = torch.sparse.mm(Trans, X)-torch.sparse.mm(Acc, dXdt) + q
        # print('min oil',torch.min(residual_p))
        # print('max oil',torch.max(residual_p))
        # print('min water',torch.min(residual_sw))
        # print('max water',torch.max(residual_sw))
        return residual

class CustomLoss(nn.Module):
    def __init__(self, initial_coefficient=[0.01, 1.0]):
        super(CustomLoss, self).__init__()
        self.coefficient = nn.Parameter(torch.tensor(initial_coefficient))

    def compute_loss(self, p_prev, sw_prev, xp, xs, loss_func, delta_t, Qinj_schedule, BHP_schedule, Perm, sim_step):
        ''' calculate the phycis loss '''
        # get physics loss
        # mse_loss1 = nn.MSELoss()
        mse_loss1 = nn.SmoothL1Loss(beta=10.0)
        # mse_loss1 = nn.L1Loss()
        # mse_loss2 = nn.MSELoss()
        # mse_loss2 = nn.SmoothL1Loss(beta=10.0)
        # mse_loss2 = nn.L1Loss()
        # mse_loss = nn.L1Loss()
        # output = output *3000    # to enforce the original CNN output is between 0 to 1
    #     f_u, f_v = loss_func.get_phy_Loss(output)
        f_p  = loss_func.get_phy_Loss(p_prev, sw_prev, xp, xs, delta_t, Qinj_schedule, BHP_schedule, Perm, sim_step)
        # loss_p =  mse_loss(f_p, torch.zeros_like(f_p).cuda())
        loss_p =  mse_loss1(f_p, torch.zeros_like(f_p).cuda())
        # loss_sw =  mse_loss2(f_sw, torch.zeros_like(f_sw).cuda())
        # loss =  mse_loss(f_p, P_out_vec)

        # loss_d = compute_loss_data(output, P_data)
        loss = loss_p
        # loss = self.coefficient[0]*loss_p + self.coefficient[1]*loss_sw 
        return loss

def compute_loss_data(output, P_data):
    p_pred = torch.cat((output[1:-1, 0:1, 9, 9], output[1:-1, 0:1, 54, 54]), dim=1)
    P_pred = p_pred.permute(1,0)
    
    P_data = P_data[:,1:]
    # mse_loss = nn.L1Loss()
    mse_loss = nn.SmoothL1Loss(beta=100.0)
    loss_data = mse_loss(P_pred, P_data).cuda()
    return loss_data

def compute_loss_adp(output, loss_func, delta_t, epoch):
    ''' calculate the phycis loss '''
    
    beta = [100, 10]
    # get physics loss
    # mse_loss = nn.MSELoss()
    if epoch < 10000: 
        mse_loss = nn.SmoothL1Loss(beta=beta[0])
    # elif epoch >=5000 and epoch < 15000:
    #     mse_loss = nn.SmoothL1Loss(beta=beta[1])
    else: 
        mse_loss = nn.SmoothL1Loss(beta=beta[1])
    # mse_loss = nn.L1Loss()
    # output = output *3000    # to enforce the original CNN output is between 0 to 1
#     f_u, f_v = loss_func.get_phy_Loss(output)
    f_p, P_out_vec = loss_func.get_phy_Loss(output, delta_t)
#     loss =  mse_loss(f_u, torch.zeros_like(f_u).cuda()) + mse_loss(f_v, torch.zeros_like(f_v).cuda()) 
    loss =  mse_loss(f_p, torch.zeros_like(f_p).cuda())
    # loss =  mse_loss(f_p, P_out_vec)
    return loss
    
def compute_loss_w(output, loss_func, delta_t):
    ''' calculate the phycis loss '''
    
    # # Padding x axis due to periodic boundary condition
    # # shape: [t, c, h, w]
    # output = torch.cat((output[:, :, :, -2:], output, output[:, :, :, 0:3]), dim=3)

    # # Padding y axis due to periodic boundary condition
    # # shape: [t, c, h, w]
    # output = torch.cat((output[:, :, -2:, :], output, output[:, :, 0:3, :]), dim=2)

    # get physics loss
    # mse_loss = nn.MSELoss()
    mse_loss = nn.L1Loss(reduction='none')
    # output = output *3000    # to enforce the original CNN output is between 0 to 1
#     f_u, f_v = loss_func.get_phy_Loss(output)
    f_p, P_out_vec = loss_func.get_phy_Loss(output, delta_t)
#     loss =  mse_loss(f_u, torch.zeros_like(f_u).cuda()) + mse_loss(f_v, torch.zeros_like(f_v).cuda()) 
    loss =  mse_loss(f_p, torch.zeros_like(f_p).cuda()).detach()     # get the abs loss tensor [1000, 1, 8192]
    lmin, lmax = torch.min(loss.view(loss.shape[0], -1), dim=1)[0], torch.max(loss.view(loss.shape[0], -1), dim=1)[0]
    lmin, lmax = lmin.reshape(loss.shape[0], 1, 1).expand(loss.shape), \
                       lmax.reshape(loss.shape[0], 1, 1).expand(loss.shape)
    
    weights = 2.0 * (loss - lmin) / (lmax - lmin)    # w = a + b* (loss - min(loss))/(max(loss) - min(loss))
    
    w_loss = torch.mean(torch.abs(weights * (f_p - torch.zeros_like(f_p).cuda())))
    
    return w_loss