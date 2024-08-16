import scipy.io as scio
import torch
import numpy as np

def harmonic_average(x1,x2):
    y   = 1/(1/x1+1/x2)
    return y

def property_avg(Discretization, FullSolution, Prop_x, Prop_y):
#         %=================================
#     % Getting the parameters from data structures
#     %=================================
    Nx      = Discretization.Nx;
    Ny      = Discretization.Ny;
    Nz      = Discretization.Nz;
    Nt      = Discretization.Nt;
    x_neg   = Discretization.x_neg;
    x_pos   = Discretization.x_pos;
    y_neg   = Discretization.y_neg;
    y_pos   = Discretization.y_pos;
    z_neg   = Discretization.z_neg;
    z_pos   = Discretization.z_pos;


    P       = FullSolution.Pcurrent;
    #     %------------------
    AvgProp_x  = torch.zeros((Nx+1,Ny,Nz)).cuda()
    AvgProp_y  = torch.zeros((Nx,Ny+1,Nz)).cuda()

#     Prop_x = Bc*Ax*Kx/Dx
#     Prop_y = Bc*Ay*Ky/Dy

    # print('perm x', Prop_x)
    x_neg_sub, x_pos_sub = np.unravel_index(x_neg, (Nx, Ny)), np.unravel_index(x_pos, (Nx, Ny))
    y_neg_sub, y_pos_sub = np.unravel_index(y_neg, (Nx, Ny)), np.unravel_index(y_pos, (Nx, Ny))
    # print(x_neg_sub)

    AvgProp_x[1:Nx,:,:]  = harmonic_average(Prop_x[x_neg_sub], Prop_x[x_pos_sub]);
    AvgProp_y[:,1:Ny,:]  = harmonic_average(Prop_y[y_neg_sub], Prop_y[y_pos_sub]);

    # print(AvgProp_x.size())
    Prop_xpos = torch.reshape(AvgProp_x[:Nx,:,:],(Nt,1)); 
    Prop_xneg = torch.reshape(AvgProp_x[1:,:,:],(Nt,1));
    Prop_ypos = torch.reshape(AvgProp_y[:,:Ny,:],(Nt,1)); 
    Prop_yneg = torch.reshape(AvgProp_y[:,1:,:],(Nt,1));

    # print(Prop_yneg)
    return Prop_xpos, Prop_xneg, Prop_ypos, Prop_yneg

def twophaseflow_geo_avg(Discretization, Rock, FullSolution, Constants):
    Dx   = Discretization.Dx;
    Dy   = Discretization.Dy;
    Dz   = Discretization.Dz;
    Ax   = Discretization.Ax;
    Ay   = Discretization.Ay;
    Az   = Discretization.Az;
    type = Discretization.Geom_AvgType; 
    # %--------------------
    Bc  = Constants.Bc;
    # %---------------------
    Kx = Rock.Kx;
    Ky = Rock.Ky;
    # %---------------------
    Prop_xpos, Prop_xneg, Prop_ypos, Prop_yneg = property_avg(Discretization, FullSolution, Bc*Ax*Kx/Dx, Bc*Ay*Ky/Dy)

    # %----------------------
    Rock.K_xpos   = Prop_xpos;
    Rock.K_xneg   = Prop_xneg;
    Rock.K_ypos   = Prop_ypos;
    Rock.K_yneg   = Prop_yneg;
    return Rock