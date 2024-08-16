import scipy.io as scio
import torch
import numpy as np
from scipy.sparse import spdiags, csr_matrix

class struct:
    pass

def upstream( x1, x2, P1, P2):
    I       = torch.argwhere(P1<P2)
    x1[I]   = x2[I]
    y       = x1
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
    y_pos   = Discretization.y_pos
    z_neg   = Discretization.z_neg
    z_pos   = Discretization.z_pos
    tsteps = Discretization.Tsteps

    P       = FullSolution.Pcurrent;
    #     %------------------
    AvgProp_x  = torch.zeros((Nx+1,Ny,tsteps)).cuda()
    AvgProp_y  = torch.zeros((Nx,Ny+1,tsteps)).cuda()

#     x_neg_sub, x_pos_sub = np.unravel_index(x_neg, (Nx, Ny)), np.unravel_index(x_pos, (Nx, Ny))
#     y_neg_sub, y_pos_sub = np.unravel_index(y_neg, (Nx, Ny)), np.unravel_index(y_pos, (Nx, Ny))
    x_neg_vec, x_pos_vec = torch.reshape(x_neg, (Nx*Ny-Ny, 1)), torch.reshape(x_pos, (Nx*Ny-Ny, 1))
    y_neg_vec, y_pos_vec = torch.reshape(y_neg, (Nx*Ny-Nx, 1)), torch.reshape(y_pos, (Nx*Ny-Nx, 1))

    AvgProp_x[1:Nx,:,:]  = upstream(Prop_x[x_neg_vec], Prop_x[x_pos_vec], P[x_neg_vec], P[x_pos_vec]).reshape((Nx-1, Ny, tsteps));
    AvgProp_y[:,1:Ny,:]  = upstream(Prop_y[y_neg_vec], Prop_y[y_pos_vec], P[y_neg_vec], P[y_pos_vec]).reshape((Nx, Ny-1, tsteps));

    Prop_xpos = torch.reshape(AvgProp_x[:Nx,:,:],(Nt,tsteps)); 
    Prop_xneg = torch.reshape(AvgProp_x[1:,:,:],(Nt,tsteps));
    Prop_ypos = torch.reshape(AvgProp_y[:,:Ny,:],(Nt,tsteps)); 
    Prop_yneg = torch.reshape(AvgProp_y[:,1:,:],(Nt,tsteps));

    # print(Prop_ypos)
    return Prop_xpos, Prop_xneg, Prop_ypos, Prop_yneg



def compute_transmisibility(Discretization, Rock, Mobility, FullSolution):
#     %============================================
#     % Getting the parameters from data structures
#     %============================================
    Nx      = Discretization.Nx;
    Ny      = Discretization.Ny;
    Nt      = Discretization.Nt;
    Fluid_AvgType="Upstream";
    K_xpos  = Rock.K_xpos;
    K_xneg  = Rock.K_xneg;
    K_ypos  = Rock.K_ypos;
    K_yneg  = Rock.K_yneg;
    tsteps = Discretization.Tsteps
    
#     %=====================================================================================================
    fro     = Mobility.fro;
    frw     = Mobility.frw;

    fro_xpos, fro_xneg, fro_ypos, fro_yneg  = property_avg(Discretization,FullSolution, fro, fro);
    frw_xpos, frw_xneg, frw_ypos, frw_yneg  = property_avg(Discretization,FullSolution, frw, frw);
    
    # print('avg rock perm: ', K_xpos)
    # print('avg mob: ', fro_xpos)
    #         % Compute components for Oil
    TN_o    = K_xpos*fro_xpos;
    TS_o    = K_xneg*fro_xneg;
    TE_o    = K_ypos*fro_ypos;
    TW_o    = K_yneg*fro_yneg;

    #     % Compute components for Water
    TN_w    = K_xpos*frw_xpos;
    TS_w    = K_xneg*frw_xneg;
    TE_w    = K_ypos*frw_ypos;
    TW_w    = K_yneg*frw_yneg;
    
    #     % Center
    TC_o    = - ( TN_o + TS_o + TE_o + TW_o )
    TC_w    = - ( TN_w + TS_w + TE_w + TW_w )

    # print(torch.diag(TS_o[Nx:].squeeze(), diagonal=-Nx).size())
#     %=================================    
    
    # print(TS_o)
    # print(TE_o)
    # print(TW_o)
    # print(TN_o)  
    # Tw = []
    # # zero_mat
    # for k in range(tsteps):
    # t   = torch.vstack((t11, t21)).to(torch.float32)
#     t11 = torch.diag(TC_o.squeeze(), diagonal=0)+ torch.diag(TS_o_vec.squeeze(), diagonal=-Nx) + torch.diag(TW_o_vec.squeeze(), diagonal=-1) + torch.diag(TE_o_vec.squeeze(), diagonal=1) + torch.diag(TN_o_vec.squeeze(), diagonal=Nx)

#     t21 = torch.diag(TC_w.squeeze(), diagonal=0)+ torch.diag(TS_w_vec.squeeze(), diagonal=-Nx) + torch.diag(TW_w_vec.squeeze(), diagonal=-1) + torch.diag(TE_w_vec.squeeze(), diagonal=1) + torch.diag(TN_w_vec.squeeze(), diagonal=Nx)
    t11 = torch.diag(TC_o.squeeze(), diagonal=0)+ torch.diag(TS_o[:-Nx].squeeze(), diagonal=-Nx) + torch.diag(TW_o[:-1].squeeze(), diagonal=-1) + torch.diag(TE_o[1:].squeeze(), diagonal=1) + torch.diag(TN_o[Nx:].squeeze(), diagonal=Nx)

    t21 = torch.diag(TC_w.squeeze(), diagonal=0)+ torch.diag(TS_w[:-Nx].squeeze(), diagonal=-Nx) + torch.diag(TW_w[:-1].squeeze(), diagonal=-1) + torch.diag(TE_w[1:].squeeze(), diagonal=1) + torch.diag(TN_w[Nx:].squeeze(), diagonal=Nx)
    
    t12 = torch.zeros_like(t11)
    t22 = torch.zeros_like(t21)
    t1 = torch.hstack((t11, t12))
    t2 = torch.hstack((t21, t22))
    t = torch.vstack((t1, t2))

    To = t11
    Tw = t21
    T = t
        
#     %=================================
#     % Filling out the data structures
#     %=================================
    Transmisibility = struct()
#     %--------------------------------
    Transmisibility.To    = To
    Transmisibility.Tw    = Tw
    Transmisibility.T     = T
    
    return Transmisibility