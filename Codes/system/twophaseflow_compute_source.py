import scipy.io as scio
import torch
import numpy as np

def compute_source(Discretization, Mobility, Wells):
    Nt          = Discretization.Nt
    fro         = Mobility.fro
    frw         = Mobility.frw
    Qinj        = Wells.Qinj;
    Pwf_pro     = Wells.Pwf_pro;
    J_pro       = Wells.J_pro
    Qinj_ind    = Wells.Qinj_ind
    Qpro_ind    = Wells.Qpro_ind
    Npro        = Wells.Npro
    tsteps = Discretization.Tsteps

#     %================================
    Qo      = torch.zeros((Nt,tsteps), dtype=torch.float32).cuda()
    Qw      = torch.zeros((Nt,tsteps), dtype=torch.float32).cuda()
    qo      = torch.zeros((Nt,tsteps), dtype=torch.float32).cuda()
    qw      = torch.zeros((Nt,tsteps), dtype=torch.float32).cuda()
    
#     zero_mat = torch.zeros((Nt, Nt), dtype=torch.float32)
    

#     % Producer wells
#     %---------------------
    # print(Pwf_pro.size())
    # if tsteps>1:
    Qo[Qpro_ind]    =  torch.unsqueeze(J_pro, 1)*fro[Qpro_ind]
    Qw[Qpro_ind]    =  torch.unsqueeze(J_pro, 1)*frw[Qpro_ind]

    qo[Qpro_ind]    =  Pwf_pro*torch.unsqueeze(J_pro, 1)*fro[Qpro_ind]
    qw[Qpro_ind]    =  Pwf_pro*torch.unsqueeze(J_pro, 1)*frw[Qpro_ind]

            

#     %---------------------
#     % Injector wells
#     %---------------------
    qw[Qinj_ind]    = qw[Qinj_ind] + Qinj

    # print('producer:', Qo)
    
    # Qo_lst = []
    # Qw_lst = []
    # qo_lst =[]
    # qw_lst =[]
    
    # for k in range(tsteps):
    diagonal_Qo = torch.diag(Qo.squeeze())
    Qo_lst = diagonal_Qo
    diagonal_Qw = torch.diag(Qw.squeeze())
    Qw_lst = diagonal_Qw
#     %=========================
    Q_o = torch.hstack((Qo_lst, torch.zeros_like(Qo_lst)))
    Q_w = torch.hstack((Qw_lst, torch.zeros_like(Qw_lst)))
    Q   = torch.vstack((Q_o, Q_w))
    # for k in range(tsteps):
    diagonal_qo = qo
    qo_lst = diagonal_qo
    diagonal_qw = qw
    qw_lst = diagonal_qw      
#     %=========================
    q   = torch.vstack((qo_lst, qw_lst))

#     %=========================================
    Wells.Qo     = Qo_lst
    Wells.Qw     = Qw_lst
    Wells.Q     = Q
    Wells.qo     = qo_lst
    Wells.qw     = qw_lst
    Wells.q      = q
    
    return Wells