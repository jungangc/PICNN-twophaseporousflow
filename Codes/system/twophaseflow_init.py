import scipy.io as scio
import torch
import numpy as np
from math import ceil

class struct:
    pass

def INITIALIZE(Qinj_schedule, BHP_schedule, Perm, tsteps):
    Nx =64
    Ny =64
    Nz =1
    Nt = Nx*Ny*Nz
    Geometry = struct()
    Geometry.Lx = 3.2808*Nx*20
    Geometry.Ly = 3.2808*Ny*20    # [ft]
    Geometry.Lz = 3.2808*Nz*20    # [ft]

    #### discretization
    Discretization = struct()
    Discretization.Nx   = Nx   
    Discretization.Ny   = Ny  
    Discretization.Nz   = Nz  
    Discretization.Nt   = Nt
    Discretization.n    = 1
    Discretization.Dt   = 1.0    # [Day]
    Discretization.Tsim = 200    # [Day]
    Discretization.Tsteps = tsteps # [testing timesteps]

    #### rock
    Rock = struct()
    # TRUE_PERM = scio.loadmat('/scratch/user/jungangc/PICNN-2phase/PICRNN-2phaseflow-constBHP-64by64-200by05/Codes/system/TRUE_PERM_64by220.mat')
    # TRUE_PERM = torch.tensor(TRUE_PERM['TRUE_PERM'], dtype=torch.float32).cuda()
    
    # TRUE_PERM = torch.tensor(0.05*np.ones((Nx, Ny)), dtype=torch.float32).cuda()
    TRUE_PERM = Perm
    Rock.Kx = TRUE_PERM       # [Darcy]
    Rock.Ky = Rock.Kx
    Rock.poro     = 0.20;    # [fraction]
    # krw_table  = np.array([[0,   0],
    #     [0.1 , 0.0001],
    #     [0.2 , 0.0025],
    #     [0.3 , 0.023],
    #     [0.4 , 0.0842],
    #     [0.5 , 0.2115],
    #     [0.6 , 0.4319],
    #     [0.7 , 0.774],
    #     [0.75, 0.9999],
    #     [0.8 , 1],
    #     [0.9 , 1],
    #     [1  ,  1]])
    Rock.krw_table = torch.tensor(np.array([[0,   0],
        [0.1 , 0.0001],
        [0.2 , 0.0025],
        [0.3 , 0.023],
        [0.4 , 0.0842],
        [0.5 , 0.2115],
        [0.6 , 0.4319],
        [0.7 , 0.774],
        [0.75, 0.9999],
        [0.8 , 1],
        [0.9 , 1],
        [1  ,  1]]), dtype=torch.float32);  
    # kro_table  = np.array([[0,   0],
    #     [.1,  0], 
    #    [ .2 ,0],
    #     [.25,0.0001],
    #     [.3, 0.0059],
    #     [.4, 0.0533],
    #     [.5 ,0.1479],
    #     [.6 ,0.2899],
    #     [.7 ,0.4793],
    #     [.8 ,0.716],
    #     [.9 ,1],
    #    [ 1 ,  1]])                   
    Rock.kro_table = torch.tensor(np.array([[0,   0],
        [.1,  0], 
       [ .2 ,0],
        [.25,0.0001],
        [.3, 0.0059],
        [.4, 0.0533],
        [.5 ,0.1479],
        [.6 ,0.2899],
        [.7 ,0.4793],
        [.8 ,0.716],
        [.9 ,1],
       [ 1 ,  1]]) , dtype=torch.float32)


    #### fluid 
    Fluid = struct()
    Fluid.P_ref    = 2800     # [psi]
    Fluid.Cw       = 3e-6     # [psi-1]
    Fluid.Co       = 1e-5     # [psi-1]
    Fluid.Cr       = 3e-6    # [psi-1]
    Uw_table =np.array([[400. ,   1.0], 
                        [2800. ,   1.0],
                        [5600.  ,  1.0]])   # [cp]
    Fluid.Uw_table = torch.tensor(Uw_table, dtype=torch.float32);       
    Uo_table = np.array([[400,  1.00],
        [2800 , 1.00],
        [5600 , 1.00]]) 
    Fluid.Uo_table = torch.tensor(Uo_table, dtype=torch.float32);    
    Bo_table = np.array([[400,  1.20],
        [2800 , 1.20],
        [5600 , 1.20]])
    Fluid.Bo_table = torch.tensor(Bo_table, dtype=torch.float32);    
    Fluid.Bw_ref   = 1.0         # [RB/STB] 

    Wells = struct()
    Wells.rw        = 0.2915     # [ft]
    Wells.Skin      = 0 
    # Qinj  = np.array([[100]]) # [STB/day]
    # Wells.Qinj      = torch.tensor(Qinj, dtype=torch.float32).repeat(1,Discretization.Tsteps).cuda()
    
    # Pwf_pro  = np.array([[2500],
    #                    [2500]]) # [psi]
    # Wells.Pwf_pro   = torch.tensor(Pwf_pro, dtype=torch.float32).repeat(1, Discretization.Tsteps).cuda() 
    Wells.Qinj  = Qinj_schedule
    Wells.Pwf_pro   = BHP_schedule
    
    Qinj_ji = np.array([[ceil(Nx/2)-1,ceil(Ny/2)-1],
                       [12, 49],
                       [49, 12]])
    Wells.Qinj_ji   = torch.tensor(Qinj_ji, dtype=torch.int32);
    Qpro_ji = np.array([[49, 49],
                        [12, 12]])
    Wells.Qpro_ji   = torch.tensor(Qpro_ji, dtype=torch.int32);
    
    Wells.Qinj_ind = np.ravel_multi_index(Qinj_ji.T,(Nx, Ny))
    Wells.Qpro_ind = np.ravel_multi_index(Qpro_ji.T,(Nx, Ny))
    ### Initial conditions
    # %=================================
    Conditions = struct()
    Conditions.Pi  = 3000; # [psi]
    Conditions.Swi = 0.2;    # [-]
    Conditions.Soi = 0.8;    # [-]
    # %=================================
    # % f) Constants (conversion factors) 
    Constants = struct()
    # %=================================
    Constants.Bc = 1.127;
    Constants.Ac = 5.614583;
    
    FullSolution = struct()
    
#     Mobility = struct()
    
    return Geometry, Discretization, Fluid, Rock, Wells, Conditions, Constants, FullSolution