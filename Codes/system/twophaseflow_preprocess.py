import scipy.io as scio
import torch
import numpy as np
from math import ceil

def compute_productivity_index(Kx,Ky,Dx,Dy,rw,Skin,h,Bc):
    Kh = torch.sqrt(Kx*Ky) 
    req =  0.28*torch.sqrt(torch.sqrt(Ky/Kx)*(Dx**2) + torch.sqrt(Kx/Ky)*(Dy**2))   
    req = req/((Ky/Kx)**0.25 + (Kx/Ky)**0.25) # [ft]
    J = 2*torch.tensor(np.pi)*Bc*Kh*h    
    J = J/( torch.log(req/rw) + Skin )     # [STB/day/psi]
    return J

def pre_processing(Geometry, Discretization, Rock, Wells, Conditions, Constants, FullSolution):
    Lx = Geometry.Lx;
    Ly = Geometry.Ly;
    Lz = Geometry.Lz;

    tsteps = Discretization.Tsteps
    Nx = Discretization.Nx;
    Ny = Discretization.Ny;
    Nz = Discretization.Nz;
    Nt = Discretization.Nt;

    rw          = Wells.rw;
    Skin        = Wells.Skin;
    Qpro_ind    = Wells.Qpro_ind;
    Qinj_ind    = Wells.Qinj_ind;
    Qpro_ji    = Wells.Qpro_ji;
    Qinj_ji    = Wells.Qinj_ji;

    Pi      = Conditions.Pi;
    Swi     = Conditions.Swi;

    Bc      = Constants.Bc;
    Dx      = Lx/Nx;   # [ft]
    Dy      = Ly/Ny;   # [ft]
    Dz      = Lz/Nz;   # [ft]  
    Ax      = Dy*Dz;
    Ay      = Dx*Dz;
    Az      = Dx*Dy;
    Vb      = Dx*Dy*Dz; # [ft3]

    kM      = torch.reshape(torch.arange(Nt),(Nx,Ny,Nz))
    x_neg   = kM[:Nx-1,:,:]
    x_pos   = kM[1:,:,:]
    y_neg   = kM[:,:Ny-1,:]
    y_pos   = kM[:,1:,:]
    z_neg   = kM[:,:,:Nz-1]
    z_pos   = kM[:,:,1:]
    # print(x_neg.size())
    # y_neg   = kM[:Nx-1,:,:]
    # y_pos   = kM[1:,:,:]
    # x_neg   = kM[:,:Ny-1,:]
    # x_pos   = kM[:,1:,:]
    # z_neg   = kM[:,:,:Nz-1]
    # z_pos   = kM[:,:,1:]

    P       = Pi*np.ones((Nt,tsteps))   # [psi]
    Sw      = Swi*np.ones((Nt,tsteps))  # [-]

    # %-----------------
    Npro    = len(Qpro_ind)       # Number of producer wells
    Ninj    = len(Qinj_ind)      # Number of producer wells

    # %-----------------
    Kx_pro  = Rock.Kx[np.array(Qpro_ji)]
    Ky_pro  = Rock.Ky[np.array(Qpro_ji)]


    J_pro   = compute_productivity_index(Kx_pro,Ky_pro,Dx,Dy,rw,Skin,Dz,Bc); # [BPD/psi]
    # print(J_pro)
    # 
    Discretization.Dx       = Dx;
    Discretization.Dy       = Dy;
    Discretization.Dz       = Dz;
    Discretization.Ax       = Ax;
    Discretization.Ay       = Ay;
    Discretization.Az       = Az;
    Discretization.Vb       = Vb;
    Discretization.x_neg    = x_neg;
    Discretization.x_pos    = x_pos;
    Discretization.y_neg    = y_neg;
    Discretization.y_pos    = y_pos;
    Discretization.z_neg    = z_neg;
    Discretization.z_pos    = z_pos;
    Discretization.n        = 0;
    Discretization.Fluid_AvgType    = 'Upstream';
    Discretization.Geom_AvgType     = 'Harmonic';
    # %==============================
    Wells.Npro = Npro
    Wells.Ninj = Ninj
    # %-----------------
    Wells.J_pro = J_pro
    # %-------------------
    # Wells.fro   = torch.zeros((Npro,1))
    # Wells.frw   = torch.zeros((Npro,1))
    # %================================

    
    FullSolution.Pcurrent  = torch.tensor(P, dtype=torch.float32)
    # %-----------------
    FullSolution.Swcurrent = torch.tensor(Sw, dtype=torch.float32)
    return Discretization, Wells, FullSolution