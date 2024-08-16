import scipy.io as scio
import torch
import numpy as np
# from scipy.sparse import spdiags, csr_matrix

def compute_mob(kr,b,U):
    y   = kr*b/U
    return y

def compute_dmob_dP(kr,b,U,db,dU):
    dydp    = kr*(db*U-dU*b)/(U**2)
    return dydp

def compute_dmob_dSw(dkr,b,U):
    dyds    = dkr*b/U
    return dyds


class struct:
    pass

def compute_mobility(Fluid,Rock):
    bo      = Fluid.bo
    bw      = Fluid.bw
    # dbo     = Fluid.dbo;
    # dbw     = Fluid.dbw;
    Uo      = Fluid.Uo
    Uw      = Fluid.Uw
    # dUo     = Fluid.dUo;
    # dUw     = Fluid.dUw;
    kro     = Rock.kro
    krw     = Rock.krw
    # dkro    = Rock.dkro;
    # dkrw    = Rock.dkrw;
#     type = Discretization.Geom_AvgType; 
    # %--------------------
    fro     = compute_mob(kro,bo,Uo)
    frw     = compute_mob(krw,bw,Uw)
    
    # print(fro)
#     %=================================================
    # dfrodP      = compute_dmob_dP(kro,bo,Uo,dbo,dUo);
    # dfrwdP      = compute_dmob_dP(krw,bw,Uw,dbw,dUw);
    # dfrodSw     = compute_dmob_dSw(dkro,bo,Uo);
    # dfrwdSw     = compute_dmob_dSw(dkrw,bw,Uw);    
#     %=================================================
    Mobility = struct()
    Mobility.fro        = fro
    Mobility.frw        = frw
    Mobility.frt        = fro +frw
    # Mobility.dfrodP     = dfrodP;
    # Mobility.dfrwdP     = dfrwdP;
    # Mobility.dfrodSw    = dfrodSw;
    # Mobility.dfrwdSw    = dfrwdSw;
    
    return Mobility