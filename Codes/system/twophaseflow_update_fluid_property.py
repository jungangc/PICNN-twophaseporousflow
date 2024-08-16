import scipy.io as scio
import torch
import numpy as np
# from scipy.interpolate import interp1d
# from scipy.interpolate import pchip_interpolate

# def property_interp(Xref,Yref,x):
#     y         = pchip_interpolate(Xref,Yref,x);
# #     I         = np.argwhere(x < np.array(Xref[0]));
#     I         = torch.argwhere(x < Xref[0]);
#     y[I]      = Yref[0];
# #     I         = np.argwhere(x > np.array(Xref[-1]));
#     I         = torch.argwhere(x > Xref[-1]);
# #     y(I)      = Yref(end);
#     y[I]      = Yref[-1]
#     return torch.tensor(y, dtype=torch.float32)

def compute_FVF(Fluid,P):
    P_ref   = Fluid.P_ref;
    Bw_ref  = Fluid.Bw_ref;
    Cw      = Fluid.Cw;
    X       = Cw*(P - P_ref);
    # Bw      = Bw_ref/(1 + X + X**2);
    Bw = Bw_ref*torch.ones_like(P)
    
#         %============================
    # Bo_table= Fluid.Bo_table;
#     Bo      = property_interp_Bo(Bo_table[:,0],Bo_table[:,1],P);    # [rbbl/stb]

    Bo = -4e-10*(P**2)-8.0e-6*P+1.1966
    # Bo = 1.2*torch.ones_like(P)
    return Bw, Bo

def compute_FVF_inv(Fluid,P):
    Bw, Bo = compute_FVF(Fluid,P)
    bo      = 1.0/Bo;
    bw      = 1.0/Bw;
    return bo, bw

# def compute_derivative(f,x):
#     if np.min(x):
#         delta = 1.0e-3*np.min(x)
#     else:
#         delta = 1.0e-3
def compute_derivative(f,x):
    # print('x: ',x)
    if torch.min(x):
        delta = 1.0e-3*torch.min(x)
    else:
        delta = 1.0e-3
        
    fo_p, fw_p = f(x+delta)
    fo_n, fw_n = f(x-delta)
    dfo             = (fo_p-fo_n)/(2.0*delta);
    dfw             = (fw_p-fw_n)/(2.0*delta);
    return dfo , dfw

def compute_volumetric_factor(Fluid,P):
    bo, bw = compute_FVF_inv(Fluid,P)
    
#     myfunct = lambda x: compute_derivative(compute_FVF_inv(Fluid,x),x)
    myfunct = lambda x: compute_FVF_inv(Fluid,x)
    
    dbo, dbw = compute_derivative(myfunct,P)
#     dbo, dbw = 
    return bo, bw, dbo, dbw

def compute_mu(Properties,P):

    # Uw_table    = Properties.Uw_table;
#     Uw          = property_interp(Uw_table[:,0],Uw_table[:,1],P); # [cp]
    # Uw = 1.0* torch.ones_like(P)

    Uw = 1.0*torch.ones_like(P)
#     %============================
    # Uo_table    = Properties.Uo_table;
#     Uo          = property_interp(Uo_table[:,0],Uo_table[:,1],P); # [cp]
    Uo = 1.2*torch.ones_like(P)
    # Uo = 2.0*torch.ones_like(P)
    return Uo, Uw

def compute_viscosity(Fluid,P):
    Uo, Uw = compute_mu(Fluid,P)
    
#     myfunct = lambda x: compute_derivative(compute_FVF_inv(Fluid,x),x)
    myfunct = lambda x: compute_mu(Fluid,x)
    
    dUo, dUw = compute_derivative(myfunct,P)
#     dbo, dbw = 
    return Uo, Uw, dUo, dUw

def update_fluid_property(Fluid,FullSolution):
    P  = FullSolution.Pcurrent;
    # Sw = FullSolution.Swcurrent;
    # print(P)
#     type = Discretization.Geom_AvgType; 
    # %--------------------
    bo, bw, dbo, dbw   = compute_volumetric_factor(Fluid,P)
    
    Uo,Uw,dUo,dUw   = compute_viscosity(Fluid,P)
    
    Fluid.bo = bo
    Fluid.bw   = bw;
    Fluid.dbo = dbo
    Fluid.dbw   = dbw; 
    Fluid.Uo   = Uo;
    Fluid.Uw   = Uw;
    # Fluid.dUo  = dUo;
    # Fluid.dUw  = dUw;
    
    return Fluid