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
#     I         = torch.argwhere(x > Xref[-1]);
#     y[I]      = Yref[-1]
#     return torch.tensor(y, dtype=torch.float32)
def compute_krw(Sw):
    return (Sw>0.8)*1.0+(Sw<0.2)*0.0+((Sw>=0.2)&(Sw<=0.8))*torch.nan_to_num(1.0*((Sw-0.2)/(1-0.2-0.2))**2)
    # return (Sw>0.75)*1.0+(Sw<=0.75)*torch.nan_to_num(4.1585*(Sw**3)-1.5732*(Sw**2)+0.1741*Sw-0.002)

def compute_kro(Sw):
    return (Sw<0.2)*1.0+(Sw>0.8)*0.0+((Sw>=0.2)&(Sw<=0.8))*torch.nan_to_num(1.0*((1-Sw-0.2)/(1-0.2-0.2))**3)
    # return (Sw<0.2)*1.0+(Sw>0.8)*0.0+((Sw>=0.2)&(Sw<=0.8))*torch.nan_to_num(2.3474*(Sw**2)-3.5361*Sw+1.3295)

def compute_relperm(Properties,Sw):
    So = 1.0-Sw;
    krw_init = 0.6
    kro_init = 0.9
    # %============================
    # krw_table   = Properties.krw_table;
#     krw         = property_interp(krw_table[:,0],krw_table[:,1],Sw);
    # krw = 4.1585*(Sw**3)-1.5732*(Sw**2)+0.1741*Sw-0.002
    krw =krw_init * compute_krw(Sw)
#     krw = Sw.apply_(lambda x: (4.1585*(x**3)-1.5732*(x**2)+0.1741*x-0.002) if x<0.75 else 1.0)

    # %-----------------------------------------------------------------
    # kro_table   = Properties.kro_table;
#     kro         = property_interp(kro_table[:,0],kro_table[:,1],So);
    # kro = 2.3474*(Sw**2)-3.5361*Sw+1.3295
    kro = kro_init * compute_kro(Sw)
#     kro = Sw.apply_(lambda x: (1.0) if x<0.1 else ((2.3474*(x**2)-3.5361*x+1.3295) if x<0.8 else 0.0))
    
    return kro, krw

# def compute_derivative(f,x):
#     if np.min(x):
#         delta = 1.0e-3*np.min(x)
#     else:
#         delta = 1.0e-3
def compute_derivative(f,x):
    if torch.min(x):
        delta = 1.0e-3*torch.min(x)
    else:
        delta = 1.0e-3
        
    fo_p, fw_p = f(x+delta)
    fo_n, fw_n = f(x-delta)
    dfo             = (fo_p-fo_n)/(2.0*delta);
    dfw             = (fw_p-fw_n)/(2.0*delta);
    return dfo , dfw

def compute_relative_permeability(Rock,Sw):
    kro,krw       = compute_relperm(Rock,Sw)
    
    myfunct = lambda x: compute_relperm(Rock,x)
    dkro, dkrw = compute_derivative(myfunct,Sw)
    
    return kro, krw, dkro, dkrw

def update_rock_property(Rock,FullSolution):
    P  = FullSolution.Pcurrent;
    Sw = FullSolution.Swcurrent;
#     type = Discretization.Geom_AvgType; 
    # %--------------------
    kro,krw,dkro,dkrw   = compute_relative_permeability(Rock,Sw)
    
#     %=======================
    Rock.kro  = kro
    Rock.krw  = krw
    Rock.dkro = dkro
    Rock.dkrw = dkrw
    
    return Rock