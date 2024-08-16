import scipy.io as scio
import torch
import numpy as np
# from scipy.interpolate import interp1d

class struct:
    pass

def compute_accumulation(Discretization, Constants, Fluid, Rock, FullSolution):

#     %=============================
    Vb      = Discretization.Vb;
    Dt      = Discretization.Dt;
    Nt      = Discretization.Nt;
    Ac      = Constants.Ac;
    bo      = Fluid.bo;
    bw      = Fluid.bw;
    Cw      = Fluid.Cw
    Co      = Fluid.Co
    Cr      = Fluid.Cr
    dbo     = Fluid.dbo;
    dbw     = Fluid.dbw;
    poro    = Rock.poro;
    tsteps = Discretization.Tsteps
    
#     Coeff   = Vb/(Ac*Dt);
    Coeff   = Vb/Ac;

#         %=============================
    Sw      = FullSolution.Swcurrent;
    So      = 1-Sw;
    
    # print(bo)
    #         %================================================================================================================
    Cwp     =  Coeff * ( poro*bw*Cr + poro*bw*Cw )* Sw;  # Consider bw.*dphi, where dphi change in the porousity wrt pressure
    Cop     =  Coeff * ( poro*bo*Cr + poro*bo*Co )* So;  # Consider bo.*dphi, where dphi change in the porousity wrt pressure
    #     %-------------------------------------------------------------------------------------------------------------------------------------
    Cws     =  Coeff * ( poro*bw  - 0 );       # Consider Sw.*phi.*dBw.*dPc, where dPc is the change in the capillary pressure wrt pressure
    Cos     =  -Coeff * ( poro*bo      );
       
    # print(Cop.size())
    # for k in range(tsteps):
    d11 = torch.diag(Cop.squeeze())
    d12 = torch.diag(Cos.squeeze())
    d21 = torch.diag(Cwp.squeeze())
    d22 = torch.diag(Cws.squeeze())

    Acc_op = d11
    Acc_os = d12
    Acc_wp = d21
    Acc_ws = d22
    
    acc_o = torch.hstack((d11, d12))
    acc_w = torch.hstack((d21, d22))
    acc   = torch.vstack((acc_o, acc_w))
    Acc = acc
#     d11     = torch.diag(Cop.squeeze())
#     d12     = torch.diag(Cos.squeeze())
#     d21     = torch.diag(Cwp.squeeze())
#     d22     = torch.diag(Cws.squeeze())

    Accumulation = struct()
#     %=============================
    Accumulation.Acc    = Acc
    Accumulation.Acc_op    = Acc_op
    Accumulation.Acc_os    = Acc_os
    Accumulation.Acc_wp    = Acc_wp
    Accumulation.Acc_ws    = Acc_ws
#     %=============================
    
    return Accumulation