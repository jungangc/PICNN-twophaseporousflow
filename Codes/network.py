import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
import time
import os
from torch.nn.utils import weight_norm


class encoder_block_p(nn.Module):
    ''' encoder with CNN '''
    def __init__(self):
        
        super(encoder_block_p, self).__init__()

        self.conv =  nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=16, kernel_size=4, stride=2, padding=1, bias=True, padding_mode='circular'),
            nn.ReLU(),
            # nn.Tanh(),
            # nn.SiLU(),
#             nn.BatchNorm2d(4), 
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1, bias=True, padding_mode='circular'),
            nn.ReLU(), 
            # nn.Tanh(),
            # nn.SiLU(),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1, bias=True, padding_mode='circular'),
            nn.ReLU(), 
            # nn.Tanh(),
            # nn.SiLU(),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=True, padding_mode='circular'),
            nn.ReLU(),
            # nn.Tanh(),
            # nn.SiLU(),
            
            nn.Flatten()
        )
        

    def forward(self, x):
        x = self.conv(x)
#         x = self.act(x)
        # x= self.Dropout(x)
        return x

class encoder_block_sw(nn.Module):
    ''' encoder with CNN '''
    def __init__(self):
        
        super(encoder_block_sw, self).__init__()

        self.conv =  nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=16, kernel_size=4, stride=2, padding=1, bias=True, padding_mode='circular'),
            nn.ReLU(),
            # nn.Tanh(),
            # nn.SiLU(),
#             nn.BatchNorm2d(4), 
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1, bias=True, padding_mode='circular'),
            nn.ReLU(), 
            # nn.Tanh(),
            # nn.SiLU(),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1, bias=True, padding_mode='circular'),
            nn.ReLU(), 
            # nn.Tanh(),
            # nn.SiLU(),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=True, padding_mode='circular'),
            nn.ReLU(),
            # nn.Tanh(),
            # nn.SiLU(),
            
            nn.Flatten()
        )
        

    def forward(self, x):
        x = self.conv(x)
#         x = self.act(x)
        # x= self.Dropout(x)
        return x

class decoder_block_p(nn.Module):
    ''' decoder with CNN '''
    def __init__(self):
        
        super(decoder_block_p, self).__init__()
        
        self.deconv =  nn.Sequential(
            
            nn.Unflatten(1, (128, 4, 4)),
            
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(),
            # nn.Tanh(),
            # nn.SiLU(),
            
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(), 
            # nn.Tanh(),
            # nn.SiLU(),
            
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(), 
            # nn.Tanh(),
            # nn.SiLU(),
            
            nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=4, stride=2, padding=1, bias=True),
            # nn.ReLU()
            # nn.Tanh()
            # nn.ELU(),
            nn.Sigmoid()
        )

    def forward(self, x):
#         x = self.upsampling(x)
        x = self.deconv(x)
        return x

class decoder_block_sw(nn.Module):
    ''' decoder with CNN '''
    def __init__(self):
        
        super(decoder_block_sw, self).__init__()
        
        self.deconv =  nn.Sequential(
            
            nn.Unflatten(1, (128, 4, 4)),
            
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(),
            # nn.Tanh(),
            # nn.SiLU(),
            
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(), 
            # nn.Tanh(),
            # nn.SiLU(),
            
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(), 
            # nn.Tanh(),
            # nn.SiLU(),
            
            nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=4, stride=2, padding=1, bias=True),
            # nn.ReLU()
            # nn.Tanh()
            # nn.ELU()
            nn.Sigmoid()
        )

    def forward(self, x):
#         x = self.upsampling(x)
        x = self.deconv(x)
        return x


class source_encoder_bhp(nn.Module):
    ''' encoder source term with CNN '''
    def __init__(self):
        
        super(source_encoder_bhp, self).__init__()

        self.souce_conv =  nn.Sequential(
            nn.PixelUnshuffle(downscale_factor= 8), 
            nn.Conv2d(in_channels=2*8*8, out_channels=128, kernel_size=4, stride=2, padding=1, bias=True, padding_mode='circular'),
            nn.Tanh(), 
            # nn.ReLU(),
            
            nn.Flatten()
        )

    def forward(self, x):
        x = self.souce_conv(x)
        return x

    
class source_encoder_qinj(nn.Module):
    ''' encoder source term with CNN '''
    def __init__(self):
        
        super(source_encoder_qinj, self).__init__()

        self.souce_conv =  nn.Sequential(
            nn.PixelUnshuffle(downscale_factor= 8), 
            nn.Conv2d(in_channels=2*8*8, out_channels=128, kernel_size=4, stride=2, padding=1, bias=True, padding_mode='circular'),
            nn.Tanh(), 
            # nn.ReLU(),
            
            nn.Flatten()
        )

    def forward(self, x):
        x = self.souce_conv(x)
        return x