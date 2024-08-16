import torch
import torch.nn.functional as F
import torch.nn as nn

import numpy as np
from torch.nn.utils import weight_norm

from network import encoder_block_p, encoder_block_sw, decoder_block_p, decoder_block_sw, source_encoder_bhp, source_encoder_qinj

# generalized version
def initialize_weights(module):
    ''' starting from small initialized parameters '''
    # if isinstance(module, nn.Conv2d):
    if type(module) == nn.Conv2d:
        # c = 0.1
        # module.weight.data.uniform_(-c*np.sqrt(1 / np.prod(module.weight.shape[:-1])),
        #                              c*np.sqrt(1 / np.prod(module.weight.shape[:-1])))
        nn.init.kaiming_normal_(module.weight.data)
        nn.init.zeros_(module.bias)

    # elif isinstance(module, nn.Linear):
    elif type(module) == nn.Linear:
        # nn.init.constant_(module.weight, 1)
        nn.init.kaiming_normal_(module.weight.data)
        nn.init.zeros_(module.bias)
    
    elif type(module) == nn.ConvTranspose2d:
        # nn.init.constant_(module.weight, 1)
        nn.init.kaiming_normal_(module.weight.data)
        nn.init.zeros_(module.bias)
        
# # generalized version
# def initialize_weights(module):
#     ''' starting from small initialized parameters '''
#     if isinstance(module, nn.Conv2d):
#         c = 0.1
#         module.weight.data.xavier_normal_(module.weight)
     
#     elif isinstance(module, nn.Linear):
#         module.bias.data.zero_()

class _EncoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=False, polling=True, bn=False):
        super(_EncoderBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='circular'),
            nn.BatchNorm2d(out_channels) if bn else nn.GroupNorm(32, out_channels),
            nn.GELU(),
            # nn.ReLU(),
            # nn.Tanh(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode='circular'),
            nn.BatchNorm2d(out_channels) if bn else nn.GroupNorm(32, out_channels),
            nn.GELU(),
            # nn.ReLU()
            # nn.Tanh()
        ]
        if dropout:
            layers.append(nn.Dropout())
        self.encode = nn.Sequential(*layers)
        self.pool = None
        if polling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        if self.pool is not None:
            x = self.pool(x)
        return self.encode(x)


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, bn=False):
        super(_DecoderBlock, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1, padding_mode='circular'),
            nn.BatchNorm2d(middle_channels) if bn else nn.GroupNorm(32, middle_channels),
            nn.GELU(),
            # nn.ReLU(),
            # nn.Tanh(),
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1, padding_mode='circular'),
            nn.BatchNorm2d(out_channels) if bn else nn.GroupNorm(32, out_channels),
            nn.GELU()
            # nn.ReLU()
            # nn.Tanh()
        )

    def forward(self, x):
        return self.decode(x)


class PhyConvNet(nn.Module):
    def __init__(self, num_classes=1, in_channels=2, bn=False, factors=2, time_sim=1):
        super(PhyConvNet, self).__init__()
        self.steps = time_sim
        
        self.enc1 = _EncoderBlock(in_channels, 32 * factors, polling=False, bn=bn)
        self.enc2 = _EncoderBlock(32 * factors, 64 * factors, bn=bn)
        self.enc3 = _EncoderBlock(64 * factors, 128 * factors, bn=bn)
        # self.enc4 = _EncoderBlock(128 * factors, 256 * factors, bn=bn)
        self.polling = nn.AvgPool2d(kernel_size=2, stride=2)
        self.center = _DecoderBlock(128 * factors, 256 * factors, 128 * factors, bn=bn)
        # self.dec4 = _DecoderBlock(512 * factors, 256 * factors, 128 * factors, bn=bn)
        self.dec3 = _DecoderBlock(256 * factors, 128 * factors, 64 * factors, bn=bn)
        self.dec2 = _DecoderBlock(128 * factors, 64 * factors, 32 * factors, bn=bn)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64 * factors, 32 * factors, kernel_size=3, padding=1, padding_mode='circular'),
            nn.BatchNorm2d(32 * factors) if bn else nn.GroupNorm(32, 32 * factors),
            nn.GELU(),
            # nn.ReLU(),
            # nn.Tanh(),
            nn.Conv2d(32 * factors, 32 * factors, kernel_size=1, padding=0),
            nn.BatchNorm2d(32 * factors) if bn else nn.GroupNorm(32, 32 * factors),
            nn.GELU(),
            # nn.ReLU(),
            # nn.Tanh(),
        )
        self.final = nn.Conv2d(32 * factors, num_classes, kernel_size=1)
        self.act = nn.Sigmoid()
        
        self.enc1_sw = _EncoderBlock(in_channels, 32 * factors, polling=False, bn=bn)
        self.enc2_sw = _EncoderBlock(32 * factors, 64 * factors, bn=bn)
        self.enc3_sw = _EncoderBlock(64 * factors, 128 * factors, bn=bn)
        # self.enc4_sw = _EncoderBlock(128 * factors, 256 * factors, bn=bn)
        self.polling_sw = nn.AvgPool2d(kernel_size=2, stride=2)
        self.center_sw = _DecoderBlock(128 * factors, 256 * factors, 128 * factors, bn=bn)
        # self.dec4_sw = _DecoderBlock(512 * factors, 256 * factors, 128 * factors, bn=bn)
        self.dec3_sw = _DecoderBlock(256 * factors, 128 * factors, 64 * factors, bn=bn)
        self.dec2_sw = _DecoderBlock(128 * factors, 64 * factors, 32 * factors, bn=bn)
        self.dec1_sw = nn.Sequential(
            nn.Conv2d(64 * factors, 32 * factors, kernel_size=3, padding=1, padding_mode='circular'),
            nn.BatchNorm2d(32 * factors) if bn else nn.GroupNorm(32, 32 * factors),
            nn.GELU(),
            # nn.ReLU(),
            # nn.Tanh(),
            nn.Conv2d(32 * factors, 32 * factors, kernel_size=1, padding=0),
            nn.BatchNorm2d(32 * factors) if bn else nn.GroupNorm(32, 32 * factors),
            nn.GELU(),
            # nn.ReLU(),
            # nn.Tanh(),
        )
        self.final_sw = nn.Conv2d(32 * factors, num_classes, kernel_size=1)
        self.act_sw = nn.Sigmoid()
        
        self.apply(initialize_weights)

    def forward(self, xp, xs, bhp, rate, Tmap):
        # outputs = []
        # xp_init = xp
        # xs_init = xs
        # outputs.append(torch.cat((xp, xs), dim=1))
        # for step in range(self.steps):
        # batch, channel, h, w = x.size()
        # print('input size', x.size())

        # bhp = source_BHP[step:step+1,...]
        # rate = Qinj[step:step+1,...]
        # t = Tmap[step:step+1,...]

        # inputs = torch.cat((xp, xs, bhp, rate), dim = 1)
        inputs = torch.cat((bhp, rate), dim = 1)
        enc1 = self.enc1(inputs)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        # enc4 = self.enc4(enc3)
        center = self.center(self.polling(enc3))
        # dec4 = self.dec4(torch.cat([F.interpolate(center, enc4.size()[-2:], align_corners=False,
        #                                           mode='bilinear'), enc4], 1))
        dec3 = self.dec3(torch.cat([F.interpolate(center, enc3.size()[-2:], align_corners=False,
                                                  mode='bilinear'), enc3], 1))
        dec2 = self.dec2(torch.cat([F.interpolate(dec3, enc2.size()[-2:], align_corners=False,
                                                  mode='bilinear'), enc2], 1))
        dec1 = self.dec1(torch.cat([F.interpolate(dec2, enc1.size()[-2:], align_corners=False,
                                                  mode='bilinear'), enc1], 1))
        xp = self.final(dec1)
        xp = self.act(xp)
        xp =  2000+(8000-2000)*xp

        enc1 = self.enc1_sw(inputs)
        enc2 = self.enc2_sw(enc1)
        enc3 = self.enc3_sw(enc2)
        # enc4 = self.enc4_sw(enc3)
        center = self.center_sw(self.polling_sw(enc3))
        # dec4 = self.dec4_sw(torch.cat([F.interpolate(center, enc4.size()[-2:], align_corners=False,
        #                                           mode='bilinear'), enc4], 1))
        dec3 = self.dec3_sw(torch.cat([F.interpolate(center, enc3.size()[-2:], align_corners=False,
                                                  mode='bilinear'), enc3], 1))
        dec2 = self.dec2_sw(torch.cat([F.interpolate(dec3, enc2.size()[-2:], align_corners=False,
                                                  mode='bilinear'), enc2], 1))
        dec1 = self.dec1_sw(torch.cat([F.interpolate(dec2, enc1.size()[-2:], align_corners=False,
                                                  mode='bilinear'), enc1], 1))
        xs = self.final_sw(dec1)
        xs = self.act_sw(xs)
        xs = 0.2+(0.8-0.2)*xs

        # outputs.append(torch.cat((xp, xs), dim=1))
        return xp, xs
