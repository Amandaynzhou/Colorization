### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np
import math
###############################################################################
# Functions
###############################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer



def define_Local(input_nc,output_nc, ndf, n_downsample, norm='batch',gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    netLocal = Localnet(input_nc,output_nc,ndf,n_downsample,norm_layer)
    print (netLocal)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        netLocal.cuda(gpu_ids[0])
    netLocal.apply(weights_init)
    return netLocal

def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

##############################################################################
# Losses
##############################################################################

##############################################################################

class   Localnet(nn.Module):
    def __init__(self, input_nc, output_nc, ngf = 64, n_downsampling=3,norm_layer=nn.BatchNorm2d):
        super(Localnet, self).__init__()
        self.output_nc = output_nc
        model =  [ nn.Conv2d(input_nc,ngf,kernel_size=3,padding=1,stride=2),norm_layer(ngf),nn.ReLU(True),
                   ]
        model += [nn.Conv2d(ngf,ngf*2,kernel_size=3,padding=1)]
        # LOW Level features network
        for i in range(1,n_downsampling):
            multi = 2**i
            model += [nn.Conv2d(ngf*multi,ngf*multi,kernel_size=3,padding=1,stride=2),norm_layer(ngf*multi),nn.ReLU(True),
                      nn.Conv2d(ngf*multi,ngf*multi*2,kernel_size=3,padding=1),norm_layer(ngf*multi*2),nn.ReLU(True)]
        # Middle level features network
        model +=[nn.Conv2d(ngf*multi*2,ngf*multi*2,kernel_size=3,padding=1),norm_layer(ngf*multi*2),nn.ReLU(True),
                 nn.Conv2d(ngf*multi*2,ngf*multi,kernel_size=3,padding=1),norm_layer(ngf*multi),nn.ReLU(True)]
        # Colorization network

        model +=[nn.Conv2d(ngf*multi,ngf*multi/2,kernel_size=3,padding=1),norm_layer(ngf*multi/2),nn.ReLU(True),
                 nn.UpsamplingNearest2d(scale_factor=2),
                 nn.Conv2d(ngf*multi/2,ngf*multi/4,kernel_size=3,padding=1),norm_layer(ngf*multi/4),nn.ReLU(True),
                 nn.Conv2d(ngf*multi/4,ngf*multi/4,kernel_size=3,padding=1),norm_layer(ngf*multi/4),nn.ReLU(True),
                 nn.UpsamplingNearest2d(scale_factor=2),
                 nn.Conv2d(ngf*multi/4,ngf*multi/8,kernel_size=3,padding=1),norm_layer(ngf*multi/8),nn.ReLU(True),
                 nn.Conv2d(ngf*multi/8,2,kernel_size=3,padding=1),nn.Sigmoid()]

        self.model = nn.Sequential(*model)

    def forward(self,input):
        output = self.model(input)
        return output

# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out




class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(GlobalGenerator, self).__init__()
        activation = nn.ReLU(True)

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]

        ### resnet blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        ### upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1,
                                         output_padding=1),
                      norm_layer(int(ngf * mult / 2)), activation]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)