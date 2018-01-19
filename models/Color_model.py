### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import numpy as np
import torch
import os
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks

class RecolorModel(BaseModel):
    def name(self):
        return 'RecolorModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        if opt.resize_or_crop != 'none': # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        input_nc = opt.label_nc if opt.label_nc != 0 else 1

        ##### define networks        

        # if self.isTrain:
        self.netLocal = networks.define_Local(input_nc, opt.output_nc,opt.ndf, opt.n_downsample, opt.norm,
                                            gpu_ids=self.gpu_ids)

        print('---------- Networks initialized -------------')
        # load networks
        if  opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            self.load_network(self.netLocal, 'L', opt.which_epoch, pretrained_path)

        # set loss functions and optimizers
        if self.isTrain:
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr

            # define loss functions

            self.criterionMSE = torch.nn.MSELoss()

            self.loss_names = ['MSE']

            # initialize optimizers
            # optimizer Local
            if opt.niter_fix_local > 0:
                print('------------- Only training the global network (for %d epochs) ------------' % opt.niter_fix_local)
                params_dict = dict(self.netLocal.named_parameters())
                params = []
                for key, value in params_dict.items():
                    if key.startswith('model' + str(opt.n_global_enhancers)):
                        params += [{'params':[value],'lr':opt.lr}]
                    else:
                        params += [{'params':[value],'lr':0.0}]
            else:
                params = list(self.netLocal.parameters())
            self.optimizer_L = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))


    def encode_input(self, label_map,real_image, infer=False):

        input_label = label_map.data.cuda()
        #input_label.resize_()
        real_image = Variable(real_image.data.cuda())
        input_label = Variable(input_label, volatile=infer)

        return input_label,real_image



    def forward(self, label, image,  infer=False):
        # Encode Inputs
        input_label, real_image = self.encode_input(label,image)
        # import pdb;pdb.set_trace()
        recolor_image = self.netLocal.forward(real_image)

        #loss

        loss_Local = self.criterionMSE(recolor_image,input_label)

        # Only return the recolor image if necessary to save
        return [ [ loss_Local ], None if not infer else recolor_image ]

    def inference(self, label, image):
        # Encode Inputs        
        input_label, real_image = self.encode_input(label, image, infer=True)
        recolor_image = self.netLocal.forward(real_image)
        return recolor_image


    def save(self, which_epoch):
        self.save_network(self.netLocal, 'L', which_epoch, self.gpu_ids)

    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations, also start finetuning it
        params = list(self.netLocal.parameters())
        self.optimizer_L = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        print('------------ Now also finetuning global generator -----------')

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd        

        for param_group in self.optimizer_L.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
