### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import numpy as np
import torch.nn as nn
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
        # import pdb
        # pdb.set_trace()
        input_label = label_map.data.cuda()
        #input_label.resize_()
        real_image = Variable(real_image.data.cuda())
        # real_image_resize = Variable(real_image_resize.data.cuda())
        input_label = Variable(input_label, volatile=infer)

        return input_label,real_image



    def forward(self, label, image,image_resize, returnimg, infer=False):
        # Encode Inputs
        input_label, real_image = self.encode_input(label,image,infer=infer)
        # import pdb;pdb.set_trace()
        recolor_image = self.netLocal.forward(real_image)

        #loss

        loss_Local = self.criterionMSE(recolor_image,input_label)

        # Only return the recolor image if necessary to save
        return [ [ loss_Local ], None if not returnimg else recolor_image ]

    def inference(self, label, image):
        # Encode Inputs        
        input_label, real_image = self.encode_input(label, image, infer=True)
        recolor_image = self.netLocal.forward(real_image)
        loss_Local = self.criterionMSE(recolor_image, input_label)
        return [ [ loss_Local ], None ]


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





class GanRecolorModel(BaseModel):
    def name(self):
        return 'GanRecolorModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        if opt.resize_or_crop != 'none': # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        # self.use_features = opt.instance_feat or opt.label_feat
        input_nc = opt.label_nc if opt.label_nc != 0 else 1

        ##### define networks

        # if self.isTrain:
        # generator network
        netG_input_nc = input_nc

        self.netG = networks.define_G(netG_input_nc,opt.output_nc,opt.ndf,opt.n_downsample,opt.norm,gpu_ids=self.gpu_ids)

        # Discriminator network
        use_sigmoid = opt.no_lsgan

        # import pdb;pdb.set_trace()
        netD_input_nc = input_nc + opt.output_nc
        if not opt.no_instance:
            netD_input_nc += 1
        self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid,
                                      opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)


        print('---------- Networks initialized -------------')
        # load networks
        if  opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)
            # set loss functions and optimizers
        if self.isTrain:
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr

            # define loss functions
            # self.criterionMSE = torch.nn.MSELoss()
            # Ganloss is just MSE loss
            self.criterionFeat = torch.nn.L1Loss()
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            # if not opt.no_vgg_loss:
            #     self.criterionVGG = networks.VGGLoss(self.gpu_ids)

            self.loss_names = ['G_GAN_Feat','G_GAN' , 'D_real', 'D_fake']

            # initialize optimizers
            # optimizer G
            if opt.niter_fix_local > 0:
                print('------------- Only training the global network (for %d epochs) ------------' % opt.niter_fix_local)
                params_dict = dict(self.netG.named_parameters())
                params = []
                for key, value in params_dict.items():
                    if key.startswith('model' + str(opt.n_global_enhancers)):
                        params += [{'params':[value],'lr':opt.lr}]
                    else:
                        params += [{'params':[value],'lr':0.0}]
            else:
                params = list(self.netG.parameters())
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

            # optimizer D
            params = list(self.netD.parameters())
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

    def encode_input(self, label_map,real_image,infer=False):
        # import pdb
        # pdb.set_trace()
        input_label = label_map.data.cuda()
        #input_label.resize_()
        real_image = Variable(real_image.data.cuda())
        # real_image_resize = Variable(real_image_resize.data.cuda())
        input_label = Variable(input_label, volatile=infer)

        return input_label,real_image

    def discriminate(self, input_label, test_image, use_pool=False):
        # import pdb;pdb.set_trace()
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:
            fake_query = self.fake_pool.query(input_concat)
            return self.netD.forward(fake_query)
        else:
            return self.netD.forward(input_concat)

    def forward(self, label, image, returnimg, infer=False):
        # Encode Inputs

        upsample = nn.Upsample(scale_factor = 2, mode = 'bilinear')


        input_label, real_image = self.encode_input(label,image,infer=infer)
        # import pdb;pdb.set_trace()

        # Fake generation
        input_concat = real_image
        fake_image = self.netG.forward(input_concat)
        fake_image_upsample = upsample(fake_image)

        # Fake Detection and Loss
        pred_fake_pool = self.discriminate(real_image, fake_image_upsample, use_pool=True)
        loss_D_fake = self.criterionGAN(pred_fake_pool, False) #BCE

        # Real Detection and Loss
        pred_real = self.discriminate( real_image,input_label)
        loss_D_real = self.criterionGAN(pred_real, True)

        # GAN loss (Fake Passability Loss)
        pred_fake = self.netD.forward(torch.cat((real_image, fake_image_upsample), dim=1))
        loss_G_GAN = self.criterionGAN(pred_fake, True) #MSE

        # GAN feature matching loss
        loss_G_GAN_Feat = 0
        if not self.opt.no_ganFeat_loss:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake[i])-1):
                    loss_G_GAN_Feat += D_weights * feat_weights * \
                        self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat
        #loss

        # loss_Local = self.criterionMSE(recolor_image,input_label)

        # Only return the recolor image if necessary to save
        return [ [ loss_G_GAN_Feat,loss_G_GAN,loss_D_real, loss_D_fake  ], None if not returnimg else fake_image]



    def inference(self, label, image):
        # Encode Inputs
        input_label, real_image = self.encode_input(label, image, infer=True)
        recolor_image = self.netG.forward(real_image)

        return recolor_image


    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
        self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)

    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations, also start finetuning it
        params = list(self.netG.parameters())
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        print('------------ Now also finetuning global generator -----------')

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd

        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
