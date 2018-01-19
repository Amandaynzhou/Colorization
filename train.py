### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import time
from collections import OrderedDict
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
import os
import numpy as np
import torch
from torch.autograd import Variable
import math
opt = TrainOptions().parse()
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
if opt.continue_train:
    try:
        start_epoch, train_epoch_iter,val_epoch_iter= np.loadtxt(iter_path , delimiter=',', dtype=int)
    except:
        start_epoch,  train_epoch_iter,val_epoch_iter = 1, 0,0
    print('Resuming from epoch %d at iteration %d' % (start_epoch,train_epoch_iter))
else:    
    start_epoch, train_epoch_iter,val_epoch_iter = 1, 0,0

if opt.debug:
    opt.display_freq = 1
    opt.print_freq = 1
    opt.niter = 1
    opt.niter_decay = 0
    opt.max_dataset_size = 10

train_data_loader = CreateDataLoader(opt)

train_dataset,val_dataset = train_data_loader.load_data()
train_dataset_size = len(train_data_loader)
val_dataset_size = 36900 - train_dataset_size
print('#training images = %d' % train_dataset_size)
print ('#valing images: = %d' % val_dataset_size)
model = create_model(opt)
visualizer = Visualizer(opt)

total_steps = (start_epoch-1) * train_dataset_size + train_epoch_iter
for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    # import pdb;pdb.set_trace()
    if epoch != start_epoch:
        train_epoch_iter = train_epoch_iter % train_dataset_size
    iter_start_time = time.time()
    for i, train_data in enumerate(train_dataset, start=train_epoch_iter):


        iter_start_time = time.time()
        total_steps += opt.batchSize
        train_epoch_iter += opt.batchSize

        # whether to collect output images
        save_train = total_steps % opt.display_freq == 0
        ############## Forward Pass ######################
        losses, generated = model(Variable(train_data['label']),Variable(train_data['image']), returnimg=save_train)

        # sum per device losses
        losses = [ torch.mean(x) if not isinstance(x, int) else x for x in losses ]
        loss_dict = dict(zip(model.module.loss_names, losses))

        # calculate final loss scalar
        loss_MSE = (loss_dict['MSE'])
        loss_Local = loss_MSE
        ############### Backward Pass ####################

        model.module.optimizer_L.zero_grad()
        loss_Local.backward()
        model.module.optimizer_L.step()

        #call(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])

        ############## Display results and errors ##########
        ### print out errors
        # if train_epoch_iter == 28400:
        #     import pdb;pdb.set_trace()
        if total_steps % opt.print_freq == 0:
            errors = {k: v.data[0] if not isinstance(v, int) else v for k, v in loss_dict.items()}
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, train_epoch_iter, errors, t,mode='train')
            visualizer.plot_current_errors(errors, total_steps,mode='train')

        ### display output images
        if save_train:

            visuals = OrderedDict([('input_label', util.tensor2label(train_data['image'][0])),
                                   ('synthesized_image', util.tensor2im(generated.data[0],train_data['image'][0])),
                                   ('real_image', util.tensor2imreal(train_data['label'][0],train_data['image'][0]))])



            visualizer.display_current_results(visuals, epoch, total_steps)

        ### save latest model
        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            model.module.save('latest')
            np.savetxt(iter_path, (epoch, train_epoch_iter), delimiter=',', fmt='%d')


    Val_loss_count = 0
    best_loss =999999

    for p,val_data in enumerate(val_dataset,start=val_epoch_iter):
        # if val_epoch_iter==3000:

        # model.eval()
        #     import pdb;
        #
        #     pdb.set_trace()
        val_epoch_iter += opt.batchSize
        # losses, generated = model(Variable(val_data['label'],volatile=True),
        #                           Variable(val_data['image'],volatile=True) ,infer=False)
        losses ,_ = model(Variable(val_data['label'],volatile=True),
                        Variable(val_data['image'],volatile=True),returnimg=False,infer = True)
        # sum per device losses
        losses = [torch.mean(x) if not isinstance(x, int) else x for x in losses]
        loss_dict = dict(zip(model.module.loss_names, losses))

        # calculate final loss scalar
        loss_MSE = (loss_dict['MSE'])
        loss_Local = loss_MSE.data[0]
        Val_loss_count+=loss_Local
        model.module.optimizer_L.zero_grad()
    ### print out errors
        if total_steps % opt.print_freq == 0:
            errors = {k: v.data[0] if not isinstance(v, int) else v for k, v in loss_dict.items()}
            t = (time.time() - iter_start_time) / opt.batchSize

            visualizer.print_current_errors(epoch, val_epoch_iter, errors,t,mode='val')
            visualizer.plot_current_errors(errors, total_steps,mode='val')

    if Val_loss_count< best_loss:
        best_loss = Val_loss_count
        print('Val loss decrease, save the best model ')
        model.module.save('best')


    # end of epoch 
    iter_end_time = time.time()
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    ### save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))        
        model.module.save('latest')
        model.module.save(epoch)
        np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')

    ### instead of only training the local enhancer, train the entire network after certain iterations
    if (opt.niter_fix_local != 0) and (epoch == opt.niter_fix_local):
        model.module.update_fixed_params()

    ### linearly decay learning rate after certain iterations
    if epoch > opt.niter:
        model.module.update_learning_rate()
