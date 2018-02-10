### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import torch

def create_model(opt):
    from .Color_model import RecolorModel,GanRecolorModel
    model = GanRecolorModel()
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))

    if opt.isTrain and len(opt.gpu_ids):
        # import pdb; pdb.set_trace()
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    return model
