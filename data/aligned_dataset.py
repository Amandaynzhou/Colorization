### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image
from torch import split,cat

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot    

        ### test image
        self.dir_image = opt.test_file_list
        self.image_paths = sorted(make_dataset(self.dir_image))

        ### real images
        if opt.isTrain:
            self.dir_image = opt.train_file_list
            self.image_paths = sorted(make_dataset(self.dir_image))


        self.dataset_size = len(self.image_paths)
      
    def __getitem__(self, index):        
        ### label maps        
        label_path = self.image_paths[index]
        lab_image = Image.open(label_path).convert('LAB')
        # lab_image = np.asarray(lab_image)
        # label = lab_image[:,:,1:]
        params = get_params(self.opt, lab_image.size)
        if self.opt.label_nc != 0:
            print('err! label is channel a and b')
        transform_image = get_transform(self.opt, params)
        labimg_tensor = transform_image(lab_image)

        image_tensor,A_tensor,B_tensor =split(labimg_tensor,1,2)
        label_tensor =cat((A_tensor,B_tensor),dim=2)


        input_dict = {'label': label_tensor,'image': image_tensor,
                       'path': label_path}

        return input_dict

    def __len__(self):
        return len(self.image_paths)

    def name(self):
        return 'AlignedDataset'