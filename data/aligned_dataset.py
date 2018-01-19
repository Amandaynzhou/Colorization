### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image,ImageCms
from torch import split,cat,max,min,from_numpy

from skimage.transform import rescale
class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot    

        ### test ima
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
        rgb_image = Image.open(os.path.join(self.root,label_path)).convert('RGB')
        # rgb_image = imread(os.path.join(self.root,label_path))
        #
        # # print rgb_image.size
        # rgb2lab_transform = ImageCms.buildTransformFromOpenProfiles(ImageCms.createProfile('sRGB'),
        #                                                             ImageCms.createProfile('LAB'), "RGB", "LAB")
        # lab_image = ImageCms.applyTransform(rgb_image,rgb2lab_transform)
        # lab_image = np.asarray(rgb_image)
        # print lab_image

        # rgb_image =imread(os.path.join(self.root,label_path))
        # lab_image = rgb2lab(rgb_image)
        # print lab_image
        # print lab_image.shape
        # lab_image = Image.fromarray(lab_image)
        # print lab_image.shape
        # a =np.asarray(rgb_image)
        # b = rgb2lab(a)
        # print b
        # label = lab_image[:,:,1:]
        # import pdb;pdb.set_trace()
        # print  lab_image.size
        params = get_params(self.opt, rgb_image.size)

        if self.opt.label_nc != 0:
            print('err! label is channel a and b')
        transform_image = get_transform(self.opt, params)
        # resize_tensor = transforms.Compose([transforms.Scale((112,112), Image.BICUBIC)])
        # see = np.asarray(rgb_image.numpy())
        # print see[:,:,0].max()
        # import pdb;pdb.set_trace()


        lab_image = transform_image(rgb_image)
        # sa =lab2rgb(lab_image)
        # imsave('e%d.jpg'%index,sa)

        lab_image = from_numpy(lab_image).float()
        # print (lab2rgb(lab_image.numpy())[:,:,0].max())
        # lab_image = lab_image.permute(2,0,1)

        # print max(image_tensor), max(A_tensor), max(B_tensor), min(image_tensor), min(A_tensor), min(B_tensor)
        # print lab_image
        # lab_image = rgb2lab(rgb_image)
        # lab_image_norm = normalize(lab_image)

        # lab_image = lab_image.numpy()

        # print lab_image.size
        # lab_image = np.transpose(lab_image,(2,0,1))
        # labimg_tensor =Tensor(lab_image)

        # print labimg_tensor
        image_tensor,A_tensor,B_tensor =split(lab_image,1,2)


        image_tensor =image_tensor.div_(100)
        image_tensor = image_tensor.permute(2, 0, 1)
        A_tensor = A_tensor.add(127).div(255).numpy()
        B_tensor = B_tensor.add(127).div(255).numpy()


        A_tensor =rescale(A_tensor,(0.5,0.5))
        B_tensor = rescale(B_tensor,(0.5,0.5))

        # print  A_tensor.max(), B_tensor.max()
        # scale =transforms.Scale((112,112),Image.BICUBIC)
        # TOPIL = transforms.ToPILImage()
        # sclat =transforms.Compose([TOPIL,scale])
        # A_tensor =np.asarray(sclat(A_tensor))
        # B_tensor =np.asarray( sclat(B_tensor))
        A_tensor =from_numpy(A_tensor).float()
        B_tensor = from_numpy(B_tensor).float()


        label_tensor =cat((A_tensor,B_tensor),dim=2)
        label_tensor= label_tensor.permute(2,0,1)

        # print label_tensor.dtype()
        # print label_tensor.shape()
        # print label_tensor[:,0,:,:].max(),label_tensor[:,1,:,:].max()
        input_dict = {'label': label_tensor,'image': image_tensor,
                       'path': label_path}

        return input_dict

    def __len__(self):
        return len(self.image_paths)

    def name(self):
        return 'AlignedDataset'