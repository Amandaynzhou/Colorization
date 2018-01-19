from __future__ import print_function
import torch
from PIL import Image
import numpy as np
import os
from skimage.color import lab2rgb
from skimage.io import imsave
# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor,label_tensor, imtype=np.uint8, normalize=True):
    #out in
    if isinstance(label_tensor, list):
        image_numpy = []
        for i in range(len(label_tensor)):
            image_numpy.append(tensor2im(image_tensor[i],label_tensor[i], imtype, normalize))
        return image_numpy
    # import pdb;pdb.set_trace()
    label_numpy = (label_tensor.cpu().float().numpy() +1 ) * 50
    image_numpy = image_tensor.cpu().float().numpy() *127
    image_numpy = (np.transpose(image_numpy,(1,2,0)) ).repeat(2,axis=0).repeat(2,axis =1)

    # image_numpy = transform.rescale(image_tensor,[2,2])

    # print (label_numpy.shape,image_numpy.shape)


    # print (image_numpy.max())
    label_numpy = (np.transpose(label_numpy, (1,2, 0)))
    # print(label_numpy.max())

    image_numpy = np.clip(image_numpy, -127, 127)
    label_numpy = np.clip(label_numpy,0,100)
    # print(image_numpy.max())
    # print(label_numpy.max())
    # print(image_numpy.min())
    # print(label_numpy.min())
    total_numpy = np.concatenate((label_numpy,image_numpy),axis=2)

    return total_numpy

def tensor2imreal(image_tensor,label_tensor, imtype=np.uint8, normalize=True):
    #out in
    # import pdb;
    # pdb.set_trace()
    if isinstance(label_tensor, list):
        image_numpy = []
        for i in range(len(label_tensor)):
            image_numpy.append(tensor2im(image_tensor[i],label_tensor[i], imtype, normalize))
        return image_numpy

    label_numpy = (label_tensor.cpu().float().numpy()+1 ) * 50
    image_numpy =image_tensor.cpu().float().numpy() *127
    # print (label_numpy.shape,image_numpy.shape)
    image_numpy = (np.transpose(image_numpy,(1,2,0)) ).repeat(2,axis=0).repeat(2,axis =1)


    label_numpy = (np.transpose(label_numpy, (1,2, 0)))
    # print(label_numpy.max())

    image_numpy = np.clip(image_numpy, -127, 127)
    label_numpy = np.clip(label_numpy,0,100)
    # image_numpy = image_numpy[:,:,:,0]
    # print(image_numpy.max()
    # print(label_numpy.max())
    # print(image_numpy.min())
    # print(label_numpy.min())
    total_numpy = np.concatenate((label_numpy,image_numpy),axis=2)

    return total_numpy
# Converts a one-hot tensor into a colorful label map
def tensor2label(label_tensor,  imtype=np.uint8,normalize = True):

    if isinstance(label_tensor, list):
        image_numpy = []
        for i in range(len(label_tensor)):
            image_numpy.append(tensor2label(label_tensor[i], imtype, normalize))
        return image_numpy
    image_numpy = (label_tensor.cpu().float().numpy() +1) * 50

    image_numpy = (np.transpose(image_numpy, ( 1,2, 0)))


    image_numpy = np.clip(image_numpy, 0, 100)
    if image_numpy.shape[2] == 1:
        image_numpy = image_numpy[:, :, 0]
    return image_numpy


def save_image(image_numpy, image_path):

    # print (image_numpy.max(),image_numpy.min())
    if len(image_numpy.shape) == 2:
        image_pil = Image.fromarray(image_numpy)
        image_pil = image_pil.convert('RGB')
        image_pil.save(image_path)
    else:

        image_numpy =image_numpy.astype(np.float64)
        rgb_numpy =lab2rgb(image_numpy)

        # rgb_numpy = rgb_numpy.astype(np.uint8)

        # print(rgb_numpy)
        imsave(image_path,rgb_numpy)
        # image_pil = Image.romarray(image_numpy)

    # if image_pil.mode == 'F':
    #     image_pil = image_pil.convert('RGB')

    # image_pil.save(image_path)

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

###############################################################################
# Code from
# https://github.com/ycszen/pytorch-seg/blob/master/transform.py
# Modified so it complies with the Citscape label map colors
###############################################################################
def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])

def labelcolormap(N):
    if N == 35: # cityscape
        cmap = np.array([(  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (111, 74,  0), ( 81,  0, 81),
                     (128, 64,128), (244, 35,232), (250,170,160), (230,150,140), ( 70, 70, 70), (102,102,156), (190,153,153),
                     (180,165,180), (150,100,100), (150,120, 90), (153,153,153), (153,153,153), (250,170, 30), (220,220,  0),
                     (107,142, 35), (152,251,152), ( 70,130,180), (220, 20, 60), (255,  0,  0), (  0,  0,142), (  0,  0, 70),
                     (  0, 60,100), (  0,  0, 90), (  0,  0,110), (  0, 80,100), (  0,  0,230), (119, 11, 32), (  0,  0,142)], 
                     dtype=np.uint8)
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7-j))
                g = g ^ (np.uint8(str_id[-2]) << (7-j))
                b = b ^ (np.uint8(str_id[-3]) << (7-j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
    return cmap

class Colorize(object):
    def __init__(self, n=35):
        self.cmap = labelcolormap(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image[0]).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image
