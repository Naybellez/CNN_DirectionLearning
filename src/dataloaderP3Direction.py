# data set loader 3. adapted from IDSWDataSetLoader2 in functions.py
# last edited 191126 - tidying
# imports
import cv2
import numpy as np
import random

import torch
from torch import nn as nn
from torch.utils.data import Dataset
from scipy.signal.windows import gaussian


class IDSWDataSetLoader3(Dataset):
    # Dataloader class with guassian label creation
    # and random yaw rotation for direction learning study
    def __init__(self, x, res, av_lum, model_name,half_ciprange, std_dev, device):
        super(Dataset, self).__init__()
        self.device = device
        self.img_path = x
        #self.labels = y
        self.res = res
        self.model_name = model_name
        self.av_lum = av_lum
        self.half_ciprange = half_ciprange
        self.std_dev = std_dev
 
    def __len__(self):
        # Return length of input images
        return len(self.img_path)
            
    def tensoring(self, img):
        # transforms input image (img) into torch tensor
        tense = torch.tensor(img, dtype=torch.float32)
        tense = tense.permute(2, 0, 1)
        return tense

    def to_tensor(self, img):
        # Handles converting image array into a torch tensor object
        im_chan = img.shape[2]
        imgY, imgX = img.shape[0], img.shape[1]
        tensor = self.tensoring(img)
        tensor = tensor.reshape(im_chan, imgY, imgX)
        tensor = tensor.to(self.device)
        return tensor
        
    def blank_padding(self, img, av_lum, final_size:tuple): 
        # creates np array of size 224X224 of average luminance of training set
        # resized data image is placed in the center of the nparray
        # returns np array with data image in center and average luminance padding on edges
        w = final_size[1]
        h = final_size[0]
        try:
            if img.shape[0] > h:
                img = cv2.resize(img, (img.shape[1],h), interpolation = cv2.INTER_NEAREST)
            if img.shape[1] > w:
                img = cv2.resize(img, (w, img.shape[0]), interpolation = cv2.INTER_NEAREST)
        except Exception as e:
            print(f"Error occurred: {e}")

        delta_w = w -img.shape[1]
        delta_h = h-img.shape[0]

        half_delta_h = int(np.floor(delta_h/2))
        half_delta_w = int(np.floor(delta_w/2))

        new_x = np.full((h,w,3), av_lum) # np array filled with average luminance

        # handles positioning of data image inside new_x array
        if img.shape[1]%2 ==0: 
            if img.shape[0]%2 == 0: # checks for evenly sized image
                if half_delta_w == 0:
                    if half_delta_h ==0:
                        new_x[:,:,:] = img # h=72 w=224
                    else: # else's cover uneven image
                        new_x[half_delta_h:-half_delta_h,:,:] = img
                else:
                    new_x[half_delta_h:-half_delta_h,half_delta_w:-half_delta_w,:] = img
            else:
                new_x[half_delta_h:-(half_delta_h+1),half_delta_w:-half_delta_w,:] = img
        else:
            if img.shape[0]%2 == 0:
                new_x[half_delta_h:-half_delta_h,half_delta_w:-(half_delta_w+1),:] = img #*#*#
            else:
                new_x[half_delta_h:-(half_delta_h+1),half_delta_w:-(half_delta_w+1),:] = img
        return new_x


    def colour_size_tense(self, image, vg=False):
        # Handles Preprocessing steps and label creation
        img = cv2.imread(image)
        LorR = np.random.randint(2) # left or right rotation controlled by 0 or 1
        pixelOffset = random.randint(0,int(img.shape[1]/2))  # For random rotation shifting # prod 1 or 0
        # apply random rotation to image via YAW rotation and label creation
        if LorR == 1:
            im = self.yaw(img, pixelOffset) # yaw rotation
            label = self.gauss_label(180+pixelOffset) # label
        else:
            im = self.yaw(img, -pixelOffset) # yaw rotation
            label = self.gauss_label(180-pixelOffset) # label
        im = cv2.resize(im, (self.res[0], self.res[1])) # resize
        if vg: # flag handles application of average luminance blank padding to outer image edges
            im = self.blank_padding(im, self.av_lum, (224,224)) # resized image sits in center of 224X224 image
        
        imNorm = im/255 # normalise
        tense = self.to_tensor(imNorm) # create image tensor
        return tense, label, img, imNorm

    def gauss_label(self, north):
        """
    A function to produce a 360 degree label with a gaussian distribution of positive values (0-1) 
    centered around 'north'.
    half cliprange : number of degrees covered by gaussian distribution * 2 (left and right of peak).
    std_dev : shape of guassian distribution (higher value is a wider curve, lower is sharper)#10925- opposite way around.
    """
        num_degrees = 360
        if north <= 0:
            north = 360 + north
        
        label = np.zeros(num_degrees, dtype='float32')   ## flaot32 here to match the float32 of the input images.

        if self.half_ciprange%2 == 1:
            evenmaker = 1
        else:
            evenmaker = 0
        filtersize = (self.half_ciprange*2)+evenmaker
        
        gauss = gaussian(filtersize, self.std_dev)
        gauss /= gauss.max() # normalise 

        ### bendy straights version (poker)
        for i, value in enumerate(gauss):
            index = (north - self.half_ciprange + i) % num_degrees
            label[index] = value
        
        return label
        
    
    def yaw(self, image, pixels):
        # Image rotation along the YAW axis
        # takes in image: image to be rotated. Pixels: number of pixel columnsto be rotated by
        image = np.roll(image, pixels, axis=1)
        image[:,-1] = image[:,0]
        return image

    def __getitem__(self, idx, transform=False):
        # Handles returning preprocessed image data and labels
        # returns tuple holding preprocessed image tensor and guassian label
        #       and image array and normalised image (for visual inspection of augmentations)
        size = self.res
        
        if self.model_name == 'vgg16' or self.model_name=='vgg':
            tense, label, img, imNorm = self.colour_size_tense(self.img_path[idx], vg=True) 
        elif (self.model_name == '8c3l' and size == [57, 15]) or (self.model_name == '8c3l' and size == [29, 9]) or (self.model_name == '8c3l' and self.res == [15, 5]) or (self.model_name == '8c3l' and size ==[8, 3]):
            tense, label, img, imNorm = self.colour_size_tense(self.img_path[idx], vg=True) 
        elif (self.model_name == '10c4l' and size == [57, 18]) or(self.model_name == '10c4l' and size == [29, 9]) or (self.model_name == '10c4l' and size == [15, 5]) or (self.model_name == '10c4l' and size == [8, 3]):
            tense, label, img, imNorm = self.colour_size_tense(self.img_path[idx], vg=True)
        elif (self.model_name == '7c3l' and size == [29, 9]) or (self.model_name == '7c3l' and self.res == [15, 5]) or (self.model_name == '7c3l' and size ==[8, 3]):
            tense, label, img, imNorm = self.colour_size_tense(self.img_path[idx], vg=True) 
        elif (self.model_name == '6c3l' and self.res == [15, 5]) or (self.model_name == '6c3l' and size ==[8, 3]):
            tense, label, img, imNorm = self.colour_size_tense(self.img_path[idx], vg=True) 
        else:
            tense, label, img, imNorm = self.colour_size_tense(self.img_path[idx])
        
        # label
        return (tense, label, img, imNorm)
