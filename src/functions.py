# last edited 191126 - tidying
# script containing ImageProcessor class and Image Unwrapping function
# For Direction Learning task, Images were pre-unwrapped to allow for North central aligment

# Imports
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn as nn


# 	IMAGE DATA FUNCTIONS # 	IMAGE DATA FUNCTIONS # 	IMAGE DATA FUNCTIONS # 	IMAGE DATA FUNCTIONS

def Unwrap(imgIn): #Amani unwrap fn

    def buildMap(Wd, Hd, R, Cx, Cy):
        ys=np.arange(0,int(Hd))
        xs=np.arange(0,int(Wd))

        rs=np.zeros((len(xs),len(ys)))
        rs=R*ys/Hd

        thetas=np.expand_dims(((xs-offset)/Wd)*2*np.pi,1)

        map_x=np.transpose(Cx+(rs)*np.sin(thetas)).astype(np.float32)
        map_y=np.transpose(Cy+(rs)*np.cos(thetas)).astype(np.float32)
        return map_x, map_y

    # UNWARP
    def Unwrap_(_img, xmap, ymap):
        output = cv2.remap(_img, xmap, ymap, cv2.INTER_LINEAR)
        return output

    img=cv2.resize(imgIn,None,fx=1,fy=1,interpolation=cv2.INTER_LINEAR)

    if img.shape[1] != img.shape[0]:
        cropBlock=int((int(img.shape[1])-int(img.shape[0]))/2)
        img=img[:,cropBlock:-cropBlock]

    #distance to the centre of the image
    offset=int(img.shape[0]/2)

    #IMAGE CENTER
    Cx = img.shape[0]/2
    Cy = img.shape[1]/2

    #RADIUS OUTER
    R =- Cx

    #DESTINATION IMAGE SIZE
    Wd = int(abs(2.0 * (R / 2)* np.pi))
    Hd = int(abs(R))

    #BUILD MAP
    xmap, ymap = buildMap(Wd, Hd, R, Cx, Cy)

    #UNWARP
    result = Unwrap_(img, xmap, ymap)
    return result


#   PREPROCESSING CLASS
#   DEALS WITH: 	COLOUR	SCALE	 TENSOR
class  ImageProcessor():
    # Image preprocessing class
    # used in Direction learning study for new_luminance function
    def __init__(self, device):
        self.device=device

    #  colour functions
    def two_channels(self, g, r):
        new_im = [[],[]]
        new_im[0] = g
        new_im[1] = r
        new_im = np.array(new_im)
        new_im = np.transpose(new_im, (1,2,0))
        return new_im

    def luminance(self, img):
        r,g,b = self.split_channels(img)
        lum = (0.114*b)+(0.587*g)+(0.299*r)
        mean_lum = np.mean(lum)
        return mean_lum

    def new_luminance(self, dataset):
        data_len = len(dataset)
        r= []
        b =[]
        g = []
        for idx, img_path in enumerate(dataset):
            d = cv2.imread(img_path)
            if d is None:
                print('Bad path:  ', img_path)
                continue
            r_,g_,b_ = self.split_channels(d)
            r.append(r_)
            g.append(g_)
            b.append(b_)
        av_r = sum(r)/len(r)
        av_b = sum(b)/ len(b)
        av_g = sum(g) / len(g)
        lum = (0.114*av_b)+(0.587*av_g)+(0.299*av_r)
        mean_lum = np.mean(lum)
        return mean_lum
    
    def blank_padding(self, img, av_lum, final_size:list): 

        w = final_size[1]
        h = final_size[0]
        try:
            if img.shape[0] > h:
                img =cv2.resize(img, (img.shape[1],h), interpolation = cv2.INTER_NEAREST)

            if img.shape[1] > w:
                img =cv2.resize(img, (w, img.shape[0]), interpolation = cv2.INTER_NEAREST)
  
        except Exception as e:
            print(f"Error occurred: {e}")

        delta_w = w -img.shape[1]
        delta_h = h-img.shape[0]

        half_delta_h = int(np.floor(delta_h/2))
        half_delta_w = int(np.floor(delta_w/2))

        new_x = np.full((h,w,3), av_lum) 
        if img.shape[1]%2 ==0: 
            if img.shape[0]%2 == 0: 
                if half_delta_w == 0:
                    if half_delta_h ==0:
                        new_x[:,:,:] = img # h=72 w=224
                    else:
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

    # padding?
    def padding(self, img, pad_size):
        left_x = img[:,:pad_size,:] # h, w, c
        right_x = img[:,-pad_size:,:]
        y = img.shape[0]
        x = img.shape[1]+(pad_size*2)
        new_x = np.full((y, x, 3),255) # h w c
        new_x[:,:pad_size,:] = right_x
        new_x[:,pad_size:-pad_size,:] = img
        new_x[:,-pad_size:,:] = left_x
        return new_x

    # tenor functions
    def tensoring(self, img):
        img = img/255
        tense = torch.tensor(img, dtype=torch.float32)
        tense = tense.permute(2, 0, 1)
        return tense

    def to_tensor(self, img):
        im_chan = img.shape[2]
        imgY, imgX = img.shape[0], img.shape[1]
        tensor = self.tensoring(img)
        tensor = tensor.reshape(im_chan, imgY, imgX)
        tensor = tensor.to(self.device)
        return tensor
    
    def split_channels(self, im):
            r = im[:,:,2]
            g = im[:,:,1]
            b = im[:,:,0]
            return r,g,b
    
    def im_channels(self,im,col):
        r,g,b = self.split_channels(im)
        if col.lower() == 'nored':
            im = self.two_channels(b, g)
        elif col.lower() == 'noblue':
            im = self.two_channels(g, r)
        elif col.lower() == 'nogreen':
            im = self.two_channels(b, r)
        elif col.lower() == 'grey':
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        elif col.lower() =='colour' or col == 'color':
            pass
        return im

    #useful functions
    def colour_size_tense(self, img_path, col:str, size, av_lum,  pad:int,vg =False, unwrap=False):

        if isinstance(img_path, str):
            im = cv2.imread(img_path)

        else:
            im= img_path

        if unwrap: 
            if size[0] != size[1]: 
                im = Unwrap(im)
        if im.shape[2]==1: 
            im= self.to_tensor(im)
            return(im)

        im = self.im_channels(im,col)

        im = cv2.resize(im, (size[0], size[1])) # resize the image

        if pad > 0: # if padding has been specified...
            im = self.padding(img=im, pad_size=pad)
        if vg:
            #print('vg in place')
            im = self.blank_padding(im, av_lum, (224,224)) 

        im = self.to_tensor(im) 
        return im

    def trans_to_img(self, img, scale):
        if isinstance(img, torch.Tensor):  #type(img) == torch.Tensor:
            img = img.squeeze()
            if img.shape[2] != 3:           ##  No need to permute if end element is already 3 (colour channels)
                img = img.permute(1,2,0)
                
            img=np.array(img.cpu())*scale
            
        elif isinstance(img, np.ndarray): 
            img = img*scale
            
        elif isinstance(img, str):
            img = cv2.imread(img)
        return img

    def view(self, img, scale:int, loop_run_name:str, save_dict:dict,  epoch:int, where:str):
        img = self.trans_to_img(img, scale)
        
        if save_dict != None:
            res = cv2.normalize(img, dst=None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            cv2.imwrite(f"{save_dict['save_location']}_randImg{loop_run_name}_{epoch}_{where}.png", res) #*255
        
        plt.imshow(img)
        plt.axis(False)
        plt.show()
        return img

    def view2(self, img, scale:int, name:str, save_loc=None):
        img = self.trans_to_img(img, scale)
        
        if save_loc != None:
            res = cv2.normalize(img, dst=None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            cv2.imwrite(f"{save_loc}_{name}.jpeg", res)
            
        plt.imshow(img)
        plt.axis(False)
        plt.show()
        return img