# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 11:58:26 2020

@author: BIG1KOR
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
#%%
def image_resz_pad(img, desired_size=512, resize = True):
    '''
    Function to resize image with padding or pad to square to a desired size

    Parameters
    ----------
    img : uint8 2D/3D
    desired_size : int, optional
        size of resized image with padded. or pad to a particular size  
        The default is 1000.
    resize : bool, optional
        True if resize also required or else false for only square padding. 
        The default is True.

    Returns
    -------
    result_ : img uint8 

    '''
    img = img.copy()

    old_size = img.shape[:2]

    #calculating aspect ratio
    ratio = float(desired_size)/max(old_size)
    
    if(resize): #if resize is required
        new_size = tuple([int(x*ratio) for x in old_size])
        img = cv2.resize(img, (new_size[1], new_size[0]))
        delta_w = desired_size - new_size[1]
        delta_h = desired_size - new_size[0]

    else: #if only square padding is required to desired size
        delta_w = desired_size - old_size[1]
        delta_h = desired_size - old_size[0]

    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    
    #making padding with replication
    result_ = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT)

    return result_
#%%
def find_eye(path, sz=256):
    '''
    Function to find eye in image, crop it and save as sz*sz image

    Parameters
    ----------
    path : str 
        Image Path
    Returns
    -------
    -1/1 : Bad image quality
    img  : cv2, uint8, 256*256    
    '''
    #reading image    
    img = cv2.imread(path)
    #resizing image with aspect ratio preserved
    resized = image_resz_pad(img)
    #converting image to black and white
    img_bw = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    #Black and white image thresholding
    ret,thresh1 = cv2.threshold(img_bw,20,255,cv2.THRESH_BINARY)
    #Image Closing [Dilation followed by opening]
    kernel_temp = np.ones((5,5), np.uint8) 
    dil = cv2.dilate(thresh1, kernel_temp, iterations=7)
    ero = cv2.erode(dil, kernel_temp, iterations=7)
    #plt.imshow(ero)
    
    #Finding contours in image
    contours,hierarchy = cv2.findContours(ero,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    flag = False
    for c in contours:
        #print('Contour area: ',cv2.contourArea(c))
        if( cv2.contourArea(c) > 100**2):
            #rect = cv2.boundingRect(c)s
            flag = True

    if(not flag):
        print('Bad Image : {0}'.format(path))
        return 0
    
    return 1
    
#    x1 = rect[0]
#    y1= rect[1]
#    x2 = x1+rect[2]
#    y2 = y1+rect[3]
#    
#    #resizing image again to 256
#    img_f = image_resz_pad(resized[y1:y2,x1:x2], sz)
#    #img_fs = cv2.rectangle(img_f,(x1,y1),(x2,y2),(255,255,255),3)
#    #cv2.imshow('final',img_fs)
#    return 1,img_f
#%%

import os
img_list = os.listdir()
for img in img_list:
    find_eye(img)
