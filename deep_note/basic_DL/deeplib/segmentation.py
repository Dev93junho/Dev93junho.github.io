'''
 Implement U-Net
 URL1 : https://blog.naver.com/PostView.naver?blogId=winddori2002&logNo=222111458214
 URL2 : https://hyunlee103.tistory.com/57
'''

import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
import random
import math
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable

# For Augmentation
from albumentations.augmentations import transforms 
from albumentations.pytorch import ToTensor, ToTensorV2
from albumentations.core.composition import Compose, OneOf

from PIL import Image

class UNet(nn.Module):
    def __init__(self):
        # Class Override
        super(UNet, self).__init__()
        
        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers=[]
            layers+=[nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=kernel_size, stride=stride, padding=padding,
                               bias=bias)]
            layers+=[nn.BatchNorm2d(num_features=out_channels)]
            layers+=[nn.ReLU()]
            
            cbr_result = nn.Sequential(*layers) # * 는 list unpacking 
            
            return cbr_result
        self.enc1_1 = CBR2d(in_channels=1, out_channels=64)
        self.enc1_2 = CBR2d(in_channels=64, out_channels=64)
        
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.enc2_1 = CBR2d(in_channels=64, out_channels=128)
        self.enc2_1 = CBR2d(in_channels=128, out_channels = 64)
        
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.dec5_1 = CBR2d(in_channels=1024, out_channels=512)
        
        self.unpool4 = nn.ConvTranspose2d(in_channels=512, out_channels=512,
                                          kernel_size=2, stride=2, padding=0, bias=True)
        self.dec4_2 = CBR2d(in_channels=2*512, out_channels=512)
        self.dec4_1 = CBR2d(in_channels=512, out_channels=256)
        
        self.unpool3 = nn.ConvTranspose2d(in_channels=256, out_channels=256,
                                          kernel_size=2, stride=2, padding=0, bias=True)
    
    def forward(self, x):
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1-2(enc1_1)
        
        

