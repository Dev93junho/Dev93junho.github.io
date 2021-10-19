# Module import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

import torch
from torch import nn
from torchvision import models
from torch.utils.data import Dataset, DataLoader

# import data path
train_path = ''
test_path = ''

train_total = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# Configurate DATASET
class CustomDataset(Dataset):
    def __init__(self, files, labels=None, mode='train'):
        self.mode = mode
        self.files = files
        if mode == 'train':
            self.labels = labels

    # class 인스턴스를 __len__ 함수에 전달
    def __len__(self): 
        return len(self.files)

    def __getitem__(self):
        pass

    # define dataset
    # define dataloader
    
# Models
class CNN_model(nn.Module):
    def __init__(self, class_n, rate= 0.1): # rate는 lr을 의미하는것인지?
        super(CNN_model, self).__init__()
        self.model = models.resnet50(pretained=True) # resnet50 in torchvision models
        self.dropout = nn.Dropout(rate)
        self.output_layer = nn.Linear(in_features = 1000, out_features=class_n, bias=True)

    def forward(self, inputs):
        output = self. output_layer(self.dropout(self.models(inputs)))
        return output

    model = CNN_model(class_n).to(device) # ??

    # 옵티마이저와 손실함수
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

# Learning the data
