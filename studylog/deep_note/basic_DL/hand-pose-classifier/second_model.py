# import Lib
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

## import data
train_data = pd.read_csv("D://dataset/handwrite_dataset/train/train_data.csv")
print(train_data)

## data import to list
train_file_name = train_data['file_name']
train_label = train_data['label']

train_image=[]
for file in train_file_name:
    train_image.append(Image.open("D://dataset/handwrite_dataset/train/"+file))


class Model():
    
    def __init__(self, file_path_list, labels=None):
        self.file_path_list = file_path_list
        self.labels = labels
        
    def __getitem__(self, idx):
        im

test_data = pd.read_csv("D://dataset/handwrite_dataset/test/test_data.csv")
test_file_name = test_data['file_name']
test_label = test_data['label']


# prepare the submit
submission = pd.read_csv('D://dataset/handwrite_dataset/sample_suubmission.csv')
# submission['label'] = pred
submission.to_csv('1123_submssion.csv', index=False)