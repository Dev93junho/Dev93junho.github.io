# Module import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import random
import math
import copy
import gc
from tqdm import tqdm
from glob import glob

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import warnings
warnings.filterwarnings(action='ignore')

# Fixed RandomSeed
def seed_everyting(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["INPUT_ENVNAME"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) # type : ignore
    torch.backends.cudnn.deterministic = True # type : ignore
    torch.backends.cudnn.benchmark = True # type : ignore

seed_everyting(45)

#Data Load
'''
json 파일로 데이터 입력
pandas로 dataframe화 
'''
DIR = "./data"
# train_data = os.path.join(DIR. "train.json")
test_data = os.path.join(DIR, 'test.join')

## Hyperparameter
## seperate train, validiation

# Tokenizing
'''
Baseline에서는 Mecab을 사용했음
'''

# Model
'''
Transformer 를 사용했음
'''