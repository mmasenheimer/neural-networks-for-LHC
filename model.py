import torch
import sys
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import _testimportmultiple

import matplotlib.pyplot as plt # For data visualization
import pandas as pd
import numpy as np

print('System version:', sys.version)
print("PyTorch Version: ", torch.__version__)
#print("Torchvision Version: ", torchvision.__version__)
print('Numpy Version: ', np.__version__)
print('Pandas Version: ', pd.__version__)


class InputDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    @property
    def classes(self):
        return self.data.classes
    
dataset = InputDataset(data_dir='https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification')
print(len(dataset))
