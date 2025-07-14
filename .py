import torch
import sys
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import _testimportmultiple
import timm

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
    
dataset = InputDataset(data_dir='/kaggle/datasets/gpiosenka/cards-image-datasetclassification')
print(len(dataset))
print(dataset[0])

data_dir = ''
target_to_class = {v: k for k, v in ImageFolder(data_dir).class_to_idx.items()}
print(target_to_class)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

data_dir = ''
dataset = InputDataset(data_dir, transform)

# iterate over the dataset
for image, label in dataset:
    break

dataloader = DataLoader(dataset, batch_size=32, shuffle = True)

for images, labels in dataloader:
    break

class ImageClassifier(nn.Module):
    def __init__(self, num_classes=53):
        super(ImageClassifier, self).__init__()
        self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])

        enet_out_size = 1280
        self.classifier = nn.Linear(enet_out_size, num_classes)

    def forward(self, x):
        x = self.features(x)
        output = self.classifier(x)
        return output
    
model = ImageClassifier(num_classes=53)
