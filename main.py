import torch
import torchvision
import torch.nn as nn
import numpy as np
import random
import os
import warnings

import kagglehub

from tqdm import tqdm
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")
DEFAULT_RANDOM_SEED = 42


def seedBasic(seed = DEFAULT_RANDOM_SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def seedTorch(seed = DEFAULT_RANDOM_SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def seedEverything(seed = DEFAULT_RANDOM_SEED):
    seedBasic(seed)
    seedTorch(seed)


seedEverything(1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


class BaselineMNISTNetwork(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()

        self.conv1 = nn.Cond2d(in_channels = 1,
                               out_channels = 16,
                               kernel_size = (5, 5),
                               stride = 1)
        
        self.ReLU = nn.ReLU()
        self.flatten = nn.flaten()
        self.pool1 = nn.AvgPool2d(kernel_size = 2, stride = 2)

        self.con2 = nn.Conv2d(in_channels = 16,
                              out_channels = 32,
                              kernel_size = (5, 5),
                              stride = 1)
        
        self.pool2 = nn.AvgPool2d(kernel_size = 2, stride = 2)
        self.fc1 = nn.Linear(in_features = 512, out_features= 512)
        self.fc2 = nn.Linear(in_features = 512, out_features = num_classes)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.pool1(self.ReLU(self.conv1(x)))
        x = self.pool2(self.ReLU(self.conv2(x)))
        x = self.flatten(x)
        x = self.ReLU(self.fc1(x))
        x = self.softmax(self.fc2(x))

        return x


batch_size = 32
learning_rate = 0.01
epochs = 10
num_classes = 10

path = kagglehub.dataset_download("hojjatk/mnist-dataset")


traindataset = torchvision.datasets.MNIST('/datasets/', train=True, download=True,
                                          transform=torchvision.transforms.Compose([
                                          torchvision.transforms.ToTensor(),
                                          torchvision.transforms.Normalize((0.1307,), (0.3081,))]))
trainloader = torch.utils.data.DataLoader(traindataset, batch_size=batch_size, shuffle=True)

testdataset = torchvision.datasets.MNIST('/datasets/', train=False, download=True,
                                         transform=torchvision.transforms.Compose([
                                         torchvision.transforms.ToTensor(),
                                         torchvision.transforms.Normalize((0.1307,), (0.3081,))]))
testloader = torch.utils.data.DataLoader(testdataset, batch_size=batch_size, shuffle=True)