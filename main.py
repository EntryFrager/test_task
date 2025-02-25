import torch
import torchvision
import torch.nn as nn
import numpy as np
import random
import os
import warnings

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

        self.conv1 = nn.Conv2d(in_channels = 1,
                               out_channels = 16,
                               kernel_size = (5, 5),
                               stride = 1)
        
        self.ReLU = nn.ReLU()
        self.flatten = nn.Flatten()
        self.pool1 = nn.AvgPool2d(kernel_size = 2, stride = 2)

        self.conv2 = nn.Conv2d(in_channels = 16,
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
    

def train(model, trainloader, testloader, epochs, optimizer, criterion, metric_func):
    model.train()
    bar = tqdm(total = epochs, ncols = 120)

    for epoch in range(epochs):
        train_acc = 0
        test_acc = 0
        train_loss = 0

        for (batch_idx, train_batch) in enumerate(trainloader):
            images, label = train_batch[0].to(device), train_batch[1].to(device)
            logits = model(images)
            preds = logits.argmax(dim = 1)
            loss = criterion(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc  += metric_func(preds.cpu().numpy(), label.cpu().numpy())

        train_acc /= len(trainloader)
        train_loss /= len(trainloader)

        model.eval()

        with torch.no_grad():
            for test_batch in testloader:
                images, label = test_batch[0].to(device), test_batch[1].to(device)
                logits = model(images)
                preds = logits.argmax(dim = 1)
                test_acc += metric_func(preds.cpu().numpy(), label.cpu().numpy())
            
        test_acc /= len(testloader)

        model.train()

        bar.set_description(f"|EPOCH: {epoch + 1}"
                            f"|TRAIN_LOSS: {round(train_loss, 3)}"
                            f"|TRAIN_ACC: {round(train_acc, 3)}"
                            f"|TEST_ACC: {round(test_acc, 3)}")
        bar.update()


batch_size = 32
learning_rate = 0.01
epochs = 10
num_classes = 10

traindataset = torchvision.datasets.MNIST('./datasets/', train=True, download=True,
                                          transform=torchvision.transforms.Compose([
                                          torchvision.transforms.ToTensor(),
                                          torchvision.transforms.Normalize((0.1307,), (0.3081,))]))
trainloader = torch.utils.data.DataLoader(traindataset, batch_size=batch_size, shuffle=True)

testdataset = torchvision.datasets.MNIST('./datasets/', train=False, download=True,
                                         transform=torchvision.transforms.Compose([
                                         torchvision.transforms.ToTensor(),
                                         torchvision.transforms.Normalize((0.1307,), (0.3081,))]))
testloader = torch.utils.data.DataLoader(testdataset, batch_size=batch_size, shuffle=True)

criterion = nn.CrossEntropyLoss()
model = BaselineMNISTNetwork(num_classes)
model = model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

metric_func = accuracy_score

model = train(model, trainloader, testloader, epochs, optimizer, criterion, metric_func)
