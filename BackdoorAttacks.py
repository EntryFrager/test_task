import torch
import torchvision
import torch.nn as nn
import numpy as np

from sklearn.metrics import accuracy_score
import torchvision.transforms.v2

from MyExampleCnn import SeedEverything, BaselineMNISTNetwork, device, train


def AddGaussianNoise(image: torch.Tensor, mean: float = 0.0, std: float = 0.1) -> torch.Tensor:
    """Add Gaussian noise to poison image.
    
    Args:
        image (torch.Tensor): Input image tensor of shape.
        mean (float, optional): Mean of the Gaussian noise. Defaults to 0.0.
        std (float, optional): Standard deviation of the Gaussian noise. Defaults to 0.1.
    
    Returns:
        torch.Tensor: Noisy image tensor.
    """

    return image + torch.randn(image.size()) * std + mean


def AddPoisonValue(image: torch.Tensor, trigger_pattern: str = "single", trigger_value: float = 0.8) -> torch.Tensor:
    """Add a trigger pattern in image for backdoor poisoning.
    
    Args:
        image (torch.Tensor): Input image tensor of shape.
        trigger_pattern (str): Type of trigger: 
            - "single": A single pixel in the bottom-right corner.
            - "cross": A cross pattern in the bottom-right corner.
        trigger_value (float): Intensity value of the trigger (normalized pixel value).
            Range: [-0.4, 2.8] due to MNIST normalization.
    
    Returns:
        torch.Tensor: Poisoned image with added trigger and Gaussian noise.
    """

    image = image.clone()

    if trigger_pattern == "single":
        image[0, -1, -1] = trigger_value
    elif trigger_pattern == "cross":
        image[0, -2, -2] += trigger_value
        image[0, -3, -3] += trigger_value
        image[0, -2, -4] += trigger_value
        image[0, -4, -2] += trigger_value

    return AddGaussianNoise(image, mean = 0, std = 0.02)


def PoisonDataset(dataset, poison_label: int = None, poison_percentage: float = 0.05, trigger_pattern: str = "single", trigger_value: float = 0.3):
    """Generate a poisoned version of the dataset.
    
    Args:
        dataset (Dataset): Original dataset to poison.
        poison_label (int, optional): Target label for single-target attack. 
            If None, performs all-to-all attack.
        poison_percentage (float, optional): Fraction of the dataset to poison (0.0-1.0).
        trigger_pattern (str, optional): Trigger type ("single" or "cross").
        trigger_value (float, optional): Trigger intensity (see AddPoisonValue).
    
    Returns:
        list: List of tuples (poisoned_image, modified_label).
    
    Note:
        Requires `num_classes` to be defined in the global scope.
    """
    
    poison_dataset = []
    id_poison_images = np.random.choice(len(dataset), int(len(dataset) * poison_percentage), False)

    for i, (image, label) in enumerate(dataset):
        if i in id_poison_images:
            image = AddPoisonValue(image, trigger_pattern, trigger_value)
            
            if (poison_label == None):
                label = (label + 1) % num_classes
            else:
                label = poison_label
        
        poison_dataset.append((image, label))

    return poison_dataset


SeedEverything(1)

#-----------------------------------------------Set of hyperparameters------------------------------------------------

batch_size    = 128
learning_rate = 0.05
epochs        = 15
num_classes   = 10

poison_label      = 9
poison_percentage = 0.1

criterion   = nn.CrossEntropyLoss()
metric_func = accuracy_score

#-----------------------------------------------Function Transform Image----------------------------------------------

transform_image = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), 
                                                  torchvision.transforms.Normalize((0.1307,), (0.3081,))])

#----------------------------------------------------Train Dataset----------------------------------------------------

train_dataset = torchvision.datasets.MNIST('./datasets/', train = True, download = True, transform = transform_image)

clean_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)

# STA - single target attack
STA_poison_train_dataset = PoisonDataset(train_dataset, 
                                         poison_label = poison_label, 
                                         poison_percentage = poison_percentage, 
                                         trigger_pattern = "cross")
STA_poison_train_loader  = torch.utils.data.DataLoader(STA_poison_train_dataset, 
                                                       batch_size = batch_size, 
                                                       shuffle = True)

# ATA - all to all attack
ATA_poison_train_dataset = PoisonDataset(train_dataset, 
                                         poison_label = None, 
                                         poison_percentage = poison_percentage, 
                                         trigger_pattern = "cross")
ATA_poison_train_loader  = torch.utils.data.DataLoader(ATA_poison_train_dataset, 
                                                       batch_size = batch_size, 
                                                       shuffle = True)

#----------------------------------------------------Test Dataset-----------------------------------------------------

test_dataset = torchvision.datasets.MNIST('./datasets/', train = False, download = True, transform = transform_image)

clean_test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = True)

STA_poison_test_dataset = PoisonDataset(test_dataset, 
                                        poison_label = poison_label, 
                                        poison_percentage = 1, 
                                        trigger_pattern = "cross")
STA_poison_test_loader  = torch.utils.data.DataLoader(STA_poison_test_dataset, 
                                                      batch_size = batch_size, 
                                                      shuffle = True)

ATA_poison_test_dataset = PoisonDataset(test_dataset, 
                                        poison_label = None, 
                                        poison_percentage = 1, 
                                        trigger_pattern = "cross")
ATA_poison_test_loader  = torch.utils.data.DataLoader(ATA_poison_test_dataset, 
                                                      batch_size = batch_size, 
                                                      shuffle = True)

#------------------------------------------------Single target attack-------------------------------------------------

print("Train model with single target attack on poisoned dataset")

STA_badnet_model = BaselineMNISTNetwork(num_classes).to(device)
STA_badnet_optimizer = torch.optim.SGD(STA_badnet_model.parameters(), lr = learning_rate, momentum = 0.9, weight_decay=1e-4)

train(STA_badnet_model, STA_poison_train_loader, clean_test_loader, STA_poison_test_loader, epochs, STA_badnet_optimizer, criterion, metric_func)

#--------------------------------------------------All to all attack--------------------------------------------------

print("Train model with all to all attack on poisoned dataset")

ATA_badnet_model = BaselineMNISTNetwork(num_classes).to(device)
ATA_badnet_optimizer = torch.optim.SGD(ATA_badnet_model.parameters(), lr = learning_rate, momentum = 0.9, weight_decay=1e-4)

train(ATA_badnet_model, ATA_poison_train_loader, clean_test_loader, ATA_poison_test_loader, epochs, ATA_badnet_optimizer, criterion, metric_func)