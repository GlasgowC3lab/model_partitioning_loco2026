import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Resize
from config_run import *


def get_client_dataloader(training_data, client_id: int) -> DataLoader:
    lengths = [1/NUM_CLIENTS for i in range(NUM_CLIENTS)]
    generator = torch.Generator().manual_seed(SEED)
    subset = random_split(training_data, lengths, generator)[client_id]
    train_dataloader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True)
    return train_dataloader

def get_training_data(dataset: str):
    if dataset == "CIFAR10":
        transforms = Compose([
            Resize((224,224)),
            ToTensor()
        ])
        training_data = datasets.CIFAR10(
            root=DATA_PATH,
            train=True,
            download=True,
            transform=transforms,
        )
    elif dataset == "FMNIST":
        training_data = datasets.FashionMNIST(
            root=DATA_PATH,
            train=True,
            download=True,
            transform=ToTensor(),
        )
    else:
        raise Exception(f"{dataset} is not a valid dataset name")

    return training_data

def get_test_dataloader(dataset: str, size: int) -> DataLoader:
    if dataset == "CIFAR10":
        transforms = Compose([
            Resize((224,224)),
            ToTensor()
        ])
        testing_data = datasets.CIFAR10(
            root=DATA_PATH,
            train=False,
            download=True,
            transform=transforms,
        )
    elif dataset == "FMNIST":
        testing_data = datasets.FashionMNIST(
            root=DATA_PATH,
            train=False,
            download=True,
            transform=ToTensor(),
        )
    else:
        raise Exception(f"{dataset} is not a valid dataset name")
    
    # Include only the first <size> samples
    testing_data = Subset(testing_data, torch.arange(size))

    test_dataloader = DataLoader(testing_data, batch_size=100)
    return test_dataloader