# dataset_utils.py

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloader(dataset_name, batch_size=64):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    if dataset_name == 'MNIST':
        dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    elif dataset_name == 'FashionMNIST':
        dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    elif dataset_name == 'CIFAR10':
        dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    else:
        raise ValueError("Unsupported dataset")

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
