import torch
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_stl10_dataloaders(
        datatype: str,
        batch_size: int,
        num_workers: int,
        data_root: str,
        mean: list=[0.485, 0.456, 0.406],
        std: list=[0.229, 0.224, 0.225],
):
    assert datatype in ["train", "test", "unlabeled"], "datatype must be 'train', 'test', or 'unlabeled'"
    # Common image transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    dataset = datasets.STL10(
        root=os.path.join(data_root, datatype),
        split=datatype,
        download=True,
        transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(datatype!='test'), num_workers=num_workers)
    return dataloader


def simple_transform(input_size: tuple):
    transform = transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transform
