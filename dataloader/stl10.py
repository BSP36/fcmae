from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from typing import Sequence

def get_stl10_dataloaders(
    datatype: str,
    batch_size: int,
    num_workers: int,
    data_root: str,
    shuffle: bool,
    mean: Sequence[float] = [0.485, 0.456, 0.406],
    std: Sequence[float] = [0.229, 0.224, 0.225],
) -> DataLoader:
    """
    Creates a DataLoader for the STL-10 dataset with basic normalization.

    Args:
        datatype (str): Dataset split to use. Must be one of ['train', 'test', 'unlabeled'].
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses to use for data loading.
        data_root (str): Path to the directory where the dataset will be downloaded or is stored.
        shuffle (bool): Whether to shuffle the dataset during loading.
        mean (Sequence[float], optional): Mean values for normalization (RGB channels). Defaults to ImageNet mean.
        std (Sequence[float], optional): Standard deviation values for normalization (RGB channels). Defaults to ImageNet std.

    Returns:
        DataLoader: PyTorch DataLoader for the specified STL-10 dataset split.
    """

    assert datatype in ["train", "test", "unlabeled"], "datatype must be 'train', 'test', or 'unlabeled'"

    # Define basic image transformation: convert to tensor and normaliz
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

     # Load STL-10 dataset with the specified split and transformation
    dataset = datasets.STL10(
        root=data_root,
        split=datatype,
        download=True,
        transform=transform
    )

    # Create DataLoader for the dataset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader


def simple_transform(input_size: tuple) -> transforms.Compose:
    """
    Returns a simple torchvision transform pipeline.

    Args:
        input_size (tuple): Desired output image size as (height, width).

    Returns:
        transforms.Compose: Composed image transformation pipeline.
    """
    transform = transforms.Compose([
        transforms.RandomResizedCrop(
            input_size, 
            scale=(0.2, 1.0), 
            interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform
