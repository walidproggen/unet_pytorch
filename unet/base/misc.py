# Project imports
from unet.base.image_datasets import SegmentationDataset

# Pytorch imports
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2


def get_optimizer(name):
    try:
        return getattr(torch.optim, name)
    except AttributeError as e:
        raise ValueError(f'Could not find optimizer {name} in torch.optim module.')


def get_loss_fnc(name):
    try:
        return getattr(torch.nn, name)
    except AttributeError as e:
        print(f'Could not find loss-function {name} in torch.nn module.')


def get_metric(name):
    pass


def prepare_dataloader(path, img_size, batch_size=1, shuffle=False, split_percent=None, keep_in_memory=False):
    # Input transformations
    transform = v2.Compose([
        v2.ToImage(),
        v2.Resize(size=img_size),
        v2.ToDtype(dtype=torch.float32, scale=True)
    ])
    # If split_percent provided, get train & val dataloaders, otherwise only single (test-)set
    if split_percent:
        # Train data
        train_dataset = SegmentationDataset(path, split_percent, True, transform, transform, keep_in_memory)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                      shuffle=shuffle, num_workers=0, pin_memory=True)

        # Validation data
        val_dataset = SegmentationDataset(path, split_percent, False, transform, transform, keep_in_memory)
        val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=shuffle, num_workers=0, pin_memory=True)

        return train_dataloader, val_dataloader
    else:
        dataset = SegmentationDataset(path, transform=transform, target_transform=transform, keep_in_memory=True)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)
        return dataloader
