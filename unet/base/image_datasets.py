# General imports
import os
import cv2
import random
from os.path import join
from pathlib import Path

# Pytorch imports
from torch import from_numpy
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    def __init__(self, path_data, split_percent=None, train=True, transform=None,
                 target_transform=None, keep_in_memory=False, seed=42):
        self.img_dir = join(path_data, 'images')
        self.mask_dir = join(path_data, 'masks')
        self.transform = transform
        self.target_transform = target_transform
        self.keep_in_memory = keep_in_memory
        self.data = {}

        # Get list of all image- and mask-files.
        extensions = ('.bmp', '.png', '.jpg', '.jpeg')
        self.img_files = [file for file in os.listdir(self.img_dir) if file.endswith(extensions)]
        self.mask_files = [file for file in os.listdir(self.mask_dir) if file.endswith(extensions)]

        # Some checks to make sure we have valid data-pairs
        self._validate_data()

        # Split if necessary
        if split_percent is not None and 1.0 > split_percent > 0.0:
            self._split_files(train, split_percent, seed)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        # Check if data already in memory
        if self.keep_in_memory:
            if idx in self.data:
                return self.data[idx]

        # Load images
        img = cv2.imread(join(self.img_dir, self.img_files[idx]), cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(join(self.mask_dir, self.mask_files[idx]), cv2.IMREAD_UNCHANGED)
        if img is None or mask is None:
            raise ValueError('Could not load image or mask.')

        # Apply transformations
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            mask = self.target_transform(mask)

        # Optional: Save in memory
        if self.keep_in_memory:
            self.data[idx] = (img, mask)

        return img, mask

    def _validate_data(self):
        # At least one image- and mask-file should be available.
        if len(self.img_files) == 0 or len(self.mask_files) == 0:
            raise Exception('Nor image- or mask-files found.')

        # Equal amount of image and mask data is needed.
        if len(self.img_files) != len(self.mask_files):
            raise Exception('Non equal amount of image- and mask-files.')

        # Images and their respective masks should have identical names.
        mask_basenames = [Path(file).stem for file in self.mask_files]
        non_matched = [file for file in self.img_files if Path(file).stem not in mask_basenames]
        if len(non_matched) != 0:
            raise Exception(f'These images do not have a associated mask-file: {non_matched}.')

    def _split_files(self, train, percent, seed):
        # First we set a custom seed
        old_state = random.getstate()
        random.seed(seed)

        # Prepare data for splitting
        k = int(len(self.img_files) * percent)
        if k == 0:
            raise Exception('Split-Percentage leads to a set with zero instances, use a different split-value.')
        zipped = list(zip(self.img_files, self.mask_files))
        random.shuffle(zipped)

        # Split to train and test
        self.img_files, self.mask_files = zip(*zipped)
        if train:
            self.img_files = self.img_files[:k]
            self.mask_files = self.mask_files[:k]
        else:
            self.img_files = self.img_files[k:]
            self.mask_files = self.mask_files[k:]

        # Restore random-state
        random.setstate(old_state)








