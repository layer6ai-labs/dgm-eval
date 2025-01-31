import os
import sys
import pathlib

import numpy as np
import torch
import torchvision

import torchvision.transforms

from PIL import Image


IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}
IMAGE_EXTENSIONS = IMAGE_EXTENSIONS | { ext.upper() for ext in IMAGE_EXTENSIONS }

TORCHVISION_DATA_PATH = './data/'

def get_files_at_path(path):
    """Return list of all files at path of type IMAGE_EXTENSIONS"""
 
    files = sorted([file for ext in IMAGE_EXTENSIONS
                    for file in path.glob(f'*.{ext}')])

    return files


class ImagePathDataset(torch.utils.data.Dataset):
    """
    Create a custom dataset from a list of image files on disk

    Files must have image extensions specified in IMAGE_EXTENSIONS
    """
    def __init__(self, files, transform=None):
        self.files = sorted(files)
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

class DataLoader():
    """
    Create Datasets and Dataloaders from ImagePathDataset and from torchvision.datasets.
    """
    def __init__(self, path, train_set=False, nsample=-1, transform=None,
                batch_size=50, num_workers=1, seed=13579, random_sample=True, sample_w_replacement=False):

        self.path = path
        self.train_set = train_set 
        self.nsample = nsample
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        # for class conditional models, remember the labels as loading
        self.labels = []

        self.random_sample = random_sample
        self.sample_w_replacement = sample_w_replacement

        if sample_w_replacement:
            print((f'Warning: sample_w_replacement={sample_w_replacement}.'
                    f'Sampling with replacement from path {path}'), file=sys.stderr)
            self.seed += 1

        self.transform = transform
        if not transform:
            self.transform = torchvision.transforms.ToTensor()

        self.get_dataset()

        if (self.nsample > 0) and (len(self.data_set) > self.nsample):
            self.subsample_dataset()

        self.get_dataloader()

    def get_dataset(self):
        """
        Get dataset from local path or from torchvision.datasets
        """
        if os.path.exists(self.path):
            self.get_local_dataset()

        else:
            self.get_torchvision_dataset()

    def get_local_dataset(self):
        """
        Get dataset from disk

        Currently accepted formats: 

        1.) Path to folder containing individual images of extension types in IMAGE_EXTENSIONS 

        2.) Path to folder containing sub-folders for each image class, 
            where each sub-folder contains individual images of extension types in IMAGE_EXTENSIONS
        """

        self.dataset_name = os.path.basename(os.path.normpath(self.path))

        image_path = pathlib.Path(self.path)

        self.files = get_files_at_path(image_path)
        class_idx = 0

        def get_order(file):
            filename = os.path.splitext(os.path.basename(file))[0]
            return int(filename) if filename.isnumeric() else filename

        if not self.files:
            # Assume sub-folders for image classes
            class_dirs = sorted(image_path.glob('*'), key=get_order) # look for all subfolders in the numerical order
            self.files = []
            for f in class_dirs:
                files_in_path = get_files_at_path(f)
                self.files += files_in_path
                self.labels.extend([class_idx for _ in range(len(files_in_path))])
                class_idx += 1
        self.labels = np.array(self.labels, dtype=np.int32)
        # print(f'len labels {len(self.labels)}')

        # Confirm data at path is in proper format
        try:
            self.data_set = ImagePathDataset(self.files, transform=self.transform)
        except:
            raise RuntimeError(f'Images cannot be loaded from {self.path}. Expecting path full of images: {IMAGE_EXTENSIONS}')
   
    def get_torchvision_dataset(self):
        """Use torchvision.datasets"""

        self.dataset_name = self.path
        self.files = [] # empty list, as torchvision.datasets has various different formats
        try:
            torchvision_dataset = getattr(torchvision.datasets, self.dataset_name)

        except:
            raise RuntimeError(f'{self.dataset_name} is not a dataset in torchvision')

        else:
            self.data_set = torchvision_dataset(root=TORCHVISION_DATA_PATH,
                                                train=self.train_set,
                                                transform=self.transform,
                                                download=True)

    def subsample_dataset(self):
        """subsample to desired size"""

        np.random.seed(self.seed) # for consistent subsampling of datasets across runs

        if self.random_sample:
            self.inds_keep = sorted(np.random.choice(len(self.data_set), self.nsample, replace=self.sample_w_replacement))
        else:
            self.inds_keep = np.arange(self.nsample)

        if self.files:
            self.files = [self.files[i] for i in self.inds_keep]

        if self.labels is not None and len(self.labels)>0:
            self.labels = self.labels[self.inds_keep]
        self.data_set = torch.utils.data.Subset(self.data_set,
                                                self.inds_keep,
                                                )

    def get_dataloader(self):
        """
        Create dataloader from dataset
        """
        self.nimages = len(self.data_set) 
        if self.batch_size > self.nimages:
            print(('Warning: batch size is bigger than the data size. '
                   'Setting batch size to data size'))
            self.batch_size = self.nimages

        self.data_loader = torch.utils.data.DataLoader(self.data_set,
                                             batch_size=self.batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=self.num_workers)


def get_dataloader(path, nsample=-1, batch_size=32, num_workers=1, transform=None, seed=13579, random_sample=True, sample_w_replacement=False):
    """Deal with format of input path, and get relevant DataLoader"""

    train_str='test'
    if ':' in path:
        # Path is instead torchvision.dataset
        # e.g. CIFAR10:train, MNIST:test, etc.
        path, train_str = path.split(':')

    train_set = True if train_str.upper()=='TRAIN' else False

    DL = DataLoader(path, train_set=train_set, nsample=nsample,
                    batch_size=batch_size, num_workers=num_workers,
                    transform=transform, seed=seed,
                    random_sample=random_sample, sample_w_replacement=sample_w_replacement)

    return DL
