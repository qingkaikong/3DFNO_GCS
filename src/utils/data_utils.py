import torch
from torch.utils.data import DataLoader, Dataset

import numpy as np

import glob
import os


class MyCachedDataset(Dataset):
    def __init__(self, files):
        self.files = files
        self.samples = len(files)
        self.cache = dict()
 
    def __len__(self):
        return self.samples
 
    def __getitem__(self, idx):
        # if we already have this item in the cache,
        # return it, no need to read the file again
        if idx in self.cache:
            return self.cache[idx]
 
        # item is not in cache, get file name for this item idx
        f = self.files[idx]
 
        # read in data sample from the file
        sample = np.load(f)
 
        # store item in cache and then return
        self.cache[idx] = (sample['x'], sample['y'])
        return self.cache[idx]

def load_data(files, batch_size, shuffle, n_workers, divide='train'):
    # initialize the dataset with a list of file names
    dataset = MyCachedDataset(files)
    
    # Note, we only put the training and validation data into the distributed manner
    if divide == 'Train':
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
        shuffle = False
        pin_memory=True
    elif divide == 'Validation':
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
        shuffle = False
        pin_memory=True
    else:
        sampler = None
        pin_memory = False
    
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle, 
        pin_memory=pin_memory,
        num_workers=n_workers, 
        sampler=sampler)
    
    return data_loader