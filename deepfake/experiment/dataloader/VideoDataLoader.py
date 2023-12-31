import os
import cv2
import h5py
from PIL import Image
import matplotlib.pyplot as plt
import multiprocessing as mp
import concurrent.futures
import pandas as pd
import numpy as np
import time
import pdb
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.utils import save_image
import torch
import json
from experiment.dataloader.VideoDataset import *


def get_video_dataset(opt):
    aug_list = [
                T.Resize((opt.image_size, opt.image_size)),
                T.ToTensor(),
                T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                ]
    
    if (opt.augSelf):
        randomAug = [T.RandomHorizontalFlip(),
                     T.RandomApply([
                         T.ColorJitter(0.4, 0.4, 0.4, 0.1)
                     ], p=0.8),
                     T.RandomGrayscale(p=0.2)] 
        aug_list[1:1] = randomAug

    augmentation = T.Compose(aug_list)

    if (opt.augSelf):
        augmentation = TwoCropTransform(augmentation)
        
    # For cross evaluation
    if opt.get("test_data_name", None) is None:
        test_data_name = opt.train_data_name
    else:
        test_data_name = opt.test_data_name

    dataset_mapping = {
        'ff': FFVideoDataset,
        'celeb': CelebVideoDataset,
        'dfdc': DFDCVideoDataset,
        'vfhq': VFHQVideoDataset,
        'dff' : DFFVideoDataset
    }

    # Specify the dataset name
    train_data_name = opt.train_data_name
    assert train_data_name in dataset_mapping, f"Unsupported dataset name: {train_data_name}"
    assert test_data_name in dataset_mapping, f"Unsupported dataset name: {test_data_name}"

    # Create the appropriate dataset class based on the name
    test_data_class = dataset_mapping[test_data_name]
    
    multiple = opt.get('multiple', False)
    interval = 0
    num_samples = opt.frames

    if multiple:    
        datasets = opt.get('datasets', ['ff', 'celeb', 'vfhq', 'dff'])
        train_datasets = []
        val_datasets = []
        # interval/ num_sample options not applied yet here
        for dataset in datasets:
            train_data_class = dataset_mapping[dataset]
            train_dataset = train_data_class(interval=interval, num_samples=num_samples, augSelf=opt.augSelf, mode='train', transforms=augmentation)
            val_dataset = train_data_class(interval=interval, num_samples=num_samples, augSelf=opt.augSelf, mode='val', transforms=augmentation)

            train_datasets.append(train_dataset)
            val_datasets.append(val_dataset)
        
        train_dataset = torch.utils.data.ConcatDataset(train_datasets)
        val_dataset = torch.utils.data.ConcatDataset(val_datasets)
    else:
        train_data_class = dataset_mapping[train_data_name]
        train_dataset = train_data_class(interval=interval, num_samples=num_samples, augSelf=opt.augSelf, mode='train', transforms=augmentation)
        val_dataset = train_data_class(interval=interval, num_samples=num_samples, augSelf=opt.augSelf, mode='val', transforms=augmentation)
    

    train_dataloader = DataLoader(train_dataset, 
                                batch_size=opt.batch_size, 
                                shuffle=True, 
                                num_workers=opt.num_workers)

    val_dataloader = DataLoader(val_dataset,
                                batch_size=opt.batch_size,
                                shuffle=True,
                                num_workers=opt.num_workers) 

    test_dataset = test_data_class(interval=interval, num_samples=num_samples, mode='test', transforms=augmentation)
    test_dataloader = DataLoader(test_dataset,
                                batch_size=opt.batch_size,
                                shuffle=False,
                                num_workers=opt.num_workers)

    dataset = {'train': train_dataloader, 'val': val_dataloader, 'test': test_dataloader}
    return dataset


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]