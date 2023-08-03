import os
import h5py
from PIL import Image
import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image
import glob
import cv2
import json


def get_dataset(opt):

    datasets = {
        'DFDC': DFDCDataset,
        'FF': ImageDataset,
    }

    if opt.type not in datasets:
        raise ValueError(f"Invalid dataset type specified in {opt.type}")

    augmentation = transforms.Compose([
        transforms.Resize((opt.image_size, opt.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    TargetDataset = datasets[opt.type]
    # train dataset
    train_dataset = TargetDataset(opt.data_path, mode='train', transforms=augmentation)
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=opt.batch_size, 
                                  shuffle=True, 
                                  num_workers=opt.num_workers)
    # val dataset
    val_dataset = TargetDataset(opt.data_path, mode='val', transforms=augmentation)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=opt.batch_size,
                                shuffle=False,
                                num_workers=opt.num_workers)
    
    # test dataset
    test_dataset = TargetDataset(opt.data_path, mode='test', transforms=augmentation)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=opt.batch_size,
                                 shuffle=False,
                                 num_workers=opt.num_workers)
    
    dataset = {'train': train_dataloader, 'val': val_dataloader, 'test': test_dataloader}
    return dataset


# TODO : ImageDataset from h5 file
class ImageDataset(Dataset):
    def __init__(self, path, mode='train', transforms=None):
        self.path = path
        self.mode = mode
        #self.mtype = ['Original', 'Deepfakes']
        self.mtype = ['Original', 'Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
        if transforms is None:
           self.transforms = transforms.ToTensor()
        else:
           self.transforms = transforms

        self.videos = []
        self.labels = []
        self.mtype_index = []
        self.data_list = None
        for i, m in enumerate(self.mtype):
            path = os.path.join(self.path, f'{m}.h5')
            with h5py.File(path, 'r') as f:
                video_keys = sorted(list(f.keys()))
                if self.mode == 'train':
                    #video_keys = video_keys[:5]
                    video_keys = video_keys[:int(len(video_keys)*0.8)]
                elif self.mode == 'val':
                    #video_keys = video_keys[5:8]
                    video_keys = video_keys[int(len(video_keys)*0.8):int(len(video_keys)*0.9)]
                elif self.mode == 'test':
                    #video_keys = video_keys[8:11]
                    video_keys = video_keys[int(len(video_keys)*0.9):]

                self.videos += video_keys
                self.labels += [0 if path.find('Original') >= 0 else 1 for _ in range(len(video_keys))] # 0: real, 1: fake
                self.mtype_index += [i for _ in range(len(video_keys))]

    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, index):
        if self.data_list is None:
            self.data_list = []
            for m in self.mtype:
                self.data_list.append(h5py.File(os.path.join(self.path, f'{m}.h5'), 'r', swmr=True))

        data_file = self.data_list[self.mtype_index[index]]
        clip = data_file[self.videos[index]]

        frame_idx = np.random.randint(0, len(clip))
        frame = clip[frame_idx]
        frame = Image.fromarray(frame[...,::-1])    # BGR2RGB

        frame = self.transforms(frame)

        data = {'frame': frame, 'label': self.labels[index]}
        return data
        
# TODO : ImageDataset from h5 file
class DFDCDataset(Dataset):
    def __init__(self, path, mode='train', transforms=None):
        self.path = path
        self.mode = mode
        self.sets = [f'dfdc_{i:02}' for i in range(5)]

        if transforms is None:
           self.transforms = transforms.ToTensor()
        else:
           self.transforms = transforms

        self.videos = []
        self.labels = []
        self.set_index = []
        self.data_list = None

        for i, set_name in enumerate(self.sets):
            set_path = os.path.join(self.path, set_name)
            video_keys = next(os.walk(set_path))[1]
            
            if self.mode == 'train':
                #video_keys = video_keys[:5]
                video_keys = video_keys[:int(len(video_keys)*0.8)]
            elif self.mode == 'val':
                #video_keys = video_keys[5:8]
                video_keys = video_keys[int(len(video_keys)*0.8):int(len(video_keys)*0.9)]
            elif self.mode == 'test':
                #video_keys = video_keys[8:11]
                video_keys = video_keys[int(len(video_keys)*0.9):]

            self.videos += video_keys

            label_path = os.path.join(set_path, 'label.json')
            label_file = open(label_path, encoding="UTF-8")
            label_data = json.loads(label_file.read())

            self.labels += [0 if label_data[video_key] == "REAL" else 1 for video_key in video_keys]
            self.set_index += [i for _ in range(len(video_keys))]

    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, index):
        if self.data_list is None:
            self.data_list = []
            for set_name in self.sets:
                self.data_list.append(os.path.join(self.path, set_name))

        
        data_path = self.data_list[self.set_index[index]]
        clip_path = os.path.join(data_path, self.videos[index])
        frames = os.listdir(clip_path)

        frame_idx = np.random.randint(0, len(frames))
        frame_path = os.path.join(clip_path, frames[frame_idx])
        frame = cv2.imread(frame_path)
        frame = Image.fromarray(frame[...,::-1])    # BGR2RGB

        frame = self.transforms(frame)

        data = {'frame': frame, 'label': self.labels[index]}
        return data
        