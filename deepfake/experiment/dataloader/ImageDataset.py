import os
import h5py
from PIL import Image
import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image


def get_dataset(opt):

    augmentation = transforms.Compose([
        transforms.Resize((opt.image_size, opt.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    # train dataset
    train_dataset = DFDCDataset(opt.data_path, mode='train', transforms=augmentation)
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=opt.batch_size, 
                                  shuffle=True, 
                                  num_workers=opt.num_workers)
    # val dataset
    val_dataset = DFDCDataset(opt.data_path, mode='val', transforms=augmentation)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=opt.batch_size,
                                shuffle=False,
                                num_workers=opt.num_workers)
    
    # test dataset
    test_dataset = DFDCDataset(opt.data_path, mode='test', transforms=augmentation)
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
        self.mtype = [f'train_{i}' for i in range(5)]
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

                self.labels += [f.get(video_key)['labels'][()] for video_key in video_keys] # 0: real, 1: fake
                self.mtype_index += [i for _ in range(len(video_keys))]

    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, index):
        if self.data_list is None:
            self.data_list = []
            for m in self.mtype:
                self.data_list.append(h5py.File(os.path.join(self.path, f'{m}.h5'), 'r', swmr=True))

        data_file = self.data_list[self.mtype_index[index]]
        clip = data_file[self.videos[index]]['frames']

        frame_idx = np.random.randint(0, len(clip))
        frame = clip[frame_idx]
        frame = Image.fromarray(frame[...,::-1])    # BGR2RGB

        frame = self.transforms(frame)

        data = {'frame': frame, 'label': self.labels[index]}
        return data
        