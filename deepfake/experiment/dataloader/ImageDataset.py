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
import random


def get_image_dataset(opt):
    name = opt.name

    augmentation = T.Compose([
        T.Resize((opt.image_size, opt.image_size)),
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
    
    # For cross evaluation
    # test_data_path에 경로 있으면 cross evaluation
    if opt.test_data_path == 'None':
        data_path = opt.data_path
        is_cross_eval = False
    else:
        data_path = opt.test_data_path
        is_cross_eval = True

    # train dataset
    train_dataset = ImageDataset(name, data_path, is_cross_eval, mode='train', transforms=augmentation)
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=opt.batch_size, 
                                  shuffle=True, 
                                  num_workers=opt.num_workers)
    # val dataset
    val_dataset = ImageDataset(name, data_path, is_cross_eval, mode='val', transforms=augmentation)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=opt.batch_size,
                                shuffle=False,
                                num_workers=opt.num_workers) 
    
    # test dataset
    test_dataset = ImageDataset(name, data_path, is_cross_eval, mode='test', transforms=augmentation)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=opt.batch_size,
                                 shuffle=False,
                                 num_workers=opt.num_workers)
    
    dataset = {'train': train_dataloader, 'val': val_dataloader, 'test': test_dataloader}
    return dataset


# TODO : ImageDataset from jpg file
class ImageDataset(Dataset):
    def __init__(self, name, path, is_cross_eval, mode='train', transforms=None):
        self.name = name
        self.path = path
        self.is_cross_eval = is_cross_eval
        self.mode = mode
        if self.name == 'ff':
            self.mtype = ['Original', 'Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
        elif self.name == 'celeb':
            self.mtype = ['Celeb-real', 'Celeb-synthesis', 'YouTube-real']
        elif self.name == 'dfdc':
            self.mtype = [f'dfdc_{i:02}' for i in range(8)] #fix 5 -> 50 when cropping is finished
        
        if transforms is None:
           self.transforms = T.ToTensor()
        else:
           self.transforms = transforms

        self.videos = []
        self.labels = []
        self.mtype_index = []
        self.data_list = None
        
        if self.name == 'ff':
            iter_path = [os.path.join(self.path, 'original_sequences', 'raw', 'crop_jpg'), 
                         os.path.join(self.path, 'manipulated_sequences', 'Deepfakes', 'raw', 'crop_jpg'),
                         os.path.join(self.path, 'manipulated_sequences', 'Face2Face', 'raw', 'crop_jpg'),
                         os.path.join(self.path, 'manipulated_sequences', 'FaceSwap', 'raw', 'crop_jpg'),
                         os.path.join(self.path, 'manipulated_sequences', 'NeuralTextures', 'raw', 'crop_jpg')]
        elif self.name == 'celeb':
            iter_path = [os.path.join(self.path, 'Celeb-real', 'crop_jpg'),
                         os.path.join(self.path, 'Celeb-synthesis', 'crop_jpg'),
                         os.path.join(self.path, 'YouTube-real', 'crop_jpg')]

        elif self.name == 'dfdc':
            iter_path = [os.path.join(self.path, set) for set in self.mtype]
            

        for i, each_path in enumerate(iter_path):
            video_keys = sorted(os.listdir(each_path))
            if self.is_cross_eval == True:
                video_keys = video_keys[:]
                
            else:
                if self.mode == 'train':
                    video_keys = video_keys[:int(len(video_keys)*0.8)]
                elif self.mode == 'test':
                    video_keys = video_keys[int(len(video_keys)*0.8):int(len(video_keys)*0.9)]
                elif self.mode == 'val':
                    video_keys = video_keys[int(len(video_keys)*0.9):]
            
            video_dirs = [os.path.join(each_path, video_key) for video_key in video_keys]
            self.videos += video_dirs
            if self.name == 'ff':
                self.labels += [0 if each_path.find('original') >= 0 else 1 for _ in range(len(video_keys))] # 0: real, 1: fake
            elif self.name == 'celeb':
                self.labels += [0 if each_path.find('real') >= 0 else 1 for _ in range(len(video_keys))] # 0: real, 1: fake
            elif self.name == 'dfdc':
                label_path = os.path.join(iter_path, 'label.json')
                label_file = open(label_path, encoding="UTF-8")
                label_data = json.loads(label_file.read())
                self.labels += [0 if label_data[video_key] == "REAL" else 1 for video_key in video_keys]
            self.mtype_index += [i for _ in range(len(video_keys))]

    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, index):
        video_dir = self.videos[index]
        frame_keys = sorted(os.listdir(video_dir))
        frame_key =random.choice(frame_keys)
        frame = Image.open(os.path.join(video_dir, frame_key))
        frame = self.transforms(frame)
        data = {'frame': frame, 'label': self.labels[index]}
        return data
            
    
if __name__ == "__main__":
    data_path = '/root/datasets/ff'
    dataset = ImageDataset('ff', data_path, False, mode='train')
    
    print(len(dataset))
    frame = dataset[2560]['frame']
    save_image(frame, 'result.jpg')
    
    # num_frames = len(dataset)
    # frame_idx = np.random.choice(num_frames, size=100, replace=False)
    # for idx in frame_idx:
    #     data = dataset[idx]
    #     frame = data['frame']
    #     save_image(frame, f'/root/result/random_result/{idx}.png')
   
    