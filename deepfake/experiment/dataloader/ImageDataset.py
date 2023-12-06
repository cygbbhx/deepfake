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
import torch
import json

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

def sampleFrame(video_dir, transforms):
    frame_keys = sorted(os.listdir(video_dir))
    frame_key = random.choice(frame_keys)
    frame = Image.open(os.path.join(video_dir, frame_key))
    frame = transforms(frame)

    return frame



def get_image_dataset(opt):

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
    # test_data_path에 경로 있으면 cross evaluation
    if opt.get("test_data_name", None) is None:
        test_data_name = opt.train_data_name
    else:
        test_data_name = opt.test_data_name

    dataset_mapping = {
        'ff': FFImageDataset,
        'celeb': CelebImageDataset,
        'dfdc': DFDCImageDataset,
        'vfhq': VFHQImageDataset,
        'dff' : DFFImageDataset
    }

    # Specify the dataset name
    train_data_name = opt.train_data_name
    assert train_data_name in dataset_mapping, f"Unsupported dataset name: {train_data_name}"
    assert test_data_name in dataset_mapping, f"Unsupported dataset name: {test_data_name}"

    # Create the appropriate dataset class based on the name
    test_data_class = dataset_mapping[test_data_name]

    multiple = opt.get('multiple', False)

    if multiple:    
        datasets = opt.get('datasets', ['ff', 'celeb', 'vfhq', 'dff'])
        train_datasets = []
        val_datasets = []
        # interval/ num_sample options not applied yet here
        for dataset in datasets:
            train_data_class = dataset_mapping[dataset]
            train_dataset = train_data_class(mode='train', transforms=augmentation)
            val_dataset = train_data_class(mode='val', transforms=augmentation)

            train_datasets.append(train_dataset)
            val_datasets.append(val_dataset)
        
        train_dataset = torch.utils.data.ConcatDataset(train_datasets)
        val_dataset = torch.utils.data.ConcatDataset(val_datasets)
    else:
        train_data_class = dataset_mapping[train_data_name]
        train_dataset = train_data_class(mode='train', transforms=augmentation)
        val_dataset = train_data_class(mode='val', transforms=augmentation)
    
    
    train_dataloader = DataLoader(train_dataset, 
                                batch_size=opt.batch_size, 
                                shuffle=True, 
                                num_workers=opt.num_workers)

    val_dataloader = DataLoader(val_dataset,
                                batch_size=opt.batch_size,
                                shuffle=True,
                                num_workers=opt.num_workers) 

    test_dataset = test_data_class(mode='test', transforms=augmentation)
    test_dataloader = DataLoader(test_dataset,
                                batch_size=opt.batch_size,
                                shuffle=False,
                                num_workers=opt.num_workers)

    dataset = {'train': train_dataloader, 'val': val_dataloader, 'test': test_dataloader}
    return dataset


class BaseImageDataset(Dataset):
    def __init__(self, name, path, mode='train', transforms=None):
        self.name = name
        self.path = path
        self.mode = mode
        
        if transforms is None:
            self.transforms = T.ToTensor()
        else:
            self.transforms = transforms
        
        self.videos = []
        self.labels = []
        
        self.mtype_index = []
        
        # Subclasses should define this during init
        self.mtype = None  
        self.iter_path = None 
        
        # load data after init
        # self._load_data()

    def _load_data(self):
        assert self.iter_path is not None, "video directories are not set"
        assert self.mtype is not None, "manipulation types are not set"

        for i, video_dir in enumerate(self.iter_path):
            assert os.path.exists(video_dir), f"{video_dir} does not exist"

            all_video_keys = sorted(os.listdir(video_dir))
            final_video_keys = self._get_splits(all_video_keys)

            video_dirs = [os.path.join(video_dir, video_key) for video_key in final_video_keys]
            self.videos += video_dirs
            self.labels += self._get_labels(video_dir, final_video_keys)
            self.mtype_index += [i for _ in range(len(final_video_keys))]

        if self.mode == 'train':
            self._oversample()

    def _oversample(self):
        ## OVERSAMPLING
        # Count the number of videos with label 0 and 1
        count_label_0 = self.labels.count(0)
        count_label_1 = self.labels.count(1)

        duplicateNum = count_label_1 // count_label_0 - 1
        additional_samples = count_label_1 % count_label_0

        # Oversample video keys with label 0
        real_videos = [video_dir for video_dir, label in zip(self.videos, self.labels) if label == 0]
        oversampled_videos = real_videos * duplicateNum + random.sample(real_videos, additional_samples)
        self.labels += [0] * (len(oversampled_videos))

        # print(f'REAL: {count_label_0} : FAKE: {count_label_1} found')
        # print(f'oversampled {len(oversampled_videos)}')

        #This should be fixed ... mtype is not used anyway tho
        #self.mtype_index += [self.mtype_index for _ in range(len(self.videos), len(self.videos) + len(oversampled_videos))]
        self.videos += [video_dir for video_dir in oversampled_videos]         

    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, index):
        video_dir = self.videos[index]
        frame_keys = sorted(os.listdir(video_dir))
        frame_key = random.choice(frame_keys)
        frame = Image.open(os.path.join(video_dir, frame_key))
        frame = self.transforms(frame)
        data = {'frame': frame, 'label': self.labels[index]}
        return data

    def _get_splits(self, video_keys):
        # Default split logic. Redefine the function if needed
        if self.mode == 'train':
            video_keys = video_keys[:int(len(video_keys)*0.8)]
        elif self.mode == 'val':
            video_keys = video_keys[int(len(video_keys)*0.8):int(len(video_keys)*0.9)]
        elif self.mode == 'test':
            video_keys = video_keys[int(len(video_keys)*0.9):]

        return video_keys

    def _get_labels(self, video_dir, video_keys):
        # Subclasses should implement this method
        raise NotImplementedError("Subclasses should implement the _get_labels() method")


class FFImageDataset(BaseImageDataset):
    def __init__(self, path='/workspace/dataset2/ff', mode='train', transforms=None):
        super().__init__('ff', path, mode, transforms)

        self.iter_path = [os.path.join(self.path, 'original_sequences', 'raw', 'crop_jpg'), 
                            os.path.join(self.path, 'manipulated_sequences', 'Deepfakes', 'raw', 'crop_jpg'),
                            os.path.join(self.path, 'manipulated_sequences', 'Face2Face', 'raw', 'crop_jpg'),
                            os.path.join(self.path, 'manipulated_sequences', 'FaceSwap', 'raw', 'crop_jpg'),
                            os.path.join(self.path, 'manipulated_sequences', 'NeuralTextures', 'raw', 'crop_jpg')]                  
        
        self.mtype = ['Original', 'Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
        self._load_data()

    def _get_labels(self, video_dir, video_keys):
        return [0 if video_dir.find('original') >= 0 else 1 for _ in range(len(video_keys))]
    
    def _get_mtype(self, video_dir):
        video_mtype = 0
        for i, mtype in enumerate(self.mtype):
            if video_dir.find(mtype) >= 0:
                video_mtype = i
                break        
        return video_mtype
    
    def _get_identity(self, video_mtype, video_dir):
        # default identity = origin
        video_identity = video_dir.split('/')[-1]
        # get the source identity of face using file format target_source 
        if self.mtype[video_mtype] == 'Deepfakes' or self.mtype[video_mtype] == 'FaceSwap':
            video_identity = video_dir.split('/')[-1].split('_')[-1]
        elif self.mtype[video_mtype] == 'Face2Face' or self.mtype[video_mtype] == 'NeuralTextures':
            # use target identity for else since face2face and NT do not change identity
            video_identity = video_dir.split('/')[-1].split('_')[0]

        video_identity = int(video_identity)
        return video_identity

    # use this as getitem function for identity-based triplets
    def __getTriplets__(self, index):
        video_dir = self.videos[index]
        video_mtype = self._get_mtype(video_dir)
        video_identity = self._get_identity(video_mtype, video_dir)

        #remove .jpg and convert into int
        current_label = self.labels[index]
        positive_indices = []
        negative_indices = []
        negative_extras = []

        for i, label in enumerate(self.labels):
            target_video = self.videos[i]
            target_mtype = self._get_mtype(target_video)
            target_id = self._get_identity(target_mtype, target_video)

            if label == current_label and target_id != video_identity:
                positive_indices.append(i)
            if label != current_label:
                if target_id == video_identity:
                    negative_indices.append(i)
                else:
                    negative_extras.append(i)

        if len(negative_indices) == 0:
            #print(f"Data: {video_dir}\n => negative pairs not exist, using negative with different id")
            negative_indices = negative_extras
        
        positive_index = random.choice(positive_indices)
        positive_video_dir = self.videos[positive_index]

        negative_index = random.choice(negative_indices)
        negative_video_dir = self.videos[negative_index]

        anchor_frame = sampleFrame(video_dir, self.transforms)
        positive_frame = sampleFrame(positive_video_dir, self.transforms)
        negative_frame = sampleFrame(negative_video_dir, self.transforms)

        frames = [anchor_frame, positive_frame, negative_frame]

        data = {'frames': frames, 'label': self.labels[index]}
        return data

class DFFImageDataset(BaseImageDataset):
    def __init__(self, path='/workspace/dff', mode='train', transforms=None):
        super().__init__('dff', path, mode, transforms)
        folders = os.listdir(os.path.join(self.path, 'manipulated_videos'))

        self.iter_path = [os.path.join(self.path, 'manipulated_videos', folder) for folder in folders]
        self.iter_path += [os.path.join(self.path, 'original_sequences/raw/crop_jpg')]
              
        self.mtype = folders
        self.mtype += ['Original']
        self._load_data()

    def _get_labels(self, video_dir, video_keys):
        return [0 if video_dir.find('original') >= 0 else 1 for _ in range(len(video_keys))]


class CelebImageDataset(BaseImageDataset):
    def __init__(self, path='/workspace/dataset2/celeb', mode='train', transforms=None):
        super().__init__('celeb', path, mode, transforms)
        self.iter_path = [os.path.join(self.path, 'Celeb-real', 'crop_jpg'),
                            os.path.join(self.path, 'Celeb-synthesis', 'crop_jpg'),
                            os.path.join(self.path, 'YouTube-real', 'crop_jpg')]

        with open(self.path + "/List_of_testing_videos.txt", "r") as f:
            self.test_list = f.readlines()
            self.test_list = [x.split("/")[-1].split(".mp4")[0] for x in self.test_list]

        self.mtype = ['Celeb-real', 'Celeb-synthesis', 'YouTube-real']
        self._load_data()

    def _get_splits(self, video_keys):
        if self.mode == 'test': 
            video_keys = [x for x in self.test_list if x in video_keys]
        elif self.mode == 'train':
            video_keys = [x for x in video_keys if x not in self.test_list]
            video_keys = video_keys[:int(len(video_keys)*0.8)]
        elif self.mode == 'val':
            video_keys = [x for x in video_keys if x not in self.test_list]
            video_keys = video_keys[int(len(video_keys)*0.8):]

        return video_keys

    def _get_labels(self, video_dir, video_keys):
        return [0 if video_dir.find('real') >= 0 else 1 for _ in range(len(video_keys))]


class DFDCImageDataset(BaseImageDataset):
    def __init__(self, path='/workspace/dataset1/dfdc_preprocessed', mode='train', transforms=None):
        super().__init__('dfdc', path, mode, transforms)
        self.mtype = [f'dfdc_{i:02}' for i in range(50)]
        self.iter_path = [os.path.join(self.path, set) for set in self.mtype]
        self._load_data()

    def _get_labels(self, video_dir, video_keys):
        label_path = os.path.join(video_dir, 'label.json')
        label_file = open(label_path, encoding="UTF-8")
        label_data = json.loads(label_file.read())

        return [0 if label_data[video_key] == "REAL" else 1 for video_key in video_keys]
   
    def _get_splits(self, video_keys):
        video_keys.remove('label.json')
        return super()._get_splits(video_keys)


class VFHQImageDataset(BaseImageDataset):
    def __init__(self, path='/workspace/dataset2/vfhq', mode='train', transforms=None):
        super().__init__('vfhq', path, mode, transforms)
        self.iter_path = [os.path.join(self.path, 'crop_jpg')]
        self._load_data()
        
    def _load_data(self):
        self.mtype = ['vfhq']

        mode_mapping  = {'train': 'training', 'val': 'validation', 'test': 'test'}
        video_key_path = os.path.join(self.iter_path[0], mode_mapping[self.mode])
        video_keys = sorted(os.listdir(video_key_path))
        video_dirs = [os.path.join(video_key_path, video_key) for video_key in video_keys]
        self.videos += video_dirs
        self.labels += self._get_labels(video_key_path, video_keys)

    def _get_labels(self, video_dir, video_keys):
        return [1 if key.split('_')[2][0] == 'f' else 0 for key in video_keys]
    

if __name__ == "__main__":
    data_path = '/root/datasets/celeb'
    dataset = ImageDataset('celeb', data_path, False, crop_ratio=1.7, mode='train')
    
    print(len(dataset))
    frame = dataset[0]['frame']
    # save_image(frame, 'result.jpg')
    
    # num_frames = len(dataset)
    # frame_idx = np.random.choice(num_frames, size=100, replace=False)
    # for idx in frame_idx:
    #     data = dataset[idx]
    #     frame = data['frame']
    #     save_image(frame, f'/root/result/random_result/{idx}.png')
   
    