import os 
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import random
import torch
from PIL import Image
import json 
import numpy as np

class BaseVideoDataset(Dataset):
    def __init__(self, name, path, mode='train', transforms=None, **kwargs):
        self.name = name
        self.path = path
        self.mode = mode

        self.num_samples = kwargs['num_samples']
        self.interval = kwargs['interval']
        
        if transforms is None:
            self.transforms = T.ToTensor()
        else:
            self.transforms = transforms
        
        self.videos = []
        self.labels = []
        
        self.mtype_index = []
        self.clips = []
        self.clip_src_idx = []
        
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
        else:
            self._get_clips()

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

    def _get_clips(self):
        for i, video_dir in enumerate(self.videos):
            frame_keys = sorted(os.listdir(video_dir))
            frame_count = len(frame_keys)
            num_samples = self.num_samples
            interval = self.interval # UNIFORM :1,2 / SPREAD: max(total_frames // num_samples, 1)
            max_length = (num_samples - 1) * self.interval + num_samples

            for starting_point in range(0, frame_count, (num_samples-1)*interval + num_samples):
                if (interval == 0) or (frame_count <= max_length):
                    sampled_keys = frame_keys[starting_point:starting_point+num_samples]
                else:
                    sampled_indices = np.arange(starting_point, frame_count, interval)[:num_samples]
                    sampled_keys = [frame_keys[idx] for idx in sampled_indices]

                if len(sampled_keys) < num_samples:
                    break

                self.clips += [sampled_keys]
                self.clip_src_idx.append(i)

    def __len__(self):
        if self.mode == 'train':
            return len(self.videos)
        else:
            return len(self.clips)
    
    def __getitem__(self, index):
        if self.mode == 'train':
            video_dir = self.videos[index]
            frame_keys = sorted(os.listdir(video_dir))
            frame_count = len(frame_keys)
            clip_length = (self.num_samples - 1) * self.interval + self.num_samples

            if (self.interval == 0) or (frame_count <= clip_length):
                starting_point = random.randint(0, frame_count - num_samples)
                sampled_keys = frame_keys[starting_point:starting_point+self.num_samples]
            else:
                starting_point = random.randint(0, frame_count - clip_length)
                sampled_indices = np.arange(starting_point, frame_count, self.interval)[:self.num_samples]
                sampled_keys = [frame_keys[idx] for idx in sampled_indices]
        else:
            src_idx = self.clip_src_idx[index]
            video_dir = self.videos[src_idx]
            sampled_keys = self.clips[index]

        frames = []

        # Fix the randomness across the video for all frames
        state = torch.get_rng_state()   
        for frame_key in sampled_keys:
            frame = Image.open(os.path.join(video_dir, frame_key))
            torch.set_rng_state(state)
            frame = self.transforms(frame)
            frames.append(frame)

        # If contrastive learning, we get 2 tensors of frame
        if isinstance(frame, list):
            frames_tensor1 = torch.stack([frame1 for frame1, frame2 in frames], dim=0).transpose(0,1)
            frames_tensor2 = torch.stack([frame2 for frame1, frame2 in frames], dim=0).transpose(0,1)

            frame_data = [frames_tensor1, frames_tensor2]
        else:
            frame_data = torch.stack(frames, dim=0).transpose(0,1)

        if self.mode == 'train':
            data = {'frame': frame_data, 'label': self.labels[index]}
        else:
            data = {'video': src_idx, 'frame': frame_data, 'label': self.labels[src_idx]}

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


class FFVideoDataset(BaseVideoDataset):
    def __init__(self, path='/workspace/dataset2/ff', mode='train', transforms=None, **kwargs):
        super().__init__('ff', path, mode, transforms, **kwargs)

        self.iter_path = [os.path.join(self.path, 'original_sequences', 'raw', 'crop_jpg'), 
                            os.path.join(self.path, 'manipulated_sequences', 'Deepfakes', 'raw', 'crop_jpg'),
                            os.path.join(self.path, 'manipulated_sequences', 'Face2Face', 'raw', 'crop_jpg'),
                            os.path.join(self.path, 'manipulated_sequences', 'FaceSwap', 'raw', 'crop_jpg'),
                            os.path.join(self.path, 'manipulated_sequences', 'NeuralTextures', 'raw', 'crop_jpg')]                  
        
        self.mtype = ['Original', 'Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
        self._load_data()

    def _get_labels(self, video_dir, video_keys):
        return [0 if video_dir.find('original') >= 0 else 1 for _ in range(len(video_keys))]

class DFFVideoDataset(BaseVideoDataset):
    def __init__(self, path='/workspace/dataset1/dff_preprocessed', mode='train', transforms=None, **kwargs):
        super().__init__('dff', path, mode, transforms, **kwargs)
        folders = os.listdir(os.path.join(self.path, 'manipulated_videos'))

        self.iter_path = [os.path.join(self.path, 'manipulated_videos', folder) for folder in folders]
        self.iter_path += [os.path.join(self.path, 'original_sequences/raw/crop_jpg')]
              
        self.mtype = folders
        self.mtype += ['Original']
        self._load_data()

    def _get_labels(self, video_dir, video_keys):
        return [0 if video_dir.find('original') >= 0 else 1 for _ in range(len(video_keys))]


class CelebVideoDataset(BaseVideoDataset):
    def __init__(self, path='/workspace/dataset2/celeb', mode='train', transforms=None, **kwargs):
        super().__init__('celeb', path, mode, transforms, **kwargs)
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


class DFDCVideoDataset(BaseVideoDataset):
    def __init__(self, path='/workspace/dataset1/dfdc_preprocessed', mode='train', transforms=None, **kwargs):
        super().__init__('dfdc', path, mode, transforms, **kwargs)
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


class VFHQVideoDataset(BaseVideoDataset):
    def __init__(self, path='/workspace/dataset2/vfhq', mode='train', transforms=None, **kwargs):
        super().__init__('vfhq', path, mode, transforms, **kwargs)
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

        if self.mode == 'train':
            self._oversample()
        else:
            self._get_clips()
            
    def _get_labels(self, video_dir, video_keys):
        return [1 if key.split('_')[2][0] == 'f' else 0 for key in video_keys]
    

if __name__ == "__main__":
    data_path = '/root/datasets/celeb'
    dataset = FFVideoDataset()
    
    print(len(dataset))
    frame = dataset[0]['frame']