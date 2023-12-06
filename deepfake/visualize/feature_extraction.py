import os #, sys
from omegaconf import OmegaConf
from argparse import ArgumentParser

import sys
sys.path.append('/workspace/deepfake/deepfake')
from experiment.engine.trainer import Trainer
from experiment.dataloader.ImageDataset import *
from experiment.dataloader.VideoDataset import *
from experiment.model.xception import XceptionNet
from experiment.model.i3d import I3D
from experiment.model.i3d import DANN_I3D
from experiment.model.model import DANN_InceptionV3
import torch.nn.functional as F
import torch.nn as nn
#from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
from tqdm import tqdm

parser = ArgumentParser('feature extraction')
parser.add_argument('-c', '--config', type=str, help='config file path')
parser.add_argument('-w', '--weight', type=str, help='model weight file path')
parser.add_argument('--a', action='store_true', help='take every data into account. filter only correctly predicted ones in default')

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

if __name__ == '__main__':
    args = parser.parse_args()

    os.makedirs('visualize/features', exist_ok=True)
    device = 'cuda:0'

    # Model
    # model = XceptionNet(opt.MODEL)
    model = DANN_I3D()
    # model = I3D(opt.MODEL)
    model.to(device)
    checkpoint = torch.load(args.weight) 

    model.load_state_dict(checkpoint)
    model.include_top = False

    augmentation = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                ])

    # dataset_mapping = {
    #         'ff': FFImageDataset,
    #         'celeb': CelebImageDataset,
    #         'dfdc': DFDCImageDataset,
    #         'vfhq': VFHQImageDataset,
    #         'dff' : DFFImageDataset
    #     }

    dataset_mapping = {
            'ff': FFVideoDataset,
            'celeb': CelebVideoDataset,
            'dfdc': DFDCVideoDataset,
            'vfhq': VFHQVideoDataset,
            'dff' : DFFVideoDataset
        }

    dataset_list = ['ff', 'celeb', 'dfdc', 'vfhq', 'dff']
    sample_num = 250    
    relu = nn.ReLU(inplace=True)
    softmax = nn.Softmax(dim=1)

    for target_ds in dataset_list:
        print(f"extracting features from {target_ds}")
        data_type = dataset_mapping[target_ds]
        dataset = data_type(mode='test', transforms=augmentation, num_samples=32, interval=0)

        if len(dataset) < sample_num:
            dataset = data_type(mode='train', transforms=augmentation, num_samples=32, interval=0)

        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

        features = None
        feature_labels = []
        count = 0

        with torch.no_grad():
            while count < sample_num:
                data = next(iter(dataloader))
                frames = data['frame'].to(device)
                labels = data['label'].to(device)

                # Forward pass
                outputs, _ = model(frames, 0)
                outputs = softmax(outputs)
                _, predicted = torch.max(outputs.data, 1)
                
                if args.a:
                    correct_indices = torch.nonzero(predicted == predicted).squeeze().cpu()
                else:
                    correct_indices = torch.nonzero(predicted == labels).squeeze().cpu()
                    print(f"{count + len(correct_indices)}/{sample_num} predicted correctly...")
                    
                if len(correct_indices) > sample_num - count:
                    correct_indices = correct_indices[:sample_num - count]

                count += len(correct_indices)
                feature = model.extract_features(frames)

                if len(feature.shape) > 2:
                    feature = torch.flatten(feature.squeeze(), 1)

                x = feature[correct_indices].cpu()
                y = labels[correct_indices].cpu()
                
                feature_labels.append(y)

                if features is None:
                    features = x
                else:
                    features = np.vstack((features, x))

        feature_labels = np.concatenate(feature_labels).ravel()
                    
        np.save(f'visualize/features/{target_ds}_features', features)
        np.save(f'visualize/features/{target_ds}_labels', feature_labels)    
