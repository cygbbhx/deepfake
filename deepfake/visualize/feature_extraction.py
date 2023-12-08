import os #, sys
from omegaconf import OmegaConf
from argparse import ArgumentParser

import sys
sys.path.append('/workspace/deepfake/deepfake')
from experiment.engine.trainer import Trainer
from experiment.dataloader.ImageDataset import *
from experiment.dataloader.VideoDataset import *
from experiment.model.xception import XceptionNet
from experiment.model.i3d import I3D, InceptionI3d
from experiment.model.i3d import DANN_I3D
from experiment.model.model import DANN_InceptionV3
import torch.nn.functional as F
import torch.nn as nn
#from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
from tqdm import tqdm
from torchvision import models

parser = ArgumentParser('feature extraction')
parser.add_argument('-c', '--config', type=str, help='config file path')
parser.add_argument('-w', '--weight', type=str, help='model weight file path')
parser.add_argument('--a', action='store_true', help='take every data into account. filter only correctly predicted ones in default')
parser.add_argument('--v', action='store_true', help='use video data. uses image in default')


if __name__ == '__main__':
    args = parser.parse_args()
    
    if args.config is None:
        exp_name = args.weight.split('/')[-1].split('.')[0] 
        data_type = "video" if args.v else "image"
    else:    
        opt = OmegaConf.load(args.config)
        exp_name = opt.EXP_NAME
        data_type = opt.DATA.get('type', "image")
    if args.a:
        exp_name = f'{exp_name}_all'

    os.makedirs(f'visualize/features/{exp_name}', exist_ok=True)
    device = 'cuda:0'

    # Model
    if 'DANN' in exp_name and 'I3D' in exp_name:
        model = DANN_I3D()
    elif 'Inception_DANN' in exp_name:
        model = DANN_InceptionV3()
        model.feature.eval()
    elif 'Inception' in exp_name:
        model = models.inception_v3(weights='Inception_V3_Weights.IMAGENET1K_V1')
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)

        feat_model = models.inception_v3(weights='Inception_V3_Weights.IMAGENET1K_V1')
        feat_model.fc = nn.Linear(num_ftrs, 2)    
        feat_model.to(device)
    elif 'i3d' in exp_name:
        model = I3D(opt.MODEL)
    elif 'I3D' in exp_name:
        model = InceptionI3d(num_classes=2)
    else:
        model = XceptionNet(opt.MODEL)

    model.to(device)
    checkpoint = torch.load(args.weight) 
    model.load_state_dict(checkpoint)

    if 'Inception' in exp_name and 'DANN' not in exp_name:
        feat_model.load_state_dict(checkpoint)
        feat_model.fc = nn.Identity()
        feat_model.eval()
        model.extract_features = feat_model


    if data_type == "video":    
        print("Using video dataset...")    
        img_size = 224
        dataset_mapping = {
                'ff': FFVideoDataset,
                'celeb': CelebVideoDataset,
                'dfdc': DFDCVideoDataset,
                'vfhq': VFHQVideoDataset,
                'dff' : DFFVideoDataset
            }
    else:
        img_size = 299
        dataset_mapping = {
                'ff': FFImageDataset,
                'celeb': CelebImageDataset,
                'dfdc': DFDCImageDataset,
                'vfhq': VFHQImageDataset,
                'dff' : DFFImageDataset
            }

    augmentation = T.Compose([
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                ])

    dataset_list = ['ff', 'celeb', 'dfdc', 'vfhq', 'dff']
    sample_num = 250    
    softmax = nn.Softmax(dim=1)
    model.eval()

    for target_ds in dataset_list:
        print(f"extracting features from {target_ds}")
        dataset_class = dataset_mapping[target_ds]
        # if opt.DATA.type == "image":
        if data_type == "video":        
            dataset = dataset_class(mode='test', transforms=augmentation, num_samples=32, interval=0)
        else:
            dataset = dataset_class(mode='test', transforms=augmentation)

        if len(dataset) < sample_num:    
            if data_type == "video":            
                dataset = dataset_class(mode='train', transforms=augmentation, num_samples=32, interval=0)
            else:
                dataset = dataset_class(mode='train', transforms=augmentation)

        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

        features = None
        feature_labels = []
        feature_mtypes = []
        count = 0

        with torch.no_grad():
            while count < sample_num:
                data = next(iter(dataloader))
                frames = data['frame'].to(device)
                labels = data['label'].to(device)
                if target_ds == 'ff':
                    mtypes = data['mtype'].to(device)

                # Forward pass
                if "DANN" in exp_name:                
                    outputs, _ = model(frames, 0)
                else:
                    outputs = model(frames)
                
                # For DANN
                outputs = softmax(outputs)
                _, predicted = torch.max(outputs.data, 1)
                
                if args.a:
                    correct_indices = torch.nonzero(predicted == predicted).squeeze().cpu()
                else:
                    correct_indices = torch.nonzero(predicted == labels).squeeze().cpu()
                    print(f"{count + len(correct_indices.size())}/{sample_num} predicted correctly...")

                    if len(correct_indices.size()) == 0:
                        continue
                    
                if len(correct_indices) > sample_num - count:
                    correct_indices = correct_indices[:sample_num - count]

                count += len(correct_indices)
                feature = model.extract_features(frames)

                if len(feature.shape) > 2:
                    feature = torch.flatten(feature.squeeze(), 1)

                x = feature[correct_indices].cpu()
                y = labels[correct_indices].cpu()
                feature_labels.append(y)

                if target_ds == 'ff':            
                    m = mtypes[correct_indices].cpu()                
                    feature_mtypes.append(m)

                if features is None:
                    features = x
                else:
                    features = np.vstack((features, x))

        feature_labels = np.concatenate(feature_labels).ravel()
                    
        np.save(f'visualize/features/{exp_name}/{target_ds}_features', features)
        np.save(f'visualize/features/{exp_name}/{target_ds}_labels', feature_labels)   
        if target_ds == 'ff':   
            feature_mtypes = np.concatenate(feature_mtypes).ravel()
            np.save(f'visualize/features/{exp_name}/{target_ds}_mtypes', feature_mtypes)    
