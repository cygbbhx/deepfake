import os #, sys
from omegaconf import OmegaConf
from argparse import ArgumentParser

from experiment.engine.trainer import Trainer
from experiment.dataloader.ImageDataset import get_test_dataset, get_image_dataset
from experiment.model.model import XceptionNet
import torch.nn.functional as F
import torch.nn as nn
#from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
from tqdm import tqdm

parser = ArgumentParser('Deepface Training Script')
parser.add_argument('config', type=str, help='config file path')

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

if __name__ == '__main__':
    args = parser.parse_args()
    opt = OmegaConf.load(args.config)

    dataloader = get_image_dataset(opt.DATA)

    device = 'cuda:0'

    # Model
    model = XceptionNet(opt.MODEL)
    model.to(device)
    checkpoint = torch.load(os.path.join(opt.CHK_DIR, opt.WEIGHTS)) 
    model.load_state_dict(checkpoint)
    relu = nn.ReLU(inplace=True)

    activation = {}
    model.bn4.register_forward_hook(get_activation('bn4'))

    features = None
    labels = []
    count = 0

    with torch.no_grad():
        for data in tqdm(dataloader):
            frames = data['frame'].to(device)
            labels += data['label']

            # Forward pass
            outputs = model(frames)
            
            x = activation['bn4'].cpu()
            x = relu(x)
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = x.view(x.size(0), -1)

            if features is None:
                features = x
            else:
                features = np.vstack((features, x))


    trained_data = opt.CHK_DIR.split('/')[-2]
    input_data = opt.DATA.name

    np.save(f'features/{trained_data}_{input_data}_test_features', features)
    np.save(f'features/{trained_data}_{input_data}_test_labels', labels)    
