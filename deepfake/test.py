import os #, sys
from omegaconf import OmegaConf
from argparse import ArgumentParser

from experiment.engine.trainer import Trainer
from experiment.dataloader.ImageDataset import get_image_dataset
from experiment.dataloader.VideoDataLoader import get_video_dataset
from experiment.model.xception import XceptionNet
from experiment.model.i3d import I3D
from experiment.engine.tester import Tester
#from torch.utils.tensorboard import SummaryWriter
import logging
import torch
from collections import OrderedDict

SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

parser = ArgumentParser('Deepface Training Script')
parser.add_argument('config', type=str, help='config file path')
parser.add_argument('weights', type=str, default=None, help='test model weights')

if __name__ == '__main__':
    args = parser.parse_args()
    opt = OmegaConf.load(args.config)
    opt.DATA.augSelf = opt.DATA.get('augSelf', False)
    opt.DATA.type = opt.DATA.get('type', 'image')
    #writer = SummaryWriter()

    # Model

    model_list = {
        'Xception': XceptionNet,
        'I3D': I3D
    }

    model = model_list[opt.MODEL.name](opt.MODEL)
    
    weights = opt.get('WEIGHTS', args.weights)
    
    if (weights == None):
        print('model weights not provided!')
        exit

    checkpoint = torch.load(weights, map_location='cuda:0') 

    # if trained on multigpu...
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        if 'module' in k:
            name = k[7:] # remove `module.`
            new_state_dict[name] = v    
    if new_state_dict:
        checkpoint = new_state_dict

    model.load_state_dict(checkpoint)

    # Logger
    # log train/val loss, acc, etc.
    # 로그 생성
    logger = logging.getLogger()

    # 로그의 출력 기준 설정
    logger.setLevel(logging.DEBUG)

    # log 출력 형식
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # log 출력
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # log를 파일에 출력
    file_handler = logging.FileHandler(opt.LOG_FILENAME)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # test
    test_data_list = ['ff', 'celeb', 'dfdc', 'vfhq', 'dff']
    print(f"Using {opt.DATA.type} dataset ...") 
    logger.debug(f'=> Tested with model weight {args.weights}')

    for dataset in test_data_list:
        opt.DATA.test_data_name = dataset
        logger.debug(f'----- Test dataset: {dataset}')
    
        # Load Dataloader
        dataloader = get_image_dataset(opt.DATA) if opt.DATA.type == 'image' else get_video_dataset(opt.DATA)

        # dataloader = get_video_dataset(opt.DATA)
        tester = Tester(opt, dataloader, model, logger)
        tester.test_each("test")
        logger.debug('------------------------------')
    