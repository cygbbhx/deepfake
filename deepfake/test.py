import os #, sys
from omegaconf import OmegaConf
from argparse import ArgumentParser

from experiment.engine.trainer import Trainer
from experiment.dataloader.ImageDataset import get_image_dataset
from experiment.dataloader.VideoDataLoader import get_video_dataset
from experiment.model.xception import XceptionNet
from experiment.model.i3d import I3D
from experiment.engine.tester import Tester
from datetime import datetime
from pytz import timezone
import logging
import torch
from collections import OrderedDict
import numpy as np
import random
import torch.backends.cudnn as cudnn

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(0)

parser = ArgumentParser('Deepface Training Script')
parser.add_argument('config', type=str, help='config directory path')
parser.add_argument('-w', '--weights', type=str, default=None, help='test model weights')
parser.add_argument('-b', '--batch_size', type=int, default=None, help='test data batch size')

if __name__ == '__main__':
    args = parser.parse_args()
    opt = OmegaConf.load(os.path.join(args.config, "config.yaml"))
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
    
    if weights == None:
        print('model weights not assigned, using the best weight')
        args.weights = 'best.pt'
        weights = os.path.join(opt.TRAIN.ckpt_dir, 'best.pt')

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
    console_logging_format = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    console_logging_format.converter = lambda *args: datetime.now(tz=timezone('Asia/Seoul')).timetuple()

    # log 출력
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(console_logging_format)
    logger.addHandler(stream_handler)

    # log를 파일에 출력
    file_handler = logging.FileHandler(os.path.join(opt.TRAIN.ckpt_dir, 'test_log'))
    file_handler.setFormatter(console_logging_format)
    logger.addHandler(file_handler)

    # test
    test_data_list = ['ff', 'celeb', 'dfdc', 'vfhq', 'dff']
    logger.debug(f'=> Tested with model weight {args.weights}')
    if args.batch_size != None:
        opt.DATA.batch_size = args.batch_size

    for dataset in test_data_list:
        opt.DATA.test_data_name = dataset
        logger.debug(f'----- Test dataset: {dataset}')
    
        # Load Dataloader
        dataloader = get_image_dataset(opt.DATA) if opt.DATA.type == 'image' else get_video_dataset(opt.DATA)

        # dataloader = get_video_dataset(opt.DATA)
        tester = Tester(opt, dataloader, model, logger)

        if opt.DATA.type == 'image':
            tester.test("test")
        else:
            tester.test_each("test")
        logger.debug('------------------------------')
    