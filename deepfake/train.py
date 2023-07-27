import os #, sys
from omegaconf import OmegaConf
from argparse import ArgumentParser

from experiment.engine.trainer import Trainer
from experiment.dataloader.ImageDataset import get_dataset
from experiment.model.model import XceptionNet
#from torch.utils.tensorboard import SummaryWriter
import logging

parser = ArgumentParser('Deepface Training Script')
parser.add_argument('config', type=str, help='config file path')

if __name__ == '__main__':
    args = parser.parse_args()
    opt = OmegaConf.load(args.config)
    #writer = SummaryWriter()

    # Load Dataloader
    dataloader = get_dataset(opt.DATA)

    # Model
    model = XceptionNet(opt.MODEL)

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
    file_handler = logging.FileHandler('sohyun/deepfake/checkpoints/' + opt.LOG_FILENAME)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # BANMo System    
    trainer = Trainer(opt, dataloader, model, logger)

    # train
    trainer.train()
