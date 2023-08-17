import os #, sys
import logging
from omegaconf import OmegaConf
from argparse import ArgumentParser

import torch.nn as nn
from torchvision import models

from experiment.engine.tester import Tester
from experiment.engine.trainer import Trainer
from experiment.dataloader.ImageDataset import get_image_dataset
from experiment.dataloader.VideoDataset import get_video_dataset
from experiment.model.model import XceptionNet
from torchsummary import summary

parser = ArgumentParser('Deepface Training Script')
parser.add_argument('config', type=str, help='config file path')

if __name__ == '__main__':
    args = parser.parse_args()
    opt = OmegaConf.load(args.config)

    # Load Dataloader
    dataloader = get_image_dataset(opt.DATA)
    
    # Model
    # model = InceptionNet(num_classes = opt.MODEL.num_classes)
    model = XceptionNet(opt.MODEL)
    summary(model, (3, opt.DATA.image_size, opt.DATA.image_size), device='cpu')

    # Logger
    # log train/val loss, acc, etc.
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG) 
    console_logging_format = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_logging_format)
    logger.addHandler(console_handler)

    # log를 파일에 출력
    file_handler = logging.FileHandler(opt.LOG_FILENAME)
    file_handler.setFormatter(console_logging_format)
    logger.addHandler(file_handler)

    # BANMo System    
    trainer = Trainer(opt, dataloader, model, logger)

    # train
    trainer.train()
    
    # test
    tester = Tester(opt, dataloader, model, logger)
    tester.test("test")
    tester.test("train")
    tester.test("val")
    