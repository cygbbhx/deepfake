import os #, sys
from omegaconf import OmegaConf
from argparse import ArgumentParser

from experiment.engine.trainer import Trainer
from experiment.dataloader.ImageDataset import get_test_dataset
from experiment.model.model import XceptionNet
#from torch.utils.tensorboard import SummaryWriter
import logging
import torch

parser = ArgumentParser('Deepface Training Script')
parser.add_argument('config', type=str, help='config file path')

if __name__ == '__main__':
    args = parser.parse_args()
    opt = OmegaConf.load(args.config)
    #writer = SummaryWriter()

    # Load Dataloader
    dataloader = get_test_dataset(opt.DATA)
    
    device = 'cuda:0'

    # Model
    model = XceptionNet(opt.MODEL)
    model.to(device)
    checkpoint = torch.load(opt.WEIGHTS) 
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

    model.eval()

    correct = 0
    total = 0


    with torch.no_grad():
        for data in dataloader:
            frames = data['frame'].to(device)
            labels = data['label'].to(device)

            # Forward pass
            outputs = model(frames)

            # Get the predicted labels
            _, predicted = torch.max(outputs.data, 1)

            # Count the total number of labels
            total += labels.size(0)

            # Count the number of correct predictions
            correct += (predicted == labels).sum().item()

    # Calculate the accuracy
    accuracy = (correct / total) * 100

    logger.debug(f'----- Test Accuracy for {opt.DATA.name} dataset: {accuracy:.2f}%')
