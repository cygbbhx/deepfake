import os #, sys
import logging
from omegaconf import OmegaConf
from argparse import ArgumentParser

import torch.nn as nn
import torch
from torchvision import models
import torch.backends.cudnn as cudnn
from experiment.engine.tester import Tester
from experiment.engine.trainer import Trainer
from experiment.dataloader.ImageDataset import get_image_dataset
from experiment.dataloader.VideoDataLoader import get_video_dataset
from experiment.model.xception import XceptionNet
from experiment.model.i3d import I3D
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
import numpy as np
import random
from datetime import datetime
from pytz import timezone
import pytz


def main(rank, opt):
    if opt.distributed:
        init_distributed_training(rank, opt)
        local_gpu_id = opt.gpu

    fix_seed()

    # Load Dataloader
    print(f"Using {opt.DATA.type} dataset ...")
    dataloader = get_video_dataset(opt.DATA) if opt.DATA.type == 'video' else  get_image_dataset(opt.DATA) 

    if opt.distributed:
        train_sampler = DistributedSampler(dataset=dataloader['train'].dataset, shuffle=True)
        val_sampler = DistributedSampler(dataset=dataloader['val'].dataset, shuffle=True)
        
        batch_sampler_train = torch.utils.data.BatchSampler(train_sampler, opt.DATA.batch_size, drop_last=True)
        train_loader = DataLoader(dataloader['train'].dataset, batch_sampler=batch_sampler_train, 
                                  pin_memory=True, num_workers = opt.num_workers)
        val_loader = DataLoader(dataloader['val'].dataset, opt.DATA.batch_size, num_workers=opt.num_workers,
                                pin_memory=True, sampler=val_sampler, drop_last=False)

        dataloader = {'train': train_loader, 'val': val_loader, 'test': dataloader['test'], 'sampler':train_sampler}
        
    # Model
    model_list = {
        'Xception': XceptionNet,
        'I3D': I3D
    }

    model = model_list[opt.MODEL.name](opt.MODEL)

    if (opt.distributed):
        model = model.cuda(local_gpu_id)
        model = DistributedDataParallel(module=model, device_ids=[local_gpu_id])

    # Logger
    # log train/val loss, acc, etc.
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG) 
    console_logging_format = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    console_logging_format.converter = lambda *args: datetime.now(tz=timezone('Asia/Seoul')).timetuple()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_logging_format)
    logger.addHandler(console_handler)

    # log를 파일에 출력
    file_handler = logging.FileHandler(os.path.join(opt.TRAIN.ckpt_dir, 'train_log'))
    file_handler.setFormatter(console_logging_format)
    logger.addHandler(file_handler)

    # BANMo System    
    trainer = Trainer(opt, dataloader, model, logger)

    # train
    trainer.train()
    
    # test
    # tester = Tester(opt, dataloader, model, logger)
    # tester.test("test")
    # tester.test("train")
    # tester.test("val")

def fix_seed():
    # fix random seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(42)

def get_time():
    korea_timezone = pytz.timezone('Asia/Seoul')
    current_time = datetime.now(korea_timezone)

    date_component = current_time.strftime("%m%d")
    time_component = current_time.strftime("%H%M")
    seconds_component = current_time.strftime("%S")

    result_string = f"{date_component}-{time_component}-{seconds_component}"

    return result_string

def init_distributed_training(rank, opt):
    opt.rank = rank
    opt.gpu = opt.rank % torch.cuda.device_count()
    local_gpu_id = int(opt.gpu_ids[opt.rank])
    torch.cuda.set_device(local_gpu_id)

    if opt.rank is not None:
        print("Use GPU: {} for training".format(local_gpu_id))
    
    torch.distributed.init_process_group(backend='nccl',
                                         init_method='tcp://127.0.0.1:' + str(opt.port),
                                         world_size=opt.ngpus_per_node,
                                         rank=opt.rank)

if __name__ == '__main__':
    
    parser = ArgumentParser('Deepface Training Script')
    parser.add_argument('config', type=str, help='config file path')
    parser.add_argument('--distributed', action='store_true')
    args = parser.parse_args()
    opt = OmegaConf.load(args.config)
    
    opt.TRAIN.loss = opt.TRAIN.get('loss', 'crossentropy')
    opt.TRAIN.ckpt_dir = os.path.join(opt.TRAIN.ckpt_dir, get_time())
    opt.DATA.augSelf = opt.DATA.get('augSelf', False)
    opt.DATA.type = opt.DATA.get('type', 'image')
    opt.distributed = args.distributed

    os.makedirs(opt.TRAIN.ckpt_dir, exist_ok=True)

    if opt.distributed:
        opt.ngpus_per_node = torch.cuda.device_count()
        opt.gpu_ids = list(range(opt.ngpus_per_node))
        opt.num_workers = opt.ngpus_per_node * 4
        opt.port = 55555

        torch.multiprocessing.spawn(main,
                                    args=(opt,),
                                    nprocs=opt.ngpus_per_node,
                                    join=True)
    else:
        main(rank=0, opt=opt)