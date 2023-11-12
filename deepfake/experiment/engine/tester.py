import time

import torch
import torch.nn as nn
import os.path as osp
import pdb
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import os 
import numpy as np
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm 

class Tester():
    def __init__(self, opt, data_loader, model, logger):
        self.opt = opt
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_loader = data_loader['train']
        self.val_loader = data_loader['val']
        self.test_loader = data_loader['test']
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt.TRAIN.lr)
        self.loss_function = nn.CrossEntropyLoss().to(self.device)
        self.logger = logger

        aug_list = [
            T.Resize((opt.DATA.image_size, opt.DATA.image_size)),
            T.ToTensor(),
            T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ]
        
        self.transforms = T.Compose(aug_list)
        
    def test(self, split="test", post_function=nn.Softmax(dim=1)):
        total, correct, loss = 0, 0, 0
        self.model.eval()
        if split == "train":
            dataloader = self.train_loader
        elif split == "test":
            dataloader = self.test_loader
        else:
            dataloader = self.val_loader
        with torch.no_grad():
            pred = []
            true = []

            for data in dataloader:
                output = self.model(data['frame'].to(self.device))
                output = post_function(output)
                _, predicted = torch.max(output.data, 1)

                pred += output.data[:, 1].cpu().tolist()
                true += data['label'].cpu()

                total += data['label'].to(self.device).size(0)
                correct += (predicted == data['label'].to(self.device)).sum().item()
                loss = self.loss_function(output, data['label'].to(self.device)).item()
                
            auc_score = roc_auc_score(true, pred) * 100
        self.logger.info('[ %s result ] loss: %.6f, Accuracy: %.2f, AUC: %.2f' %(split, loss, 100*correct/total, auc_score))

    def test_each(self, split="test", post_function=nn.Softmax(dim=1)):
        total, correct, loss = 0, 0, 0
        accuracy = []
        self.model.eval()

        if split == "train":
            dataloader = self.train_loader
        elif split == "test":
            dataloader = self.test_loader
        else:
            dataloader = self.val_loader
        with torch.no_grad():
            pred = []
            true = []

            videoNum = len(dataloader.dataset.videos)
            clipsNum = len(dataloader.dataset.clips)

            total = videoNum
            
            tmp_pred, tmp_true = [], []
            count = 0
            cur_video = 0
            
            for data in (dataloader):
                output = self.model(data['frame'].to(self.device))
                output = post_function(output)
                _, predicted = torch.max(output.data, 1)

                pred_results = output.data.cpu().tolist()

                for i, batch_inst in enumerate(data['video']):
                    if batch_inst != cur_video:
                        cur_video = batch_inst
                        ensembled_outputs = torch.stack(tmp_pred).mean(dim=0)
                        _, prediction = torch.max(ensembled_outputs, 0)

                        pred.append(ensembled_outputs[1])
                        correct += int(prediction == tmp_true)
                        true.append(tmp_true)

                        tmp_pred, tmp_true = [], []

                    tmp_pred += [torch.tensor(pred_results[i])]
                    tmp_true = data['label'].cpu()[i]

                count += 1

                # if (count % 500 == 0):
                #     print(f'Tested {count}/{clipsNum} clips...')
            
            # Evaluate last video 
            if len(tmp_pred) != 0:
                ensembled_outputs = torch.stack(tmp_pred).mean(dim=0)
                _, prediction = torch.max(ensembled_outputs, 0)

                pred.append(ensembled_outputs[1])
                correct += int(prediction == tmp_true)
                true.append(tmp_true)

            print(f'{len(true)}/{videoNum} videos evaluated.')

            auc_score = roc_auc_score(true, pred) * 100
        self.logger.info('[ %s result ] loss: %.6f, Accuracy: %.2f, AUC: %.2f' %(split, loss, 100*correct/total, auc_score))
