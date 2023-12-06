import time
import os
import torch
import torch.nn as nn
import os.path as osp
import pdb
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from experiment.engine.loss import get_loss_function
from experiment.engine.parallel import DataParallelCriterion
from datetime import datetime, timedelta
import math
from omegaconf import OmegaConf

class Trainer():
    def __init__(self, opt, data_loader, model, logger):
        self.opt = opt
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_loader = data_loader
        self.train_loader = data_loader['train']
        self.val_loader = data_loader['val']
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt.TRAIN.lr)

        self.loss_name = opt.TRAIN.loss
        self.loss_function, self.contrastive_learning = get_loss_function(opt.TRAIN.loss)
        self.loss_function = self.loss_function.to(self.device)
        
        self.logger = logger

        self.total_epoch = opt.TRAIN.epochs
        self.log_interval = opt.TRAIN.log_interval
        self.save_interval = opt.TRAIN.save_interval
        self.ckpt_dir = opt.TRAIN.ckpt_dir
        self.load_ckpt_dir = opt.TRAIN.load_ckpt_dir

        self.sampler = data_loader['sampler'] if opt.distributed else None
        
        
    def train(self):
        os.makedirs(osp.join(self.ckpt_dir, 'weights'), exist_ok=True)
        self.logger.info(f'CONFIG: \n{self.opt}')
        OmegaConf.save(self.opt, os.path.join(self.ckpt_dir, "config.yaml"))

        if self.load_ckpt_dir != 'None':
            self.load_model()
            print('load model from ', self.load_ckpt_dir)
        else:
            print('no ckpt to load!')
        print('start training')

        total_steps = 0
        train_loss_list = []
        val_loss_list = []
        best_val = math.inf
        best_acc = 0
        best_info = ''

        for epoch in range(self.total_epoch):
            self.model.train()
            if self.sampler is not None:
                self.sampler.set_epoch(epoch)
                
            steps = 0
            train_loss = 0
            for data in self.train_loader: # len(self.train_loader) = train data size / batch size  # len(self.train_loader.dataset) = train data size
                steps += 1
                total_steps += 1
                # run step
                train_loss += self.run_step(data) 
                if self.logger is not None:
                    if steps%self.log_interval == 0:
                        print(f"loss: {train_loss.item()/steps:>7f}  [{steps:>5d}/{len(self.train_loader):>5d}]")
    
                if total_steps%self.save_interval == 0:
                    self.save_model(steps, epoch)
                    
            train_loss_list.append(train_loss/len(self.train_loader))
            total, correct, val_loss, auc_score = self.validate()
            val_loss_list.append(val_loss)

            if self.contrastive_learning:
                if val_loss < best_val:
                    torch.save(self.model.state_dict(), osp.join(self.ckpt_dir, 'best.pt'))
                    best_info = f'epoch{epoch+1}_step{steps}'
                    best_val = val_loss
            else:      
                val_acc = correct / total * 100      
                if val_acc > best_acc:
                    torch.save(self.model.state_dict(), osp.join(self.ckpt_dir, 'best.pt'))
                    best_info = f'epoch{epoch+1}_step{steps}'
                    best_acc = val_acc

            self.save_loss_graph(epoch+1, train_loss_list, val_loss_list)

            if self.contrastive_learning:
                self.logger.info('Epoch: %d/%d, Train loss: %.6f, val loss: %.6f' 
                                 %(epoch+1, self.total_epoch, train_loss/len(self.train_loader), val_loss))
            else:
                self.logger.info('Epoch: %d/%d, Train loss: %.6f, val loss: %.6f, Accuracy: %.2f AUC score: %.2f' 
                                %(epoch+1, self.total_epoch, train_loss/len(self.train_loader), val_loss, 100*correct/total, auc_score))
                
        self.save_model(steps, epoch)
        self.logger.info('Finished Training : total steps %d' %total_steps)
        self.logger.info(f'Best checkpoint at:{best_info}\n')

    def validate(self, post_function=nn.Softmax(dim=1)):
        total, correct, val_loss = 0, 0, 0
        self.model.eval()
        with torch.no_grad():
            pred = []
            true = []
            for data in self.val_loader:
                output = self.get_output(data)

                if self.contrastive_learning:
                    val_loss += self.loss_function(output, data['label'].to(self.device)).item()
                else:
                    output = post_function(output)
                    _, predicted = torch.max(output.data, 1)

                    pred += output.data[:, 1].cpu().tolist()
                    true += data['label'].cpu()

                    total += data['label'].to(self.device).size(0)
                    correct += (predicted == data['label'].to(self.device)).sum().item()
                    loss = self.loss_function(output, data['label'].to(self.device)).item()
                    val_loss += loss

            if self.contrastive_learning:
                # Return dummy values for unused metrics
                return 0, 0, val_loss/len(self.val_loader), 0
            
            auc_score = roc_auc_score(true, pred) * 100
        return total, correct, val_loss/len(self.val_loader), auc_score

    def run_step(self, data):
        # forward / loss / backward / update
        if self.optimizer is not None:
            self.optimizer.zero_grad()

        output = self.get_output(data)
        train_loss = self.loss_function(output, data['label'].to(self.device))
        train_loss.backward()
        if self.optimizer is not None:
            self.optimizer.step()
        return train_loss

    def get_output(self, data):
        if self.loss_name == 'SupCon':
            x1, x2 = data['frame']
            data['frame'] = torch.cat([x1, x2], dim=0)
            output = self.model(data['frame'].to(self.device))
            bsz = data['frame'].shape[0] // 2
            f1, f2 = torch.split(output, [bsz, bsz], dim=0)
            output = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        elif self.loss_name == 'triplet':
            anchor, pos, neg = data['frames']
            anchor_features = self.model(anchor.to(self.device))
            pos_features = self.model(pos.to(self.device))
            neg_features = self.model(neg.to(self.device))
            output = [anchor_features, pos_features, neg_features]  
        elif self.loss_name == 'SupConHnm':
            anchors, positives, negatives = data['frames']
            a1_features = self.model(anchors[0].to(self.device))
            a2_features = self.model(anchors[1].to(self.device))

            pos_features = self.model(positives[0].to(self.device))
            neg_features = self.model(negatives[0].to(self.device))
            output = [[a1_features, a2_features], pos_features, neg_features] 
        else:
            output = self.model(data['frame'].to(self.device))

        return output


    def load_model(self):
        self.model.load_state_dict(torch.load(self.load_ckpt_dir))
            
    def save_model(self, steps, epoch):
        torch.save(self.model.state_dict(), osp.join(self.ckpt_dir, 'weights', f'ep{epoch+1:03}_step{steps:04}.pt'))

    def save_loss_graph(self, epoch, train_loss, val_loss):   
        epochs = [i+1 for i in range(epoch)]    
        train_loss_list = torch.tensor(train_loss, device = 'cpu')
        val_loss_list = torch.tensor(val_loss, device = 'cpu')
    
        plt.plot(epochs, train_loss_list, label=f'Train Loss')
        plt.plot(epochs, val_loss_list, label=f'Val Loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(osp.join(self.ckpt_dir, 'loss.png'))
        plt.close()