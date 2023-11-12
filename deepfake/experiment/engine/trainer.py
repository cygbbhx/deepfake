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

        print(f'Using {opt.TRAIN.loss} for training...')
        
        
    def train(self):
        if self.load_ckpt_dir != 'None':
            self.load_model()
            print('load model from ', self.load_ckpt_dir)
            return
        else:
            print('no ckpt to load!')
        print('start training')
        total_steps = 0
        train_loss_list = []
        val_loss_list = []

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
                    self.save_model(total_steps, epoch)
                    
            train_loss_list.append(train_loss/len(self.train_loader))
            total, correct, val_loss, auc_score = self.validate()
            val_loss_list.append(val_loss)
            self.save_loss_graph(epoch+1, train_loss_list, val_loss_list)

            if (self.contrastive_learning):
                self.logger.info('Epoch: %d/%d, Train loss: %.6f, val loss: %.6f' 
                                 %(epoch+1, self.total_epoch, train_loss/len(self.train_loader), val_loss))
            else:
                self.logger.info('Epoch: %d/%d, Train loss: %.6f, val loss: %.6f, Accuracy: %.2f AUC score: %.2f' 
                                %(epoch+1, self.total_epoch, train_loss/len(self.train_loader), val_loss, 100*correct/total, auc_score))
                
        self.save_model(total_steps, epoch)
        self.logger.info('Finished Training : total steps %d' %total_steps)

    def validate(self, post_function=nn.Softmax(dim=1)):
        total, correct, val_loss = 0, 0, 0
        self.model.eval()
        with torch.no_grad():
            pred = []
            true = []

            videoNum = len(self.val_loader.dataset.videos)
            clipsNum = len(self.val_loader.dataset.clips)

            total = videoNum
            
            tmp_pred, tmp_true = [], []
            count = 0
            cur_video = 0

            for data in self.val_loader:
                if (self.loss_name == 'SupCon'):
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
                else:
                    output = self.model(data['frame'].to(self.device))
                    

                if (self.contrastive_learning):
                    val_loss += self.loss_function(output, data['label'].to(self.device)).item()
                else:
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
                            val_loss += self.loss_function(ensembled_outputs.to(self.device), tmp_true.to(self.device)).item()

                            tmp_pred, tmp_true = [], []
                        tmp_pred += [torch.tensor(pred_results[i])]
                        tmp_true = data['label'].cpu()[i]

                    count += 1

            # Evaluate last video 
            if len(tmp_pred) != 0:
                ensembled_outputs = torch.stack(tmp_pred).mean(dim=0)
                _, prediction = torch.max(ensembled_outputs, 0)

                pred.append(ensembled_outputs[1])
                correct += int(prediction == tmp_true)
                true.append(tmp_true)
                val_loss += self.loss_function(ensembled_outputs.to(self.device), tmp_true.to(self.device)).item()

            if self.contrastive_learning:
                # Return dummy values for unused metrics
                return 0, 0, val_loss/len(self.val_loader), 0
            
            auc_score = roc_auc_score(true, pred) * 100
        return total, correct, val_loss/len(self.val_loader), auc_score

    def run_step(self, data):
        # forward / loss / backward / update
        if self.optimizer is not None:
            self.optimizer.zero_grad()

        if self.loss_name == 'SupCon':
            x1, x2 = data['frame']
            data['frame'] = torch.cat([x1, x2], dim=0)       
            output = self.model(data['frame'].to(self.device))        
            bsz = data['frame'].shape[0] // 2
            f1, f2 = torch.split(output, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            output = features
        elif self.loss_name == 'triplet':
            anchor, pos, neg = data['frames']
            anchor_features = self.model(anchor.to(self.device))
            pos_features = self.model(pos.to(self.device))
            neg_features = self.model(neg.to(self.device))
            output = [anchor_features, pos_features, neg_features]       
        else:
            output = self.model(data['frame'].to(self.device))

        train_loss = self.loss_function(output, data['label'].to(self.device))
        train_loss.backward()
        if self.optimizer is not None:
            self.optimizer.step()
        return train_loss

    def load_model(self):
        self.model.load_state_dict(torch.load(self.load_ckpt_dir))
            
    def save_model(self, steps, epoch):
        os.makedirs(self.ckpt_dir, exist_ok=True)
        torch.save(self.model.state_dict(), osp.join(self.ckpt_dir, f'step{steps}_ep{epoch+1}.pt'))

    def save_loss_graph(self, epoch, train_loss, val_loss):   
        os.makedirs(self.ckpt_dir, exist_ok=True)
        epochs = [i+1 for i in range(epoch)]    
        train_loss_list = torch.tensor(train_loss, device = 'cpu')
        val_loss_list = torch.tensor(val_loss, device = 'cpu')
    
        plt.plot(epochs, train_loss_list, label=f'Train Loss')
        plt.plot(epochs, val_loss_list, label=f'Val Loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(osp.join(self.ckpt_dir, 'loss.png'))
        plt.close()