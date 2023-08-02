import torch
import os

class Trainer():
    def __init__(self, opt, data_loader, model, logger):
        self.opt = opt
        self.data_loader = data_loader
        self.train_loader = data_loader['train']
        self.val_loader = data_loader['val']
        self.test_loader = data_loader['test']
        self.model = model
        self.optimizer = None
        self.scheduler = None
        self.logger = logger
        self.criterion = torch.nn.CrossEntropyLoss(reduction='sum')

        self.total_epoch = opt.TRAIN.EPOCH
        self.log_interval = opt.TRAIN.LOG_INTERVAL
        self.save_interval = opt.TRAIN.SAVE_INTERVAL
        self.lr = opt.TRAIN.lr
        self.momentum = opt.TRAIN.momentum
        self.weight_decay = opt.TRAIN.weight_decay

        self.lr_step_size = opt.TRAIN.lr_step_size
        self.lr_gamma = opt.TRAIN.lr_gamma
        self.steps = 0

        self.checkpoint_dir = opt.CHK_DIR

        assert torch.cuda.is_available() == True, \
        "GPU device not detected"
        
        self.device = 'cuda'


    def train(self):
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=self.lr_step_size, gamma=self.lr_gamma)
        
        self.model.to(self.device)

        for epoch in range(self.total_epoch):
            self.model.train()
            total_loss = 0

            for data in self.train_loader:
                self.optimizer.zero_grad()

                # Move data to the device
                data = {key: value.to(self.device) for key, value in data.items()}

                # run step
                loss = self.run_step(data)
                total_loss += loss.item()
                
                self.steps += 1

                if self.logger is not None:
                    if self.steps%self.log_interval == 0:
                        self.logger.debug(f'Epoch {epoch:03} | Step {self.steps:09} | Loss {loss:09}')
    
                if self.steps%self.save_interval == 0:
                    self.save_model()
                    
                    
            self.scheduler.step()
            average_loss = total_loss / len(self.train_loader)

            self.model.eval()
            correct = 0
            total = 0

            with torch.no_grad():
                for data in self.train_loader:
                    frames = data['frame'].to(self.device)
                    labels = data['label'].to(self.device)

                    # Forward pass
                    outputs = self.model(frames)

                    # Get the predicted labels
                    _, predicted = torch.max(outputs.data, 1)

                    # Count the total number of labels
                    total += labels.size(0)

                    # Count the number of correct predictions
                    correct += (predicted == labels).sum().item()

            # Calculate the accuracy
            accuracy = (correct / total) * 100
            self.logger.debug(f'----- Epoch {epoch:03} | Training Accuracy: {accuracy:.2f}% | Average Loss: {average_loss:09}')
            self.validate()
            self.test()

    def validate(self):
        self.model.eval()

        correct = 0
        total = 0
        total_loss = 0

        with torch.no_grad():
            for data in self.val_loader:
                frames = data['frame'].to(self.device)
                labels = data['label'].to(self.device)

                # Forward pass
                outputs = self.model(frames)

                # Get the predicted labels
                _, predicted = torch.max(outputs.data, 1)

                # Count the total number of labels
                total += labels.size(0)

                # Count the number of correct predictions
                correct += (predicted == labels).sum().item()

                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

        average_loss = total_loss / len(self.val_loader)
        # Calculate the accuracy
        accuracy = (correct / total) * 100

        self.logger.debug(f'----- Validation Accuracy: {accuracy:.2f}% | Validation loss: {average_loss:09}')


    def test(self):
        self.model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for data in self.test_loader:
                frames = data['frame'].to(self.device)
                labels = data['label'].to(self.device)

                # Forward pass
                outputs = self.model(frames)

                # Get the predicted labels
                _, predicted = torch.max(outputs.data, 1)

                # Count the total number of labels
                total += labels.size(0)

                # Count the number of correct predictions
                correct += (predicted == labels).sum().item()

        # Calculate the accuracy
        accuracy = (correct / total) * 100

        self.logger.debug(f'----- Test Accuracy: {accuracy:.2f}%')


    def run_step(self, data):
        output = self.model(data['frame'])
        loss = self.criterion(output, data['label'])
        loss.backward()
        self.optimizer.step()
        return loss

    def load_model(self, path):
        checkpoint = torch.load(self.checkpoint_dir + path) 
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        
    
    def save_model(self):
        torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, f'{self.opt.EXP_NAME}_iter_{self.steps}.pt'))

