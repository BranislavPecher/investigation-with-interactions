import torch
import numpy as np
import torch.nn.functional as F
from src.modeling.models.helpers import compute_accuracy, tensors_to_device
import os


class BERT():
    def __init__(self, model, params={}, device=None):
        self.model = model.to(device=device)
        self.lr = params['lr']
        self.optimizer = params['optimizer'](self.model.parameters(), lr=self.lr)
        self.loss_function = params['loss_function'] if params.get('loss_function') else F.cross_entropy
        self.device = device


    def train(self, dataloader, batch_size, epochs):
        self.model.train()
        
        for epoch in range(epochs):
            print(f'Running epoch {epoch + 1}/{epochs}')

            batch_idx = 0
            for batch in dataloader.sample_data_in_batch(epoch, batch_size=batch_size):
                batch_idx += 1
                self.optimizer.zero_grad()
                batch = tensors_to_device(batch, device=self.device)
                
                inputs, targets = batch
                logits = self.model(inputs)
                loss = self.loss_function(logits, targets)

                loss.backward()
                self.optimizer.step()


    def evaluate_in_batch(self, dataloader, batch_size):
        self.model.eval()
        predicted = []
        labels = []

        for batch_idx, batch in enumerate(dataloader.batch_data_for_evaluation(batch_size)):
            batch = tensors_to_device(batch, device=self.device)
            inputs, targets = batch['test']

            with torch.set_grad_enabled(self.model.training):
                logits = self.model(inputs)
                _, predictions = torch.max(logits, dim=1)
            predicted.extend(predictions.cpu().detach().tolist())
            labels.extend(targets.cpu().detach().tolist())
        return (predicted, labels)

    
    def serialize(self, path):
        torch.save(self.model.state_dict(), os.path.join(path, 'model.pt'))
    
    def load_serialization(self, path):
        self.model.load_state_dict(torch.load(os.path.join(path, 'model.pt')))

    def serialize_or_load(self, path):
        if self.serialization_exists(path):
            self.model.load_state_dict(torch.load(os.path.join(path, 'model.pt')))
        else:
            torch.save(self.model.state_dict(), os.path.join(path, 'model.pt'))
    
    def serialization_exists(self, path):
        return os.path.exists(os.path.join(path, 'model.pt'))