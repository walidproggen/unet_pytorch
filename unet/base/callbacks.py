# General imports
import os
from os.path import join
from abc import ABC, abstractmethod

# Pytorch imports
import torch


class BaseCallback(ABC):
    def __init__(self, path, model):
        self.path = path
        self.model = model

    @abstractmethod
    def run(self, epoch, loss):
        pass


class CheckpointCallback(BaseCallback):
    def __init__(self, optimizer, **kwargs):
        super().__init__(kwargs['path'], kwargs['model'])
        self.output_path = join(self.path, 'checkpoints')
        os.makedirs(self.output_path, exist_ok=True)

        self.optimizer = optimizer

    def run(self, epoch, loss):
        torch.save({
            'epoch': epoch,
            'loss': loss,
            'optim_state': self.optimizer.state_dict(),
            'model_state': self.model.state_dict()
        }, join(self.output_path, f'model_chkpt_epoch_{epoch}.pt'))


class SaveBestCallback(BaseCallback):
    def __init__(self, minimize=True, **kwargs):
        super().__init__(kwargs['path'], kwargs['model'])
        self.minimize = minimize

        # Create output-dir
        self.output_path = join(self.path, 'weights')
        os.makedirs(self.output_path, exist_ok=True)

        # Init loss_value
        self.loss_value = 0.0
        if self.minimize:
            self.loss_value = float('inf')

    def run(self, epoch, loss):
        if self.minimize:
            if loss > self.loss_value:
                return
        else:
            if loss < self.loss_value:
                return

        print(f'Saving best-weight. Improved from {self.loss_value:.3f} to {loss:.3f}.')
        torch.save(self.model.state_dict(), join(self.output_path, f'best_weight.pth'))
        self.loss_value = loss

