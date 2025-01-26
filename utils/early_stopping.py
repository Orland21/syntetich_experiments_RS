'''Early stoping'''
import torch
import torch.nn as nn
import numpy as np


class EarlyStopping:
    'early stopping'
    def __init__(self,patience=10,delta=0,save_path=None):
        
        self.patience = patience
        self.delta = delta
        self.save_path = save_path 
        self.counter = 0
        self.best_loss=np.inf

    def __call__(self,val_loss,model):

        if val_loss <self.best_loss-self.delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.save_path is not None:
                torch.save(model.state_dict(),self.save_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


