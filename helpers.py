import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import random
import math
from IPython.display import clear_output

class BaseTransform():
    "Numericalization for any set of items"
    
    def __init__(self, vocab):
        "vocab: the set of items to be numericalized"
        self.vocab = vocab
        self.count = len(self.vocab)
        self.item2num = {item:num for num,item in enumerate(self.vocab)}
        self.num2item = {num:item for num,item in enumerate(self.vocab)}
    
    def encode(self,o):
        "Convert a list of items to a tensor of numbers"
        return torch.tensor([self.item2num[item] for item in o])
    
    def decode(self,o):
        "Convert a tensor of numbers to a list of items"
        return [self.num2item[num] for num in o.tolist()]

class TokTransform(BaseTransform):
    "Numericalization for a vocabulary of tokens"
    
    def __init__(self, vocab):
        vocab = ['xxunk'] + vocab
        super().__init__(vocab)
    
    def encode(self,o):
        "Convert a list of tokens to a tensor of numbers"
        return torch.tensor([self.item2num[item] if item in self.vocab else 0 for item in o])

class DataLoader():
    "Groups x and y datasets with method to get batches"
    def __init__(self, x_set, y_set, bs):
        """
        x_set: list of x tensors
        y_set: list of y tensors
        bs: batch size
        """
        self.x_set, self.y_set, self.bs = x_set, y_set, bs
        assert(len(self.x_set) == len(self.y_set))
        self.n_items = len(self.x_set)
        self.n_batches = math.ceil(self.n_items / bs)
    
    def get_batches(self):
        "Generator that yields pairs of x and y batches"
        self.x_set, self.y_set = self._shuffle_same(self.x_set, self.y_set)
        for start in range(0,self.n_items,self.bs):
            end = start + self.bs
            if end > self.n_items:
                end = self.n_items
            xb, yb = self.x_set[start:end], self.y_set[start:end]
            xb, yb = torch.stack(xb), torch.stack(yb)
            assert xb.shape == yb.shape
            yield xb, yb
    
    def _shuffle_same(self, x_set, y_set):
        "Shuffle both x_set and y_set, but keep them lined up with each other"
        zipped = list(zip(x_set, y_set))
        random.shuffle(zipped)
        return list(zip(*zipped))

class DataLoaders():
    "Groups training and validation DataLoader objects"
    def __init__(self, train, val):
        """
        train: training DataLoader
        val: validation DataLoader
        """
        self.train, self.val = train, val

class Learner():
    "Groups all parts of deep learning process into one object"
    def __init__(self, dls, model, loss_func, opt, metric):
        """
        dls: a DataLoaders object
        model: the neural network object (PyTorch module)
        loss_func: the loss function
        opt: the optimizer
        metric: the evaluation metric
        """
        self.dls, self.model, self.loss_func, self.opt, self.metric = dls, model, loss_func, opt, metric
        self.n_batches = self.dls.train.n_batches
        self.train_loss_log = []
        self.val_loss_log = []
        self.metric_log = []
    
    def fit(self, n_epochs):
        "Train network for n_epochs number of epoch"
        for epoch in range(n_epochs):
            train_loss = self.train_epoch()
            self.train_loss_log.append(train_loss)
            val_loss, metric = self.validate()
            self.val_loss_log.append(val_loss)
            self.metric_log.append(metric)
    
    def train_epoch(self):
        "Train a single epoch"
        train_losses = []
        self.model.train()
        for i,(xb,yb) in enumerate(self.dls.train.get_batches()):
            self._print_progress(i)
            preds = self.model(xb)
            loss = self.loss_func(preds, yb)
            train_losses.append(loss.item())
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()
        mean_loss = torch.tensor(train_losses).mean().item()
        return mean_loss
    
    def validate(self):
        "Validate on full validation dataset"
        valid_losses = []
        metrics = []
        self.model.eval()
        with torch.no_grad():
            for xb,yb in self.dls.val.get_batches():
                preds = self.model(xb)
                loss = self.loss_func(preds, yb)
                valid_losses.append(loss)
                metrics.append(self.metric(preds, yb))
        mean_loss = torch.tensor(valid_losses).mean().item()
        mean_metric = torch.tensor(metrics).mean().item()
        return mean_loss, mean_metric

    def print_logs(self):
        "Print the history of all losses and metrics"
        print(pd.DataFrame({'Train loss': self.train_loss_log, 
                           'Val loss': self.val_loss_log, 
                           'Metric': self.metric_log}))
    
    def _print_progress(self, i):
        "Print the current progress within an epoch"
        clear_output(wait=True)
        print(f"Batch {i+1}/{self.n_batches}")
