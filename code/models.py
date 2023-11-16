import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch import autograd
import torch.nn.functional as F
import time


#mlp baseline 
class MLP(nn.Module):
    def __init__(self, num_genes, hidden1=2048, hidden2=512):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_genes, hidden1),
            nn.Softplus(),
            nn.Linear(hidden1, hidden2),
            nn.Softplus(),
            nn.Linear(hidden2, 2)  # Two output neurons for binary classification
        )

    def forward(self, x):
        '''Forward pass'''
        return self.layers(x)  # Output raw logits for CrossEntropyLoss

    def forward_plus(self, x):
        '''Forward pass with intermediate layer access'''
        x1 = self.layers[0](x)
        y = self.layers[1:](x1)
        return y, x1

    def layer1_weight(self):
        return self.layers[0].weight

