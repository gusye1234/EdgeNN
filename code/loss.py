import torch 
from torch.nn import Module


class BasicLoss(Module):
    def __init__(self, *args, **kwargs):
        super(BasicLoss, self).__init__()
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError
        