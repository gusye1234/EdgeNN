import torch
from torch.nn import Module
import torch.nn.functional as F

class BasicLoss(Module):
    def __init__(self, *args, **kwargs):
        super(BasicLoss, self).__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError

class CrossEntropy(BasicLoss):
    def __init__(self):
        super(CrossEntropy, self).__init__()
        pass

    def forward(self, probability, groundTruth):
        '''
            probability already sum=1
        '''
        log_likelihood = torch.log(probability)
        return F.nll_loss(log_likelihood, groundTruth.long())