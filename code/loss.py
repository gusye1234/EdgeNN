import world
import torch
from torch.nn import Module
from data import Graph
import torch.nn.functional as F

class BasicLoss(Module):
    def __init__(self, *args, **kwargs):
        super(BasicLoss, self).__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError

class CrossEntropy(BasicLoss):
    def __init__(self, **kwargs):
        super(CrossEntropy, self).__init__(**kwargs)
        pass

    def forward(self, probability, groundTruth, mask):
        '''
            probability already sum=1
        '''
        probability = probability['poss_node'][mask]
        groundTruth = groundTruth[mask]
        log_likelihood = torch.log(probability)
        loss = F.nll_loss(log_likelihood, groundTruth.long())
        if torch.isnan(loss):
            pass
        return loss


class EdgeLoss(BasicLoss):
    def __init__(self, **kwargs):
        super(EdgeLoss, self).__init__(**kwargs)
        self.G : Graph = kwargs['graph']
        self.semi_lambda = kwargs['semi_lambda']
        self.edge_lambda = kwargs['edge_lambda']

    def forward(self, probability, groundTruth, mask):
        probability_node = probability['poss_node'][mask]
        groundTruth_mask = groundTruth[mask]
        log_likelihood = torch.log(probability_node)
        loss = F.nll_loss(log_likelihood, groundTruth_mask.long())
        #
        intrust = probability['poss_edge'][:, -1].detach()
        # edges = torch.LongTensor([(pair[0], pair[1]) for pair, _ in self.G.edges()]).to(world.DEVICE)
        edges, weights = self.G.edges_tensor()
        #
        semi_loss = torch.sum((1 - intrust) * torch.sum(
            (probability['poss_edge'][edges[:, 0]] -
             probability['poss_edge'][edges[:, 1]]).pow(2)))
        semi_loss *= self.semi_lambda
        #
        poss_edge = probability['poss_edge']
        edge_loss = 0.
        # TODO
        label_mask = mask[edges[:, 0]] | mask[edges[:, 1]]
        both_label = mask[edges[:, 0]] & mask[edges[:, 1]]
        single_label = label_mask - both_label 
        # for i, edge in enumerate(edges):
        #     if mask[edge[0]] or mask[edge[1]]:
        #         if mask[edge[0]] and mask[edge[1]]:
        #             if groundTruth[edge[0]] == groundTruth[edge[1]]:
        #                 edge_loss += -torch.log(poss_edge[i][groundTruth[edge[0]]])
        #             else:
        #                 edge_loss += -torch.log(poss_edge[i][-1])
        #         else:
        #             labeled = edge[0] if mask[edge[0]] else edge[1]
        #             edge_loss += -torch.log(poss_edge[i][-1] +
        #                                     poss_edge[i][groundTruth[labeled]])
        edge_loss *= self.edge_lambda
        # print(loss, semi_loss, edge_loss)
        return loss + semi_loss + edge_loss
