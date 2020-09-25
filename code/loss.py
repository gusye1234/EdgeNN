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
        # print(groundTruth_mask[:20])
        log_likelihood = torch.log(probability_node)
        loss = F.nll_loss(log_likelihood, groundTruth_mask.long())
        #
        intrust = probability['poss_edge'][:, -1].detach()
        # edges = torch.LongTensor([(pair[0], pair[1]) for pair, _ in self.G.edges()]).to(world.DEVICE)
        edges, weights = self.G.edges_tensor()
        #
        if self.semi_lambda >= 1e-9:
            semi_loss = torch.sum((1 - intrust) * torch.sum(
                (probability['poss_edge'][edges[:, 0]] -
                probability['poss_edge'][edges[:, 1]]).pow(2)))
            semi_loss *= self.semi_lambda
        else:
            semi_loss = 0
        # sorry about the coming coda, but vectorization is much faster
        poss_edge = probability['poss_edge']
        edge_loss = 0.
        label_mask = mask[edges[:, 0]] | mask[edges[:, 1]]
        both_label = mask[edges[:, 0]] & mask[edges[:, 1]]

        same_label = (groundTruth[edges[both_label][:, 0]] == groundTruth[edges[both_label][:, 1]])
        diff_label = ~same_label

        single_label = label_mask - both_label
        left_single = mask[edges[single_label][:, 0]]
        right_single = mask[edges[single_label][:, 1]]
        range_index = torch.arange(same_label.sum())
        edge_loss += -torch.sum(torch.log(
            poss_edge[both_label][same_label][range_index, groundTruth[edges[both_label][same_label][:,0]]]
        ))
        edge_loss += -torch.sum(torch.log(
            poss_edge[both_label][diff_label][:, -1]
        ))
        range_index_left = torch.arange(left_single.sum())
        range_index_right = torch.arange(right_single.sum())
        edge_loss += -(torch.sum(torch.log(
            poss_edge[single_label][left_single][:, -1] + \
                poss_edge[single_label][left_single][range_index_left, groundTruth[edges[single_label][left_single][:,0]]]
        )) + torch.sum(torch.log(
            poss_edge[single_label][right_single][:, -1] + \
                poss_edge[single_label][right_single][range_index_right, groundTruth[edges[single_label][right_single][:,1]]]
        )))
        edge_loss *= self.edge_lambda
        edge_loss /= torch.sum(label_mask)
        # The above code is equal to below:
        '''
        for i, edge in enumerate(edges):
            if mask[edge[0]] or mask[edge[1]]:
                if mask[edge[0]] and mask[edge[1]]:
                    if groundTruth[edge[0]] == groundTruth[edge[1]]:
                        edge_loss += -torch.log(poss_edge[i][groundTruth[edge[0]]])
                    else:
                        edge_loss += -torch.log(poss_edge[i][-1])
                else:
                    labeled = edge[0] if mask[edge[0]] else edge[1]
                    edge_loss += -torch.log(poss_edge[i][-1] +
                                            poss_edge[i][groundTruth[labeled]])
        '''
        edge_loss *= self.edge_lambda
        edge_loss /= torch.sum(label_mask)
        return loss + semi_loss + edge_loss
