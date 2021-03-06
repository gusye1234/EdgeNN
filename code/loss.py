import world
import torch
import numpy as np
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
        self.select = kwargs['select'] if kwargs.get('select') else ['task', 'semi', 'edge']

    def forward(self, probability, groundTruth, mask):
        loss = 0.

        if 'task' in self.select:
            probability_node = probability['poss_node'][mask]

            groundTruth_mask = groundTruth[mask]
            # print(groundTruth_mask[:20])
            log_likelihood = torch.log(probability_node)
            loss += F.nll_loss(log_likelihood, groundTruth_mask.long())
        #
        edges, weights = self.G.edges_tensor()

        # ==========================================================
        # peak = 1000
        # peak_candi = np.random.randint(len(edges), size=(peak, ))
        # peak_list = []
        # peak_prob = []
        # for i in peak_candi:
        #     edge = edges[i]
        #     if mask[edge[0]] and mask[edge[1]]:
        #         peak_list.append((groundTruth[edge[0]].item(), groundTruth[edge[1]].item()))
        #         if groundTruth[edge[0]] == groundTruth[edge[1]]:
        #             peak_prob.append(
        #                 probability['poss_edge'][i][groundTruth[edge[1]]].item())
        #         else:
        #             peak_prob.append(probability['poss_edge'][i][-1].item())
        #     else:
        #         if mask[edge[0]] or mask[edge[1]]:
        #             label = groundTruth[edge[0]] if mask[edge[0]] else groundTruth[edge[1]]
        #             peak_list.append((label.item(), '-1'))
        #             peak_prob.append(probability['poss_edge'][i][label].item() + probability['poss_edge'][i][-1].item())
        #         else:
        #             continue
        # count = 0
        # for i, j in zip(peak_list, peak_prob):
        #     if j < 0.5:
        #         count += 1
        #         # print(f"{j:.3f} : {i}")
        # print(count)
        # ==========================================================


        if 'semi' in self.select:
            intrust = probability['poss_edge'][:, -1].detach()
            if self.semi_lambda >= 1e-9:
                semi_loss = (1/2)*torch.sum( \
                    (torch.sum((probability['poss_node'][edges[:, 0]] - \
                    probability['poss_node'][edges[:, 1]]).pow(2), dim=1))
            )
                semi_loss *= self.semi_lambda
            else:
                semi_loss = 0
            # print(semi_loss, loss)
            loss += semi_loss
        if 'edge' in self.select:
            # sorry about the coming coda, but vectorization is much faster
            poss_edge = probability['poss_edge']
            edge_loss = 0.
            label_mask = mask[edges[:, 0]] | mask[edges[:, 1]]
            both_label = mask[edges[:, 0]] & mask[edges[:, 1]]

            same_label = (groundTruth[edges[both_label][:, 0]] == groundTruth[edges[both_label][:, 1]])
            diff_label = ~same_label

            single_label = label_mask ^ both_label
            left_single = mask[edges[single_label][:, 0]]
            right_single = mask[edges[single_label][:, 1]]
            range_index = torch.arange(same_label.sum())
            edge_loss += -torch.sum(torch.log(
                poss_edge[both_label][same_label][range_index, groundTruth[edges[both_label][same_label][:,0]]]
            ))
            # edge_loss += -torch.sum(torch.log(
            #     poss_edge[both_label][diff_label][:, -1]
            # ))
            range_index = torch.arange(diff_label.sum())
            edge_loss += -torch.sum(
                1/2*torch.log(poss_edge[both_label][diff_label][range_index, groundTruth[edges[both_label][diff_label][:,0]]]) +\
                    1/2*torch.log(poss_edge[both_label][diff_label][range_index, groundTruth[edges[both_label][diff_label][:,1]]])
            )
            
            # range_index_left = torch.arange(left_single.sum())
            # range_index_right = torch.arange(right_single.sum())
            # edge_loss += -(torch.sum(torch.log(
            #     poss_edge[single_label][left_single][:, -1] + \
            #         poss_edge[single_label][left_single][range_index_left, groundTruth[edges[single_label][left_single][:,0]]]
            # )) + torch.sum(torch.log(
            #     poss_edge[single_label][right_single][:, -1] + \
            #         poss_edge[single_label][right_single][range_index_right, groundTruth[edges[single_label][right_single][:,1]]]
            # )))
            # range_index_left = torch.arange(left_single.sum())
            # range_index_right = torch.arange(right_single.sum())
            # edge_loss += -(1/2*torch.sum(torch.log(
            #     poss_edge[single_label][left_single][range_index_left, groundTruth[edges[single_label][left_single][:,0]]]
            # )) + 1/2*torch.sum(torch.log(
            #     poss_edge[single_label][right_single][range_index_right, groundTruth[edges[single_label][right_single][:,1]]]
            # )))


            edge_loss *= self.edge_lambda
            edge_loss /= torch.sum(label_mask)
            loss += edge_loss
            # The above code is equal to below:
            '''
            edge_loss1 = 0.
            for i, edge in enumerate(edges):
                if mask[edge[0]] or mask[edge[1]]:
                    if mask[edge[0]] and mask[edge[1]]:
                        if groundTruth[edge[0]] == groundTruth[edge[1]]:
                            edge_loss1 += -torch.log(poss_edge[i][groundTruth[edge[0]]])
                        else:
                            edge_loss1 += -torch.log(poss_edge[i][-1])
                    else:
                        labeled = edge[0] if mask[edge[0]] else edge[1]
                        edge_loss1 += -torch.log(poss_edge[i][-1] +
                                                poss_edge[i][groundTruth[labeled]])
            edge_loss1 *= self.edge_lambda
            edge_loss1 /= torch.sum(label_mask)
            '''
            # print(loss.item(), edge_loss.item())
        return loss
