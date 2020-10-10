"""
Store all the trainable model
"""
import world
import math
import torch
from torch import nn
from data import Graph
from torch.nn import Module
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class BasicModel(Module):
    def __init__(self, *args, **kwargs):
        super(BasicModel, self).__init__()

    def predict_edges(self, src, dst):
        raise NotImplementedError

    def forward(self):
        'predict all the label'
        edges, weights = self.G.edges_tensor()
        poss_edge = self.predict_edges(edges[:, 0], edges[:, 1])
        poss_node = torch.zeros(self.num_nodes,
                             self.num_class + 1).to(world.DEVICE)

        value = poss_edge * weights.unsqueeze(1)
        index = edges[:, 0].repeat(self.num_class + 1, 1).t()
        poss_node.scatter_add_(0, index, value)
        poss_node /= self.G.neighbours_sum()
        return {'poss_node':poss_node, 'poss_edge': poss_edge}


class EmbeddingP(BasicModel):
    def __init__(self, CONFIG, G: Graph):
        super(EmbeddingP, self).__init__()
        self.G = G
        self.num_nodes = CONFIG['the number of nodes']
        self.num_dims = CONFIG['the number of embed dims']
        self.num_class = CONFIG['the number of classes']
        self.feature_dim = CONFIG['the dimension of features']
        self.init()

    # def operator(self, src, dst):
    #     return src + dst

    def operator(self, src, dst):
        return self.operator_naive(src, dst)

    def operator_naive(self, src, dst):
        return torch.cat([src, dst], dim=1)

    def operator_sy1(self, src, dst):
        E1 = (src + dst) / 2
        E2 = (src - dst).pow(2)
        return torch.cat([E1, E2], dim=1)

    def operator_sy2(self,src, dst):
        E1 = (src + dst) / 2
        E2 = (src - dst).abs()
        return torch.cat([E1, E2], dim=1)

    def init(self):
        # self.node_embedding = torch.nn.Embedding(
        #     self.num_nodes, self.num_dims
        # )
        self.embed = nn.Linear(self.feature_dim, self.num_dims)
        self.trans = nn.Sequential(nn.Linear(self.num_dims * 2, self.num_class + 1),
                                   nn.Softmax(dim=1))
        # nn.init.normal_(self.node_embedding.weight)

    def predict_edges(self, src, dst):
        embed = self.embed(self.G['features'])
        src_embed = embed[src]
        dst_embed = embed[dst]
        E = self.operator(src_embed, dst_embed)
        return self.trans(E)




class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        # return F.log_softmax(x, dim=1)
        return x


class GCN_single(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN_single, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nclass)
        # self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        # return F.log_softmax(x, dim=1)
        return x


class GCNP(BasicModel):
    def __init__(self, CONFIG, G):
        super(GCNP, self).__init__()
        self.G = G
        self.num_nodes = CONFIG['the number of nodes']
        self.num_dims = CONFIG['the number of embed dims']
        self.num_class = CONFIG['the number of classes']
        self.num_features = CONFIG['the dimension of features']
        hidden_dim = CONFIG['gcn_hidden']
        dropout = CONFIG['dropout_rate']
        self.gcn = GCN(self.num_features, hidden_dim, self.num_dims, dropout)
        self.trans = nn.Sequential(nn.Linear(self.num_dims*2, self.num_class + 1),
                                   nn.Softmax(dim=1))

    def operator(self, src, dst):
        E1 = (src + dst)/2
        E2 = (src - dst).pow(2)
        return torch.cat([E1, E2], dim=1)
        # return torch.cat([src, dst], dim=1)

    def predict_edges(self, src, dst):
        embedding = self.gcn(self.G['features'], self.G.adj)
        src_embed = embedding[src]
        dst_embed = embedding[dst]
        E = self.operator(src_embed, dst_embed)
        # print(E[:5])
        return self.trans(E)


class GCNP_aligned(Module):
    def __init__(self, CONFIG, G):
        super(GCNP_aligned, self).__init__()
        self.G = G
        self.num_nodes = CONFIG['the number of nodes']
        self.num_dims = CONFIG['the number of embed dims']
        self.num_class = CONFIG['the number of classes']
        self.num_features = CONFIG['the dimension of features']
        hidden_dim = CONFIG['gcn_hidden']
        dropout = CONFIG['dropout_rate']
        self.gcn = GCN(self.num_features, hidden_dim, self.num_dims, dropout)
        self.trans = nn.Sequential(nn.Linear(self.num_dims * 2, 16), nn.ReLU(),
                                   nn.Linear(16, self.num_class ),
                                   nn.ReLU(), nn.Softmax(dim=1))

    def operator(self, src, dst):
        E1 = (src + dst) / 2
        E2 = (src - dst).pow(2)
        return torch.cat([E1, E2], dim=1)

    def predict_edges(self, src, dst):
        embedding = self.gcn(self.G['features'], self.G.adj)
        src_embed = embedding[src]
        dst_embed = embedding[dst]
        E = self.operator(src_embed, dst_embed)
        # print(E[:5])
        return self.trans(E)

    def forward(self):
        'predict all the label'
        edges, weights = self.G.edges_tensor()
        poss_edge = self.predict_edges(edges[:, 0], edges[:, 1])
        poss_node = torch.zeros(self.num_nodes,
                                self.num_class).to(world.DEVICE)

        value = poss_edge * weights.unsqueeze(1)
        index = edges[:, 0].repeat(self.num_class, 1).t()
        poss_node.scatter_add_(0, index, value)
        poss_node /= self.G.neighbours_sum()
        return {'poss_node': poss_node, 'poss_edge': poss_edge}
