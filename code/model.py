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

    def forward_withEdge(self, poss_edge):
        'Given edge feature, forward into nodes feature'
        edges, weights = self.G.edges_tensor()
        dims = poss_edge.shape[-1]
        poss_node = torch.zeros(self.num_nodes,
                                dims).to(world.DEVICE)

        value = poss_edge * weights.unsqueeze(1)
        index = edges[:, 0].repeat(dims, 1).t()
        poss_node.scatter_add_(0, index, value)
        recall_node = poss_node
        poss_node = poss_node/self.G.neighbours_sum()
        return {'poss_node': poss_node,
                'poss_edge': poss_edge,
                'recall_node': recall_node}

    def forward(self):
        'predict all the label'
        edges, weights = self.G.edges_tensor()
        poss_edge = self.predict_edges(edges[:, 0], edges[:, 1])
        return self.forward_withEdge(poss_edge)


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
        return self.operator_sy1(src, dst)

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


class EmbeddingP_noalign(BasicModel):
    def __init__(self, CONFIG, G: Graph):
        super(EmbeddingP_noalign, self).__init__()
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

    def operator_sy2(self, src, dst):
        E1 = (src + dst) / 2
        E2 = (src - dst).abs()
        return torch.cat([E1, E2], dim=1)

    def init(self):
        # self.node_embedding = torch.nn.Embedding(
        #     self.num_nodes, self.num_dims
        # )
        self.embed = nn.Linear(self.feature_dim, self.num_dims)
        self.trans = nn.Sequential(
            nn.Linear(self.num_dims * 2, self.num_class),
            nn.Softmax(dim=1))
        # nn.init.normal_(self.node_embedding.weight)

    def predict_edges(self, src, dst):
        embed = self.embed(self.G['features'])
        src_embed = embed[src]
        dst_embed = embed[dst]
        E = self.operator(src_embed, dst_embed)
        return self.trans(E)


class EmbeddingP_multiLayer(BasicModel):
    def __init__(self, CONFIG, G: Graph):
        super(EmbeddingP_multiLayer, self).__init__()
        self.G = G
        self.recursion = 2
        self.num_nodes = CONFIG['the number of nodes']
        self.num_dims = CONFIG['the number of embed dims']
        self.num_class = CONFIG['the number of classes']
        self.hidden_dims = CONFIG['gcn_hidden']
        self.feature_dim = CONFIG['the dimension of features']
        self.init()

    # def operator(self, src, dst):
    #     return src + dst

    def operator(self, src, dst):
        return self.operator_sy2(src, dst)

    def operator_naive(self, src, dst):
        return torch.cat([src, dst], dim=1)

    def operator_sy1(self, src, dst):
        E1 = (src + dst) / 2
        E2 = (src - dst).pow(2)
        return torch.cat([E1, E2], dim=1)

    def operator_sy2(self, src, dst):
        E1 = (src + dst) / 2
        E2 = (src - dst).abs()
        return torch.cat([E1, E2], dim=1)

    def init(self):
        self.embed = nn.Linear(self.feature_dim, self.num_class + 1)
        self.trans = nn.Linear((self.num_class + 1)*2, self.num_class + 1)
        self.f = nn.Softmax(dim=1)

    def predict_edges(self, src, dst):
        # 1
        embed = self.embed(self.G['features'])
        for layer in range(1, self.recursion+1):
            src_embed, dst_embed = embed[src], embed[dst]
            E = self.trans(self.operator(src_embed, dst_embed))
            if layer != self.recursion:
                embed = self.forward_withEdge(E)['poss_node']
        return self.f(E)



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


class GATSingleAttentionHead(nn.Module):
    def __init__(self, in_feats, out_feats, activation, dropout_prob):
        super(GATSingleAttentionHead, self).__init__()
        self.in_feats_dropout = nn.Dropout(dropout_prob)
        self.linear = nn.Linear(in_feats, out_feats, bias=False)
        nn.init.xavier_uniform_(self.linear.weight)
        self.attention_linear = nn.Linear(2 * out_feats, 1, bias=False)
        nn.init.xavier_uniform_(self.attention_linear.weight)
        self.attention_head_dropout = nn.Dropout(dropout_prob)
        self.linear_feats_dropout = nn.Dropout(dropout_prob)
        self.bias = nn.Parameter(
            th.ones(1, out_feats, dtype=th.float32, requires_grad=True))
        nn.init.xavier_uniform_(self.bias.data)
        self.activation = activation

    def calculate_node_pairwise_attention(self, edges):
        h_concat = th.cat([edges.src['Wh'], edges.dst['Wh']], dim=1)
        e = self.attention_linear(h_concat)
        e = F.leaky_relu(e, negative_slope=0.2)
        return {'e': e}

    def message_func(self, edges):
        return {'Wh': edges.src['Wh'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        a = F.softmax(nodes.mailbox['e'], dim=1)
        a_dropout = self.attention_head_dropout(a)
        Wh_dropout = self.linear_feats_dropout(nodes.mailbox['Wh'])
        return {'h_new': th.sum(a_dropout * Wh_dropout, dim=1)}

    def forward(self, g, feature):
        Wh = self.in_feats_dropout(feature)
        Wh = self.linear(Wh)
        g.ndata['Wh'] = Wh
        g.apply_edges(self.calculate_node_pairwise_attention)
        g.update_all(self.message_func, self.reduce_func)
        h_new = g.ndata.pop('h_new')
        h_new = self.activation(h_new + self.bias)
        return h_new


# Adapted from https://docs.dgl.ai/tutorials/models/1_gnn/9_gat.html
class GAT(nn.Module):
    def __init__(self, in_feats, out_feats, activation, num_heads,
                 dropout_prob, merge):
        super(GAT, self).__init__()
        self.attention_heads = nn.ModuleList()
        for _ in range(num_heads):
            self.attention_heads.append(
                GATSingleAttentionHead(in_feats, out_feats, activation,
                                       dropout_prob))
        self.merge = merge

    def forward(self, g, feature):
        all_attention_head_outputs = [
            head(g, feature) for head in self.attention_heads
        ]
        if self.merge == 'cat':
            return th.cat(all_attention_head_outputs, dim=1)
        else:
            return th.mean(th.stack(all_attention_head_outputs), dim=0)


class GATNet(nn.Module):
    def __init__(self, num_input_features, num_output_classes, num_hidden,
                 num_heads_layer_one, num_heads_layer_two, dropout_rate):
        super(GATNet, self).__init__()
        self.gat1 = GAT(num_input_features, num_hidden, F.elu,
                        num_heads_layer_one, dropout_rate, 'cat')
        self.gat2 = GAT(num_hidden * num_heads_layer_one, num_output_classes,
                        lambda x: x, num_heads_layer_two, dropout_rate, 'mean')

    def forward(self, g, features):
        x = self.gat1(g, features)
        x = self.gat2(g, x)
        return x