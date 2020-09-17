"""
Store all the trainable model
"""
import world
from world import CONFIG
import torch as th
from torch import nn
from torch.nn import Module



class BasicModel(Module):
    def __init__(self, *args, **kwargs):
        super(BasicModel, self).__init__()

    def predict_edges(self, src, dst):
        raise NotImplementedError


class EmbeddingP(BasicModel):
    def __init__(self):
        super(EmbeddingP, self).__init__()
        self.num_nodes = CONFIG['the number of nodes']
        self.num_dims  = CONFIG['the number of embed dims']
        self.num_class = CONFIG['the number of classes']
        self.init()

    def operator(self, src, dst):
        return src + dst

    def init(self):
        self.node_embedding = th.nn.Embedding(
            self.num_nodes, self.num_dims
        )
        self.voting = nn.Sequential(
            nn.Linear(self.num_dims, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_class + 1),
            nn.ReLU(),
            nn.Softmax(dim=1)
        )
        nn.init.normal_(self.node_embedding.weight)

    def predict_edges(self, src, dst):
        src_embed = self.node_embedding(src)
        dst_embed = self.node_embedding(dst)
        E = self.operator(src_embed, dst_embed)
        return self.voting(E)