import os
import argparse
import torch
from os.path import join
from typing import NewType
import scipy.sparse as spp
import warnings
warnings.filterwarnings("ignore")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--dim', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--decay', type=float, default=5e-2)
    parser.add_argument('--epoch', type=int, default=3000)
    parser.add_argument('--decay_factor', type=float, default=0.8)
    parser.add_argument('--decay_patience', type=int, default=50)
    parser.add_argument('--stop_patience', type=int, default=100)
    parser.add_argument('--tensorboard', type=bool, default=False)
    parser.add_argument('--comment', type=str, default='Edge')
    parser.add_argument('--model', type=str, default='gcn')
    parser.add_argument('--gcn_hidden', type=int, default=16)
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument('--edge_lambda', type=float, default=0.0001)
    parser.add_argument('--semi_lambda', type=float, default=0.)
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--split', type=str, default='[0.6, 0.2, 0.2]')
    parser.add_argument('--semi_split', type=int, default=0)
    parser.add_argument('--quite', type=int, default=0)
    parser.add_argument('--topk', type=int, default=25)
    parser.add_argument('--pertrain', type=int, default=1)
    args = parser.parse_args()
    return args


ROOT = "/Users/gus/Desktop/edges"
CODE = join(ROOT, "code")
DATA = join(ROOT, "data")
LOG  = join(ROOT, "log")

GPU = torch.cuda.is_available()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CONFIG = {**vars(parse_args())}
CONFIG['the number of embed dims'] = CONFIG['dim']
CONFIG['comment'] = CONFIG['comment'] + '-' + CONFIG['model']
CONFIG['split'] = eval(CONFIG['split'])
SEED = CONFIG['seed']
TOPK = CONFIG['topk']
PERTRAIN = CONFIG['pertrain']
SEMI = CONFIG['semi_split']

NODE = NewType("NODE", int)
# assert nodes' index start from 0, for all dataset
GRAPH = NewType("GRAPH", spp.spmatrix)



"""annotation for start a functional section"""
#################################
# data helper
#################################
