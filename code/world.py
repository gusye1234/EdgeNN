import os
import argparse
import torch
from os.path import join
from typing import NewType
import scipy.sparse as spp
from pprint import pprint as print
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora')
    # TODO
    parser.add_argument('--dim', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--decay', type=float, default=1e-4)
    parser.add_argument('--epoch', type=int, default=1000)

    args = parser.parse_args()
    return args


ROOT = "/Users/gus/Desktop/edges"
CODE = join(ROOT, "code")
DATA = join(ROOT, "data")
GPU = torch.cuda.is_available()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CONFIG = {**vars(parse_args())}
CONFIG['the number of embed dims'] = CONFIG['dim']



NODE = NewType("NODE", int)
# assert nodes' index start from 0, for all dataset
GRAPH = NewType("GRAPH", spp.spmatrix)



"""annotation for start a functional section"""
#################################
# data helper
#################################
