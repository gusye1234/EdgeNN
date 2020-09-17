"""
load data and convert format.
General way to store a graph is scipy.sparse
"""
import world
from world import print
from world import GRAPH
import torch
import sys
import pickle as pkl
import numpy as np
import networkx as nx
from os.path import join
import scipy as sp
import scipy.sparse as spp
from sklearn.model_selection import ShuffleSplit


#################################
# main loaders
#################################

# Possible datasets layout
_all_datasets = {
    "chameleon": join(world.DATA, "chameleon"),
    "wisconsin": join(world.DATA, "wisconsin"),
    "squirrel" : join(world.DATA, "squirrel"),
    "cornell"  : join(world.DATA, "cornell"),
    "texas"    : join(world.DATA, "texas"),
    "film"     : join(world.DATA, "film"),
    "pubmed"   : join(world.DATA, "INDS"),
    # below datasets have unconnected nodes.
    "citeseer" : join(world.DATA, "INDS"),
    "cora"     : join(world.DATA, "INDS"),
}
_splits_files = join(world.DATA, "SPLITS")

def update(name, path):
    global _all_datasets
    _all_datasets[name] = path
def all_datasets():
    return list(_all_datasets)

# A: adj matrix, F: features, L: labels, G: nx or dgl graph
def loadAFL(name, splitFile=None, trainP=0.8, valP=0.1, testP=0.1):
    try:
        path = _all_datasets[name]
    except:
        raise KeyError(f"Please update your dataset {name}")

    if name in ['citeseer', 'cora', 'pubmed']:
        A, F, L =  load_ind(name)
        L = np.argmax(L, axis=-1)
    else:
        graphs = open(join(_all_datasets[name], 'out1_graph_edges.txt'), 'r')
        A = load_edges(graphs)
        FL_file = open(join(_all_datasets[name], 'out1_node_feature_label.txt'),'r')
        _, F, L = load_feature_label(FL_file, convert=(932 if (name == 'film') else False))
    if True:
        F = preprocess_features(F)
    # TODO, add process for cora and citeseer

    if splitFile:
        try:
            assert splitFile.startswith(name)
        except AssertionError:
            raise AssertionError(f"Wrong split file, expect {splitFile} starts with {name}")
        with np.load(join(_splits_files, splitFile)) as splits_file:
            train_mask = splits_file['train_mask']
            valid_mask = splits_file['val_mask']
            test__mask = splits_file['test_mask']
    else:
        try:
            assert (trainP + valP + testP) == 1
        except AssertionError:
            raise AssertionError(f"Expect separation {trainP}+{valP}+{testP}=1")
        (train_mask,
         valid_mask,
         test__mask) = generate_mask(F.shape[0], trainP, valP, testP)
    return Graph({
        "name"      : name,
        "labels"    : L,
        "features"  : F,
        "adj matrix": A,
        "test mask" : test__mask,
        "train mask": train_mask,
        "valid mask": valid_mask,
    })


#################################
# data helper
#################################
class Graph:
    def __init__(self, data_dict : dict):
        self.__dict = data_dict
        self.__pre_label = None
        self.__upd_label  = False
        self.__revelant_sets = None
        self.__class = np.unique(data_dict['labels'])
        self.__index_A = self.__dict['adj matrix'].tocsr()
        self.__edges_A =  self.__dict['adj matrix'].todok()

    def num_nodes(self):
        return len(self.__dict['labels'])

    def num_classes(self):
        return len(np.unique(self.__dict['labels']))

    def __repr__(self):
        flag = f"""
        {self.__dict['name']}:
            Adj matrix     -> {self.__dict['adj matrix'].shape}
            Feature matrix -> {self.__dict['features'].shape}
            Label vector   -> {self.__dict['labels'].shape}
            Spilt          -> {(
                sum(self.__dict['train mask']),
                sum(self.__dict['valid mask']),
                sum(self.__dict['test mask']),
            )}
        """
        return flag

    def __getitem__(self, key):
        return self.__dict[key]

    def update_predict(self, labels):
        """using newly predicted labels"""
        self.__pre_label = labels
        self.__upd_label = True

    def neighboors(self, node):
        return self.__index_A[node].nonzero()[1]

    def _Revelant_sets(self):
        """
        Calculate all the revelant sets, if there are new comming labels
        """
        if not self.__upd_label:
            if self.__revelant_sets is None:
                raise ValueError("labels haven't been updated")
            else:
                return
        del self.__revelant_sets
        self.__revelant_sets = {'unaligned' : []}
        self.__upd_label = False
        for pair, _ in self.__edges_A.items():
            src = pair[0]
            dst = pair[1]
            if self.__pre_label[src] == self.__pre_label[dst]:
                label = self.__pre_label[src]
                self.__revelant_sets[label] = self.__revelant_sets.get(label, [])
                self.__revelant_sets[label].append(pair)
            else:
                self.__revelant_sets['unaligned'].append(pair)

    def C_(self, label):
        """
        retrieve all the edges in C_{label}
        """
        self._Revelant_sets()
        assert label == 'unaligned' or label in self.__class
        return self.__revelant_sets[label]

    def intersection(self, node, label):
        """
        Return edges, Connect to node and in the C_{label}
        """
        assert self.__pre_label is not None
        label_u = self.__pre_label[node]
        if label_u != label:
            return []
        neigh = self.neighboors(node)
        index = (self.__pre_label[neigh] == label)
        for vk in neigh[index]:
            yield (node, vk)


#################################
# data helper
#################################

# adapted from Geom-GCN
def load_ind(name):
    suffix = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(suffix)):
        with open(f"{_all_datasets[name]}/ind.{name}.{suffix[i]}", 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(f"{_all_datasets[name]}/ind.{name}.test.index")
    test_idx_range = np.sort(test_idx_reorder)
    if name == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder),max(test_idx_reorder) + 1)
        tx_extended = spp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended
    features = np.array(np.vstack((allx.todense(), tx.todense())))
    # print(features.shape, test_idx_range)
    features[test_idx_reorder, :] = features[test_idx_range, :]
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    return nx.adjacency_matrix(nx.from_dict_of_lists(graph)), features, labels

def load_edges(handle):
    """
    parameter:
        file object
    return:
        scipy.sparse
    """
    G = nx.DiGraph()
    handle.readline()
    for line in handle:
        line = line.rstrip().split('\t')
        assert (len(line) == 2)
        G.add_edge(int(line[0]), int(line[1]))
    return nx.adjacency_matrix(G, sorted(G.nodes()))

def load_feature_label(handle, convert=None):
    """
    Assume the lines are sorted by node id.
    """
    handle.readline()

    ids, features, labels = [], [], []
    for line in handle.readlines():
        line = line.rstrip().split('\t')
        ids.append(int(line[0]))
        labels.append(int(line[2]))
        feature = np.asanyarray([int(k) for k in line[1].split(',')], dtype=np.int32)
        if convert:
            temp = np.zeros(convert, dtype=np.uint8)
            temp[feature] = 1
            feature = temp
        features.append(feature)
    return np.asanyarray(ids), np.asanyarray(features), np.asanyarray(labels)

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def preprocess_features(features):
    # rowsum = np.array(features.sum(1))
    # r_inv = np.power(rowsum, -1).flatten()
    # r_inv[np.isinf(r_inv)] = 0.
    # r_mat_inv = sp.diags(r_inv)
    # features = r_mat_inv.dot(features)
    rowsum = features.sum(1)
    rowsum[rowsum < 1e-9] = 1.
    normalized = (features.T/rowsum).T
    return normalized

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return adj_normalized

def generate_mask(length, trainP, valP, testP):
    """
    given proportion of train set, validation set, test set
    return:
        train mask : np.ndarray
        val mask   : np.ndarray    
        test mask  : np.ndarray
    """
    train_mask = np.zeros(length, dtype=np.uint8)
    valid_mask = np.zeros(length, dtype=np.uint8)
    test__mask = np.zeros(length, dtype=np.uint8)

    TV      = (trainP + valP)
    split_1 = ShuffleSplit(n_splits=1, train_size=TV)
    split_2 = ShuffleSplit(n_splits=1, train_size=float(trainP/TV))

    TV_index, t_index = split_1.split(train_mask).__next__()
    T_index , V_index    = split_2.split(TV_index).__next__()
    T_index , V_index    = TV_index[T_index], TV_index[V_index]

    train_mask[T_index] = 1
    valid_mask[V_index] = 1
    test__mask[t_index] = 1

    return train_mask, valid_mask, test__mask