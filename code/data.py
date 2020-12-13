"""
load data and convert format.
General way to store a graph is scipy.sparse
"""
import world
import torch
import sys
import os
import pickle as pkl
import numpy as np
import networkx as nx
from os.path import join
import scipy as sp
import scipy.sparse as spp
from sklearn.model_selection import ShuffleSplit
import rich


#################################
# main loaders
#################################

# Possible datasets layout
_all_datasets = {
    "cham": join(world.DATA, "chameleon"),
    "wisc": join(world.DATA, "wisconsin"),
    "squi" : join(world.DATA, "squirrel"),
    "corn"  : join(world.DATA, "cornell"),
    "texa"    : join(world.DATA, "texas"),
    "film"     : join(world.DATA, "film"),
    "pubm"   : join(world.DATA, "INDS"),
    # below datasets have unconnected nodes.
    "cite" : join(world.DATA, "INDS"),
    "cora"     : join(world.DATA, "INDS"),
}
_splits_files = join(world.DATA, "SPLITS")
_unconnected_files = join(world.DATA, 'UNCONNECTED')


def update(name, path):
    global _all_datasets
    _all_datasets[name] = path
def all_datasets():
    return list(_all_datasets)

def mask_name(name, split):
    split = [str(i) for i in split]
    split = "-".join(split)
    mask = f"{name}-{world.SEED}-{split}.npz"
    return mask

# A: adj matrix, F: features, L: labels, G: nx or dgl graph
def loadAFL(name, splitFile=None, split=[0.6, 0.2, 0.2], need_dgl=False):
    try:
        path = _all_datasets[name]
    except:
        raise KeyError(f"Please update your dataset {name}")

    if name in ['cite', 'cora', 'pubm']:
        A, F, L, test_num =  load_ind(name)
        L = np.argmax(L, axis=-1)
    else:
        graphs = open(join(_all_datasets[name], 'out1_graph_edges.txt'), 'r')
        A = load_edges(graphs)
        FL_file = open(join(_all_datasets[name], 'out1_node_feature_label.txt'),'r')
        _, F, L = load_feature_label(FL_file, convert=(932 if (name == 'film') else False))
    if True:
        F = preprocess_features(F)

    if name in ['cora', 'cite']:
        connected_subset = process_unconnected(name, L)

    if world.SEMI and (name in ['cora', 'cite', 'pubm']):
        (train_mask,
        valid_mask,
        test__mask) = generate_mask_semi(F.shape[0],
                                        len(np.unique(L)),
                                        test_num)
    else:
        if os.path.exists(join(path, mask_name(name, split))):
            npz_name = mask_name(name, split)
            mask = np.load(join(path, npz_name))
            train_mask = mask['train_mask']
            valid_mask = mask['valid_mask']
            test__mask = mask['test_mask']
            rich.print(f"[bold yellow]Load Spliting from {npz_name}[/bold yellow]")
        # if splitFile:
        #     try:
        #         assert splitFile.startswith(name)
        #     except AssertionError:
        #         raise AssertionError(f"Wrong split file, expect {splitFile} starts with {name}")
        #     with np.load(join(_splits_files, splitFile)) as splits_file:
        #         train_mask = splits_file['train_mask']
        #         valid_mask = splits_file['val_mask']
        #         test__mask = splits_file['test_mask']
        else:
            try:
                assert sum(split) == 1
            except AssertionError:
                raise AssertionError(f"Expect separation {trainP}+{valP}+{testP}=1")
            trainP, valP, testP = split
            if name in ['cora', 'cite']:
                (train_mask,
                valid_mask,
                test__mask) = generate_mask(F.shape[0], trainP, valP, testP, subset=connected_subset)
            else:
                (train_mask,
                valid_mask,
                test__mask) = generate_mask(F.shape[0], trainP, valP, testP)
            mask = join(path, mask_name(name, split))
            np.savez(mask, train_mask=train_mask,
                     valid_mask=valid_mask, test_mask=test__mask)
            print(f"Save splits to {mask}")
    return Graph({
        "name": name,
        "labels": torch.LongTensor(L),
        "features": torch.Tensor(F),
        "adj matrix": A + spp.eye(A.shape[0]),
        "test mask":torch.BoolTensor(test__mask),
        "train mask": torch.BoolTensor(train_mask),
        "valid mask": torch.BoolTensor(valid_mask),
    }, need_dgl=need_dgl)


#################################
# data helper
#################################
class Graph:
    '''wrap data with set operations'''
    def __init__(self, data_dict : dict, need_dgl=False):
        self.device = 'cpu'
        self.__dict = data_dict
        self.__pre_label = None
        self.__upd_label  = False
        self.__revelant_sets = None
        self.__class = np.unique(data_dict['labels'])
        self.__index_A = self.__dict['adj matrix'].tocsr()
        # index_A is fast for row operations
        self.__edges_A = self.__dict['adj matrix'].todok()
        # edges_A is stored in (Pair, value) format
        self.adj = sparse_mx_to_torch_sparse_tensor(
            preprocess_adj(self.__dict['adj matrix']))

        self.dgl_g = DGLGraph(self.__dict['adj matrix']) if need_dgl else None

        self.sum = torch.FloatTensor(self.__index_A.sum(1))
        Edges = torch.Tensor([(pair[0], pair[1], weights) for pair, weights in self.edges()])
        self.tensor_edges = Edges[:, :2].long()
        self.tensor_weights = Edges[:, 2]
        
        if need_dgl:
            self.dgl_g = self._perpare_dgl()
        else:
            self.dgl_g = None

    def num_nodes(self):
        return len(self.__dict['labels'])

    def num_classes(self):
        return len(np.unique(self.__dict['labels']))

    def _perpare_dgl(self):
        from dgl import DGLGraph
        import dgl.init as init
        dgl_g = DGLGraph(self.__dict['adj matrix'])
        deg = dgl_g.in_degrees().float()
        norm = torch.pow(degs, -0.5)
        norm[torch.isinf(norm)] = 0.
        dgl_g.ndata['norm'] = norm.unsqueeze(1)
        dgl_g.set_n_initializer(init.zero_initializer)
        dgl_g.set_e_initializer(init.zero_initializer)
        return dgl_g

    def getCSR(self):
        return self.__index_A

    def getDOK(self):
        return self.__edges_A

    def to(self, device):
        self.__dict['labels'] = self.__dict['labels'].to(device)
        self.__dict['features'] = self.__dict['features'].to(device)
        self.adj = self.adj.to(device)
        self.tensor_edges = self.tensor_edges.to(device)
        self.tensor_weights = self.tensor_weights.to(device)
        self.sum = self.sum.to(device)
        if self.dgl_g is not None:
            self.dgl_g.to(device)
        self.device = device
        return self

    def __repr__(self):
        length = len(self.__dict['train mask'])
        edge_length = len(list(self.edges()))
        splits = f"{self.__dict['train mask'].sum().item()/length:.2f}," + f"{self.__dict['valid mask'].sum().item()/length:.2f}," + f"{self.__dict['test mask'].sum().item()/length:.2f}"
        if not world.SEMI:
            assert all((self.__dict['train mask'] + self.__dict['valid mask'] + self.__dict['test mask']) < 2)
        flag = f"""
        {self.__dict['name']}({str(self.device)}) - {"SEMI" if world.SEMI else "FULL"}:
            Adj matrix     -> {self.__dict['adj matrix'].shape}
            Feature matrix -> {self.__dict['features'].shape}
            Label          -> {np.unique(self.__dict['labels'].cpu().numpy())}
            Spilt          -> {splits} = {length}
            EDGE:
                Label ratio-> {(self.count_edges(self.__dict['train mask'])[0]+self.count_edges(self.__dict['train mask'])[1])/edge_length:.2f}
                Train      -> {self.count_edges(self.__dict['train mask'])}
                Valid      -> {self.count_edges(self.__dict['valid mask'])}
                Test       -> {self.count_edges(self.__dict['test mask'])}
        """
        return flag

    def __getitem__(self, key):
        return self.__dict[key]

    def count_edges(self, mask):
        groundTruth = self.__dict['labels']
        edge_count = [0,0,0,0]
        for edge, _ in self.edges():
            if mask[edge[0]] or mask[edge[1]]:
                if mask[edge[0]] and mask[edge[1]]:
                    if groundTruth[edge[0]] == groundTruth[edge[1]]:
                        edge_count[0] += 1
                    else:
                        edge_count[1] += 1
                else:
                    # labeled = edge[0] if mask[edge[0]] else edge[1]
                    edge_count[2] += 1
            else:
                edge_count[3] += 1
        return edge_count

    def neighbours(self, node):
        return self.__index_A[node].nonzero()[1]

    def neighbours_sum(self):
        return self.sum

    def nodes(self):
        for i in range(self.__index_A.shape[0]):
            yield i

    def edges(self):
        for pair, weight in self.__edges_A.items():
            yield (pair, weight)

    def edges_tensor(self):
        return self.tensor_edges, self.tensor_weights

    def splitByEdge(self, ratio:float):
        EDGES = list(self.edges())
        total_edges = len(EDGES)
        wanted_edges = int(total_edges*ratio)
        train_mask = np.zeros_like(self.__dict["train mask"], dtype=np.bool)
        valid_mask = np.zeros_like(self.__dict["valid mask"], dtype=np.bool)
        test_mask = np.zeros_like(self.__dict["test mask"], dtype=np.bool)
        edge_order = np.arange(total_edges)
        np.random.shuffle(edge_order)
        already_edge = set()
        for index in edge_order:
            if len(already_edge) >= wanted_edges:
                break
            node1, node2 = EDGES[index][0]
            labeled_num = np.sum(train_mask[[node1, node2]])
            train_mask[node1] = 1
            labled_neighbours1 = self.neighbours(node1)[train_mask[self.neighbours(node1)]]
            for neighbours1 in labled_neighbours1:
                already_edge.add((int(node1), int(neighbours1)))
                already_edge.add((int(neighbours1), int(node1)))
            train_mask[node2] = 1
            labled_neighbours2 = self.neighbours(node2)[train_mask[self.neighbours(node2)]]
            for neighbours2 in labled_neighbours2:
                already_edge.add((int(node2), int(neighbours2)))
                already_edge.add((int(neighbours2), int(node2)))
            already_edge.add((int(node1), int(node2)))
            already_edge.add((int(node2), int(node1)))
            # exit()
        print(f"split {total_edges} of {len(already_edge)} ~ {wanted_edges}")
        unlabeled = np.where(train_mask == 0)[0]
        valid_size = len(unlabeled)//2
        # test_size = len(unlabeled) - valid_size
        # split valid set and test set in half, half
        where = np.random.choice(np.arange(len(unlabeled)), size=(valid_size,), replace=False)
        valid_where = unlabeled[where]
        valid_mask[valid_where] = 1
        test_mask = (1 - train_mask) - valid_mask
        self.__dict['train mask'] = torch.ByteTensor(train_mask)
        self.__dict['valid mask'] = torch.ByteTensor(valid_mask)
        self.__dict['test mask'] = torch.ByteTensor(test_mask)


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
    if name == 'cite':
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
    test_num = len(ty)
    return nx.adjacency_matrix(nx.from_dict_of_lists(graph)), features, labels, test_num

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
            temp = np.zeros(convert, dtype=np.bool)
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
    rowsum = features.sum(1)
    rowsum[rowsum < 1e-9] = 1.
    normalized = (features.T/rowsum).T
    return normalized

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj)
    return adj_normalized

def normalize_adj(adj):
    rowsum = np.array(adj.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = spp.diags(r_inv)
    mx = r_mat_inv.dot(adj)
    return mx


def process_unconnected(name, labels):
    disconnected_node_file_path = join(_unconnected_files, f'{name}_unconnected_nodes.txt')
    with open(disconnected_node_file_path) as disconnected_node_file:
        disconnected_node_file.readline()
        disconnected_nodes = []
        for line in disconnected_node_file:
            line = line.rstrip()
            disconnected_nodes.append(int(line))

    disconnected_nodes = np.array(disconnected_nodes)
    connected_nodes = np.setdiff1d(np.arange(labels.shape[0]), disconnected_nodes)
    return connected_nodes


def generate_mask(length, trainP, valP, testP, subset=None):
    """
    given proportion of train set, validation set, test set
    return:
        train mask : np.ndarray
        val mask   : np.ndarray    
        test mask  : np.ndarray
    """
    train_mask = np.zeros(length, dtype=np.bool)
    valid_mask = np.zeros(length, dtype=np.bool)
    test__mask = np.zeros(length, dtype=np.bool)
    if subset is not None:
        length_= length
        length = len(subset)

    fake_mask = np.zeros(length, dtype=np.bool)

    TV      = (trainP + valP)
    split_1 = ShuffleSplit(n_splits=1, train_size=TV)
    split_2 = ShuffleSplit(n_splits=1, train_size=float(trainP/TV))

    TV_index, t_index = split_1.split(fake_mask).__next__()
    T_index , V_index = split_2.split(TV_index).__next__()
    T_index , V_index = TV_index[T_index], TV_index[V_index]

    if subset is not None:
        train_mask[subset[T_index]] = 1
        valid_mask[subset[V_index]] = 1
        test__mask[subset[t_index]] = 1
    else:
        train_mask[T_index] = 1
        valid_mask[V_index] = 1
        test__mask[t_index] = 1
    # print("SEE", train_mask.sum(), valid_mask.sum(), test__mask.sum())
    return train_mask, valid_mask, test__mask


def generate_mask_semi(length, label_num, test_num):
    '''
        only for cite, cora, pubm datasets, following the setting of 
        Yang et al. Revisiting Semi-Supervised Learning with Graph Embeddings
    '''
    total_labeled = 20*label_num
    train_mask = np.zeros(length, dtype=np.bool)
    valid_mask = np.zeros(length, dtype=np.bool)
    test__mask = np.zeros(length, dtype=np.bool)
    train_mask[:total_labeled] = 1
    valid_mask[-test_num:] = 1
    test__mask[-test_num:] = 1

    return train_mask, valid_mask,test__mask

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)