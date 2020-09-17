"""
helper function.
Including 
    pretty output
    log
    sample
"""
import world
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import dok_matrix, csr_matrix
from tabulate import tabulate

#################################
# metrics
#################################

def accuracy(pred, groundtruth, mask):
    with torch.no_grad():
        return torch.eq(pred[mask], groundtruth[mask]).float().mean().item()

#################################
# analyse and visualize data
#################################

def summary_dataset(data_dict):
    D = data_dict
    feat_dim = data_dict["features"].shape[1]
    node_num = data_dict["adj matrix"].shape[0]
    labels   = np.unique(data_dict["labels"])
    T_split  = D["train mask"].sum()/node_num
    V_split  = D["valid mask"].sum()/node_num
    t_split  = D["test mask"].sum()/node_num
    split = f"{T_split:.2f},{V_split:.2f},{t_split:.2f}"
    summary = [
        ["node", node_num],
        ["feature dim", feat_dim],
        ["labels", labels],
        ["split", split],
        ["edges", revelant_sets(data_dict)]
    ]
    see = {"1" : 2, "3": 4}
    # print(tabulate(summary))

def revelant_sets(data_dict):
    D = data_dict
    labels = D['labels']
    adj : dok_matrix = D['adj matrix'].todok()
    C = np.unique(labels)
    total_edges = 0
    counts = {f"C{label}": 0 for label in C}
    counts['unaligned'] = 0
    with timer():
        for pair, value in adj.items():
            total_edges += 1
            if (labels[pair[0]] == labels[pair[1]]):
                counts[f"C{labels[pair[0]]}"] += 1
            else:
                counts['unaligned'] += 1
    print(timer.get())
    counts = {name : round(value/total_edges, 2) for name, value in counts.items()}
    return counts

def dict2table(table : dict):
    return tabulate([[key, value] for key, value in table.items()])


def state_edges_distribution(data_dict):
    D = data_dict
    labels = D['labels']
    adj: csr_matrix = D['adj matrix'].tocsr()
    C = np.unique(labels)
    node_num = data_dict["adj matrix"].shape[0]
    M = np.zeros((len(C), len(C)))
    for node in range(node_num):
        N_node = adj[node].nonzero()[1]
        N_label= labels[N_node]
        M[labels[node], N_label] += 1
    return M

def state_node_by_edges(data_dict):
    D = data_dict
    adj: csr_matrix = D['adj matrix'].tocsr()
    labels = D['labels']
    C = np.unique(labels)
    node_num = data_dict["adj matrix"].shape[0]
    M = np.zeros((len(C), len(C)))
    for label in C:
        index = np.where(labels == label)[0]
        for node in index:
            N_node = adj[node].nonzero()[1]
            if not len(N_node):
                continue
            N_label = labels[N_node]
            M[label, _mode(N_label)] += 1
    return np.round((M.T / M.sum(1)).T, 2)


def _mode(data : np.ndarray):
    all_poss = np.unique(data)
    counts = [np.sum(data == label) for label in all_poss]
    return all_poss[np.argmax(counts)]

def vis_ConfusedMatrix(data_dict):
    M = state_edges_distribution(data_dict)
    labels = data_dict['labels']
    C = np.unique(labels)
    plt.xticks(C)
    plt.yticks(C)
    plt.imshow(M, cmap=plt.cm.hot)
    plt.colorbar()
    plt.title(data_dict['name'])
    plt.savefig(f"../{data_dict['name']}.png")
    plt.show()
    plt.close()


def set_seed(seed):
    '''
        fix Randomness
    '''
    import random
    import torch
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def minibatch(*tensors, **kwargs):

    batch_size = kwargs.get('batch_size', world.config['bpr_batch_size'])

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def shuffle(*arrays, **kwargs):

    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have ' 'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result


def TO(*tensors, **kwargs):
    if kwargs.get("device"):
        device = torch.device(kwargs['device'])
    else:
        device = torch.device('cpu')
    results = []
    for tensor in tensors:
        results.append(tensor.to(device))
    return results


class timer:
    """
    Time context manager for code block
    """
    from time import time
    TAPE = [-1]  # global time record

    @staticmethod
    def get():
        return timer.TAPE[-1]

    def __init__(self, tape=None):
        self.tape = tape or timer.TAPE

    def __enter__(self):
        self.start = timer.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.tape.append(timer.time() - self.start)


if __name__ == "__main__":
    from data import all_datasets
    from data import loadAFL
    for dataset in all_datasets():
        vis_ConfusedMatrix(loadAFL(dataset))