"""
helper function.
Including 
    pretty output
    log
    sample
"""
from os.path import join
import world
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import dok_matrix, csr_matrix
from tabulate import tabulate
from data import Graph

#################################
# metrics
#################################

def accuracy(pred, groundtruth, mask):
    with torch.no_grad():
        return torch.eq(pred[mask], groundtruth[mask]).float().mean().item()*100

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

def dict2table(table : dict, headers='row'):
    if headers == 'row':
        tab = []
        for key in sorted(list(table)):
            tab.append([key, table[key]])
        return tabulate(tab, floatfmt=".4f")
    elif headers == 'firstrow':
        head = []
        data = []
        for key in sorted(list(table)):
            head.append(key)
            data.append(table[key])
        return tabulate([head, data], headers='firstrow', floatfmt=".4f")


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

def uniqueFileFlag():
    from world import CONFIG
    return f"{CONFIG['comment']}-{CONFIG['dataset']}-{CONFIG['dim']}"

def Path3(father, son, grandson):
    '''father/son/grandson'''
    return join(join(father, son), grandson)

class timer:
    """
    Time context manager for code block
        with timer():
            do something
        timer.get()
    """
    from time import time
    TAPE = [-1]  # global time record
    NAMED_TAPE = {}

    @staticmethod
    def get():
        if len(timer.TAPE) > 1:
            return timer.TAPE.pop()
        else:
            return -1

    @staticmethod
    def dict(select_keys = None):
        hint = "|"
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                hint = hint + f"{key}:{value:.2f}|"
        else:
            for key in select_keys:
                value = timer.NAMED_TAPE[key]
                hint = hint + f"{key}:{value:.2f}|"
        return hint

    @staticmethod
    def zero(select_keys = None):
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                timer.NAMED_TAPE[key] = 0
        else:
            for key in select_keys:
                timer.NAMED_TAPE[key] = 0

    def __init__(self, tape=None, **kwargs):
        if kwargs.get('name'):
            timer.NAMED_TAPE[kwargs['name']] = timer.NAMED_TAPE[kwargs['name']] if timer.NAMED_TAPE.get(kwargs['name']) else 0.
            self.named = kwargs['name']
        else:
            self.named = False
            self.tape = tape or timer.TAPE

    def __enter__(self):
        self.start = timer.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.named:
            timer.NAMED_TAPE[self.named] += timer.time() - self.start
        else:
            self.tape.append(timer.time() - self.start)


class EarlyStop:
    def __init__(self, patience, model, filename):
        self.patience = patience
        self.model = model
        self.filename = filename
        self.suffer = 0
        self.best = 0
        self.best_result = None
        self.best_epoch = 0
        self.mean = 0
        self.sofar = 1

    def step(self, epoch, performance, where):
        if performance[where] < self.mean:
            self.suffer += 1
            if self.suffer >= self.patience:
                return True
            self.sofar += 1
            self.mean = self.mean * \
                        (self.sofar -1) / self.sofar + \
                        performance[where] / self.sofar
            print(f" * Suffer {self.suffer:.4f} : {self.mean:.4f}", end='')
            return False
        else:
            self.suffer = 0
            self.mean = performance[where]
            self.sofar = 1
            self.best = performance[where]
            self.best_result = performance
            self.best_epoch = epoch
            self.best_model = self.model.state_dict()
            # torch.save(self.model.state_dict(), self.filename)
            return False

def table_info(stop_at, seed):
    from world import CONFIG
    info = f'''
#################################################
seed: {seed}, lr:{CONFIG['lr']}, decay:{CONFIG['decay']}, semi: {CONFIG['semi_lambda']}, 
edge: {CONFIG['edge_lambda']}, factor:{CONFIG['decay_factor']}, stop:{stop_at}/{CONFIG['epoch']}
splits: {CONFIG['split']}\n'''
    return info


def get_Flag(prediction, groundtruth, k):
    """compute NDCG for one seq

    Args:
        prediction (ndarray): shape (node_num, ) (sorted)
        groundtruth (ndarray): shape (test_num, )
        k (int): top-K
    
    Returns:
        dict: 
            key: rate: A 0-1 mask with shape (k, ), 
                       indicate if the topk elements 
                       in prediction are in groundtruth
            key: truth: The groundtruth
    """
    flag = {}
    seq_need = prediction[:k]
    flag["rate"] = np.asanyarray(list(map(lambda x: x in groundtruth, seq_need)),
                                 dtype='float')
    flag["truth"] = groundtruth
    return flag


def Recall_Precision_AtK(flag, k):
    """compute recall, precision for one seq

    Args:
        flag (dict) 
        k (int): top-K
    """
    R = flag["rate"][:k]
    precision = R.sum()/k
    recall = R.sum()/len(flag['truth'])
    return {"precision":precision, "recall":recall}

def NDCG_AtK(flag, k):
    """compute recall, precision for one seq

    Args:
        flag (dict)
        k (int): top-K
    """

    k = len(flag["rate"]) if k >= len(flag["rate"]) else k
    R = flag["rate"][:k]
    num_data = len(flag['truth'])
    assert num_data > 0
    Max_len = num_data if k >= num_data else k
    ideal_R = np.zeros((k, ))
    ideal_R[:Max_len] = 1.
    idcg = np.sum(ideal_R * 1./np.log2(np.arange(2, k+2)))
    dcg = np.sum(R * 1. / np.log2(np.arange(2, k + 2)))
    ndcg = dcg / idcg if abs(idcg) > 1e-9 else 0
    return ndcg

def HR_AtK(flag, k):
    """Abandon
    compute recall, precision for one seq

    Args:
        flag (dict) 
        k (int): top-K
    """
    pass

def Group_ByPrediction_mask_all(Overall,
                                dataset: Graph,
                                train_mask=None,
                                test_mask=None,
                                sortby='recall',
                                plot=False):
    """helper function to evaluate rank

    Args:
        Overall (dict): check run_topk.py
        dataste (Graph):
        sortby (str, optional): how to sort prediction. Defaults to 'recall'.

    Returns:
        dict: key: label, value: (sorted prediction, groundtruth)
    """
    train_mask = train_mask.numpy().astype(np.bool)
    labels = dataset['labels'][test_mask].cpu().numpy()
    nodes_in_mask = torch.arange(len(test_mask))[test_mask].numpy()
    # labels = dataset['labels'][~train_mask].cpu().numpy()
    # nodes_in_mask = torch.arange(len(test_mask))[~train_mask].numpy()
    table = {}
    classes = np.unique(labels)
    # Overall = {name: value[~train_mask] for name, value in Overall.items()}
    Overall = {name: value[test_mask] for name, value in Overall.items()}
    # nodes_out_train = np.arange(len(test_mask))[~train_mask]
    nodes_out_train = np.arange(len(test_mask))[test_mask]
    test_labels = {}
    logs = []
    for label in classes:
        pred_where = nodes_out_train
        truth_where = np.where(labels == label)[0]
        truth_where = nodes_in_mask[truth_where]
        if plot:
            logs.append(Overall[sortby][:, label])
        if sortby == 'recall':
            recall = Overall['recall'][:, label]
            index = np.argsort(recall)[::-1]
        elif sortby == 'precision':
            precision = Overall['precision'][:, label]
            index = np.argsort(precision)[::-1]
        elif sortby == 'f1':
            recall = Overall['recall'][:, label]
            recall_max = np.max(recall) if len(recall) > 0 else 1
            recall = recall / recall_max
            precision = Overall['precision'][:, label]
            f1 = (2 * recall * precision / (recall + precision))
            index = np.argsort(f1)[::-1]
        # print(label, len(pred_where), len(truth_where),np.std(dataset.neighbours_sum().cpu().numpy()[truth_where]))
        test_labels[label] = len(truth_where)
        sorted_pred = pred_where[index]
        table[label] = (sorted_pred, truth_where)
    if plot:
        plot_curve(logs, classes)
    return table, test_labels

def peak(dataset: Graph, score, topk=world.TOPK):
    train_mask = dataset['train mask'].numpy()
    score[train_mask] = 0
    # matrix = dataset.generate_co()
    # matrix = dataset.generate_recall()
    matrix = dataset.generate_precision()
    table = {}
    for label in range(len(np.unique(dataset['labels']))):
        truth = (dataset['labels'] == label).numpy()
        index = np.argsort(score[:, label])[::-1]
        truth = truth[index]
        table[label] = (
            np.sum(matrix[index[:topk], label][truth[:topk]] > 0.95) /
            np.sum(truth[:topk]), np.sum(truth[:topk]))

    return table

def count_inside(dataset : Graph, threshold=0.95):
    # mask = dataset['test mask'].numpy()
    labels = dataset['labels'].numpy()
    # matrix = dataset.generate_co()
    # matrix = dataset.generate_recall()
    matrix = dataset.generate_precision()
    count = np.zeros((len(np.unique(labels)), ))
    for node in range(len(labels)):
        label = labels[node]
        if matrix[node, label] >= threshold:
            count[label] += 1
    return count

def topk_metrics(rank_table, top_k):
    rate_table = {
        label: get_Flag(pred, truth, top_k)
        for label, (pred, truth) in rank_table.items()
    }
    recall_precision = {
        label: Recall_Precision_AtK(flag, top_k)
        for label, flag in rate_table.items()
    }
    # --------------------------------------------
    recall = {
        label: R_P['recall']
        for label, R_P in recall_precision.items()
    }
    precision = {
        label: R_P['precision']
        for label, R_P in recall_precision.items()
    }
    NDCG = {
        label: NDCG_AtK(flag, top_k)
        for label, flag in rate_table.items()
    }
    return recall, precision, NDCG


def plot_curve(values, labels, index=None):
    import matplotlib.pyplot as plt
    now = 0
    for i, value in enumerate(values):
        x = np.arange(now, now+len(value))
        # print(x)
        now += len(value)
        if index is None:
            value = np.sort(value)
        plt.plot(x, value, label=str(labels[i]))
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)
    # plt.ylim([0.1, 1.0])
    # plt.xlim(left=-500)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    from data import all_datasets
    from data import loadAFL
    for dataset in all_datasets():
        print(loadAFL(dataset))
        # vis_ConfusedMatrix(loadAFL(dataset))



# def Group_ByPrediction_mask(Overall, dataset: Graph, train_mask = None,test_mask = None,sortby='recall', plot=False):
#     """helper function to evaluate rank

#     Args:
#         Overall (dict): check run_topk.py
#         dataste (Graph):
#         sortby (str, optional): how to sort prediction. Defaults to 'recall'.

#     Returns:
#         dict: key: label, value: (sorted prediction, groundtruth)
#     """
#     train_mask = train_mask.numpy().astype(np.bool)
#     labels = dataset['labels'][test_mask].cpu().numpy()
#     nodes_in_mask = torch.arange(len(test_mask))[test_mask].numpy()
#     table = {}
#     classes = np.unique(labels)
#     Overall = {
#         name: value[~train_mask] for name, value in Overall.items()
#     }
#     nodes_out_train = np.arange(len(test_mask))[~train_mask]
#     # print(classes)
#     # print(Overall['recall'].shape, Overall['precision'].shape)
#     test_labels = {}
#     logs = []
#     for label in classes:
#         pred_where = np.where(Overall['prediction'] == label)[0]
#         truth_where = np.where(labels == label)[0]
#         truth_where = nodes_in_mask[truth_where]
#         if plot:
#             logs.append(Overall[sortby][pred_where, label])
#         if sortby == 'recall':
#             recall = Overall['recall'][pred_where, label]
#             index = np.argsort(recall)[::-1]
#         elif sortby == 'precision':
#             precision = Overall['precision'][pred_where, label]
#             index = np.argsort(precision)[::-1]
#         elif sortby == 'f1':
#             recall = Overall['recall'][pred_where, label]
#             recall_max = np.max(recall) if len(recall)>0 else 1
#             recall = recall/recall_max
#             precision = Overall['precision'][pred_where, label]
#             f1 = (2*recall*precision/(recall + precision))
#             index = np.argsort(f1)[::-1]
#         # print(label, len(pred_where), len(truth_where),np.std(dataset.neighbours_sum().cpu().numpy()[truth_where]))
#         test_labels[label] = len(truth_where)
#         sorted_pred = pred_where[index]
#         sorted_pred = nodes_out_train[sorted_pred]
#         table[label] = (sorted_pred, truth_where)
#     if plot:
#         plot_curve(logs, classes)
#     return table, test_labels