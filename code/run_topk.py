import world
import utils
import torch
import torch.nn.functional as F
import numpy as np
from pprint import pprint
from world import join, CONFIG
from utils import timer, Path3, set_seed, table_info
from data import loadAFL, Graph
from loss import CrossEntropy, EdgeLoss
from tabulate import tabulate
from tensorboardX import SummaryWriter

seed = world.SEED
set_seed(seed)

top_k=world.TOPK
#################################
# data
#################################

'''A:adj, F: feature, L:label'''
dataset : Graph = loadAFL(CONFIG['dataset'], split=CONFIG['split'])
#   splitFile=f"{world.CONFIG['dataset']}_split_0.6_0.2_1.npz")
# dataset.splitByEdge(0.45)
print(dataset)

CONFIG['the number of nodes'] = dataset.num_nodes()
CONFIG['the number of classes'] = dataset.num_classes()
CONFIG['the dimension of features'] = dataset['features'].shape[1]

unique_name = utils.uniqueFileFlag()
#################################
# model and loss
#################################

if CONFIG['model'] == 'embedding':
    from model import EmbeddingP
    MODEL = EmbeddingP(CONFIG, dataset)
elif CONFIG['model'] == 'multi_embedding':
    from model import EmbeddingP_multiLayer
    MODEL = EmbeddingP_multiLayer(CONFIG, dataset)
elif CONFIG['model'] == 'gcn':
    from model import GCNP
    MODEL = GCNP(CONFIG, dataset)

# print([name for name, para in list(MODEL.named_parameters())])

optim = torch.optim.Adam(MODEL.parameters(),
                         lr=CONFIG['lr'],
                         weight_decay=CONFIG['decay'])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optim,
    factor=CONFIG['decay_factor'],
    patience=CONFIG['decay_patience'])
# LOSS = CrossEntropy()
LOSS_edge = EdgeLoss(graph=dataset, select=['edge'], **CONFIG)
LOSS_task = EdgeLoss(graph=dataset, select=['task', 'semi'], **CONFIG)

#################################
# logger
#################################
if CONFIG['tensorboard']:
    logger = SummaryWriter(logdir=Path3(world.LOG, 'runs', unique_name))
earlystop = utils.EarlyStop(
    CONFIG['stop_patience'], MODEL,
    Path3(world.LOG, 'checkpoints', f"{unique_name}.pth.tar"))

(MODEL, dataset) = utils.TO(MODEL, dataset, device=world.DEVICE)
print(utils.dict2table(CONFIG))

#################################
# main training loop
#################################
test=False
if not test:
    for epoch in range(1, CONFIG['epoch'] + 1):
        report = {}
        with timer(name='total'):
            with timer(name='FL'):
                MODEL.train()
                probability = MODEL()
                loss_edge = LOSS_edge(probability, dataset['labels'],
                                    dataset['train mask'])
            with timer(name='B'):
                optim.zero_grad()
                loss_edge.backward()
                optim.step()
                report['edge_loss'] = loss_edge.item()
            for i in range(world.PERTRAIN):
                with timer(name='FL'):
                    MODEL.train()
                    probability = MODEL()
                    loss = LOSS_task(probability, dataset['labels'],
                                    dataset['train mask'])
                with timer(name='B'):
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
            # print(loss_edge.item(), loss.item())
            with torch.no_grad():
                MODEL.eval()
                report['train loss'] = loss.item()
                with timer(name='F'):
                    probability_valid = MODEL()
                # probability_valid = probability
                # print(probability['poss_node'][:5])
                with timer(name='L'):
                    report['valid loss'] = LOSS_task(probability_valid,
                                                    dataset['labels'],
                                                    dataset['valid mask']).item()

                # remove unaligned dim
                with timer(name='M'):
                    # unlabeled = (~dataset['train mask'])
                    # values, args = torch.topk(torch.max(probability['poss_node'][unlabeled][:, :-1], dim=1)[0], 10)
                    # print(values)
                    # print(prediction[unlabeled][args])
                    # print(dataset['labels'][unlabeled][args])
                    prediction = probability['poss_node'][:, :-1].argmax(dim=1)
                    prediction_valid = probability_valid[
                        'poss_node'][:, :-1].argmax(dim=1)
                    report['train acc'] = utils.accuracy(prediction,
                                                        dataset['labels'],
                                                        dataset['train mask'])
                    report['valid acc'] = utils.accuracy(prediction_valid,
                                                        dataset['labels'],
                                                        dataset['valid mask'])
                    report['test acc'] = utils.accuracy(prediction_valid,
                                                        dataset['labels'],
                                                        dataset['test mask'])

        if not CONFIG['quite']:
            print(
                # f"[{epoch:4}/{CONFIG['epoch']}] : {timer.dict()}"
                f"[{epoch:4}/{CONFIG['epoch']}] : "
                f" E loss {report['edge_loss']:.3f}#"
                f" T loss {report['train loss']:.3f}#"
                f" T acc {report['train acc']:.2f}#"
                f" V loss {report['valid loss']:.3f}#"
                f" V acc {report['valid acc']:.2f}",
                f" Acc {report['test acc']:.2f}",
                end='')
        else:
            print(f"[{epoch:4}/{CONFIG['epoch']}] : ", end='')
        timer.zero()
        if CONFIG['tensorboard']:
            logger.add_scalar('train loss', report['train loss'], epoch)
            logger.add_scalar('valid loss', report['valid loss'], epoch)
            logger.add_scalar('valid acc', report['valid acc'], epoch)
            logger.add_scalar('train acc', report['train acc'], epoch)

        if np.isnan(report['train loss']) or np.isnan(report['valid loss']):
            # import ipdb
            # ipdb.set_trace()
            exit()
        scheduler.step(report['valid loss'])
        if earlystop.step(epoch, report, 'valid acc'):
            break
        print("\r", end='') if CONFIG['quite'] else print()

    #################################
    # Test
    #################################
    final_report = earlystop.best_result
    MODEL.load_state_dict(earlystop.best_model)
    torch.save(earlystop.best_model, earlystop.filename)
else:
    final_report = {"test":True}
    MODEL.load_state_dict(
        torch.load(Path3(world.LOG, 'checkpoints', f"{unique_name}.pth.tar")))

with torch.no_grad():
    MODEL.eval()
    probability = MODEL()
    prediction = probability['poss_node'][:, :-1].argmax(dim=1)
    final_report['test acc'] = utils.accuracy(prediction, dataset['labels'],
                                              dataset['test mask'])
    final_report['test loss'] = LOSS_task(probability, dataset['labels'],
                                          dataset['test mask']).item()
    Overall = {
        "prediction": prediction.cpu().numpy(),
        "recall": probability['recall_node'].cpu().numpy(),
        "precision": probability['poss_node'].cpu().numpy()
    }
pprint(final_report)
np.savetxt("edge_cora.txt", Overall['precision'])

pprint(utils.peak(dataset, Overall['precision']))
pprint(utils.peak(dataset, Overall['recall']))
#################################
# Rank
#################################
import rich
print()
# rank_table = utils.Group_ByPrediction(Overall, dataset, sortby='recall')
train_mask = dataset['train mask'].cpu()
test_mask = dataset['test mask'].cpu()
test_total = float(torch.sum(test_mask).item())
print("total test: ", test_total)

# recall, precision, NDCG = utils.topk_metrics(rank_table, top_k)
recall_all = {'recall':[], 'precision':[], 'f1':[]}
precision_all = {'recall': [], 'precision': [], 'f1': []}
ndcg_all = {'recall': [], 'precision': [], 'f1': []}

for method in ['recall', 'precision', 'f1']:
    rich.print(f"[bold yellow]{method}[/bold yellow]")
    rank_table, test_labels = utils.Group_ByPrediction_mask_all(Overall,
                                                            dataset,
                                                            sortby=method,
                                                            train_mask=train_mask,
                                                            test_mask=test_mask,)
    recall, precision, NDCG = utils.topk_metrics(rank_table, top_k)
    rich.print("[bold green]   topk STAT[/bold green]")

    pred_dict = {
        "recall":
        np.sum([value * test_labels[index]
                for index, value in recall.items()]) / test_total,
        "precision":
        np.sum(
            [value * test_labels[index]
             for index, value in precision.items()]) / test_total,
        "NDCG":
        np.sum([value * test_labels[index]
                for index, value in NDCG.items()]) / test_total
    }
    print(utils.dict2table(pred_dict, headers='firstrow'))
    for label in range(len(NDCG)):
        recall_all[method].append(recall[label])
        precision_all[method].append(precision[label])
        ndcg_all[method].append(NDCG[label])
# print(recall_all)
# print(precision_all)
# print(ndcg_all)
#################################
# Log
#################################
try:
    handler = open(Path3(world.LOG, 'results', f"{unique_name}.txt"), 'a')
    info = table_info(earlystop.best_epoch, seed)
    handler.write(info)
    handler.write(utils.dict2table(final_report, headers='firstrow'))
    handler.write('\n')
finally:
    handler.close()

if CONFIG['tensorboard']:
    logger.close()