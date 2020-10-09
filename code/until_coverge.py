import world
import utils
import torch
import torch.nn.functional as F
import numpy as np
from world import join, CONFIG
from utils import timer, Path3, set_seed
from data import loadAFL
from loss import CrossEntropy, EdgeLoss
from tabulate import tabulate
from tensorboardX import SummaryWriter

set_seed(2020)
#################################
# data
#################################
dataset = loadAFL(CONFIG['dataset'], )
#   splitFile=f"{world.CONFIG['dataset']}_split_0.6_0.2_1.npz")
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
LOSS_edge = EdgeLoss(graph=dataset, select=['edge'],**CONFIG)
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
print(dataset)
print(utils.dict2table(CONFIG))

#################################
# main train loop
#################################
for epoch in range(1, CONFIG['epoch'] + 1):
    report = {}
    with timer(name='total'):
        with timer(name='FL'):
            MODEL.train()
            probability = MODEL()
            loss_edge = LOSS_edge(probability, dataset['labels'], dataset['train mask'])
        with timer(name='B'):
            optim.zero_grad()
            loss_edge.backward()
            optim.step()
            report['edge_loss'] = loss_edge.item()
        with timer(name='FL'):
            MODEL.train()
            probability = MODEL()
            loss = LOSS_task(probability, dataset['labels'],dataset['train mask'])
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
                prediction = probability['poss_node'][:, :-1].argmax(dim=1)
                prediction_valid = probability_valid['poss_node'][:, :-1].argmax(dim=1)
                # TODO: Loss function?
                report['train acc'] = utils.accuracy(prediction,
                                                     dataset['labels'],
                                                     dataset['train mask'])
                report['valid acc'] = utils.accuracy(prediction_valid,
                                                     dataset['labels'],
                                                     dataset['valid mask'])
                report['test acc'] = utils.accuracy(prediction_valid,
                                                     dataset['labels'],
                                                     dataset['test mask'])

    print(
        f"[{epoch:4}/{CONFIG['epoch']}] : {timer.dict()}"
        f" E loss {report['edge_loss']:.3f}#"
        f" T loss {report['train loss']:.3f}#"
        f" T acc {report['train acc']:.2f}#"
        f" V loss {report['valid loss']:.3f}#"
        f" V acc {report['valid acc']:.2f}",
        f" Acc {report['test acc']:.2f}",
        end='')
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
    print()

#################################
# Test
#################################
final_report = earlystop.best_result
MODEL.load_state_dict(earlystop.best_model)
with torch.no_grad():
    MODEL.eval()
    probability = MODEL()
    prediction = probability['poss_node'][:, :-1].argmax(dim=1)
    final_report['test acc'] = utils.accuracy(prediction, dataset['labels'],
                                              dataset['test mask'])
    final_report['test loss'] = LOSS_task(probability, dataset['labels'],
                                     dataset['test mask']).item()
torch.save(earlystop.best_model, earlystop.filename)
try:
    handler = open(Path3(world.LOG, 'results', f"{unique_name}.txt"), 'a')
    info = f'''
    lr:{CONFIG['lr']}, decay:{CONFIG['decay']}, semi: {CONFIG['semi_lambda']}, edge: {CONFIG['edge_lambda']}, factor:{CONFIG['decay_factor']}, stop:{earlystop.best_epoch}/{CONFIG['epoch']}
    '''
    handler.write(info)
    handler.write(utils.dict2table(final_report, headers='firstrow'))
    handler.write('\n')
finally:
    handler.close()

if CONFIG['tensorboard']:
    logger.close()