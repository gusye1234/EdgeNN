import world
import utils
import torch
import torch.nn.functional as F
from world import join, CONFIG
from utils import timer, Path3, set_seed
from data import loadAFL
from loss import CrossEntropy
from tabulate import tabulate
from tensorboardX import SummaryWriter


set_seed(2020)
#################################
# data
#################################
dataset = loadAFL(CONFIG['dataset'],
                  splitFile=f"{world.CONFIG['dataset']}_split_0.6_0.2_1.npz")
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


optim = torch.optim.Adam(MODEL.parameters(),
                         lr=CONFIG['lr'],
                         weight_decay=CONFIG['decay'])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optim,
                                                       factor=CONFIG['decay_factor'],
                                                       patience=CONFIG['decay_patience'])
LOSS = CrossEntropy()

#################################
# logger
#################################
if CONFIG['tensorboard']:
    logger = SummaryWriter(
        logdir=  Path3(world.LOG, 'runs', unique_name)
    )
earlystop = utils.EarlyStop(CONFIG['stop_patience'],
                            MODEL,
                            Path3(world.LOG, 'checkpoints', f"{unique_name}.pth.tar"))


(MODEL, dataset) = utils.TO(MODEL, dataset, device=world.DEVICE)
print(dataset)
print(utils.dict2table(CONFIG))

#################################
# main train loop
#################################
for epoch in range(1, CONFIG['epoch']+1):
    report = {}
    with timer():
        MODEL.train()
        probability = MODEL()
        # TODO: Loss function?
        loss = LOSS(probability[dataset['train mask']],
                    dataset['labels'][dataset['train mask']])
        optim.zero_grad()
        loss.backward()
        optim.step()

        with torch.no_grad():
            report['train loss'] = loss.item()
            report['valid loss'] = LOSS(probability[dataset['train mask']],
                                        dataset['labels'][dataset['train mask']]).item()
            MODEL.eval()
            # remove unaligned dim
            prediction = probability[:, :-1].argmax(dim=1)
            # TODO: Loss function?
            report['train acc'] = utils.accuracy(prediction,
                                                dataset['labels'],
                                                dataset['train mask'])
            report['valid acc'] = utils.accuracy(prediction,
                                                dataset['labels'],
                                                dataset['valid mask'])


    print(f"[{epoch}/{CONFIG['epoch']}] : {timer.get():.3f}|"
          f" T loss {report['train loss']:.4f}|"
          f" T acc {report['train acc']:.2f}|"
          f" V loss {report['valid loss']:.4f}|"
          f" V acc {report['valid acc']:.2f}", end='')
    scheduler.step(report['valid loss'])
    if earlystop.step(epoch, report, 'valid acc'):
        break
    print()

#################################
# Test
#################################
final_report = earlystop.best_result
with torch.no_grad():
    probability = MODEL()
    prediction = probability[:, :-1].argmax(dim=1)
    final_report['test acc'] = utils.accuracy(prediction,
                                              dataset['labels'],
                                              dataset['test mask'])
    final_report['test loss'] = LOSS(probability[dataset['test mask']],
                                     dataset['labels'][dataset['test mask']]).item()

try:
    handler = open(Path3(world.LOG, 'results', f"{unique_name}.txt"), 'a')
    info = f'''
    lr:{CONFIG['lr']}, decay:{CONFIG['decay']}, factor:{CONFIG['decay_factor']}, stop:{earlystop.best_epoch}/{CONFIG['epoch']}
    '''
    handler.write(info)
    handler.write(utils.dict2table(final_report, headers='firstrow'))
    handler.write('\n')
finally:
    handler.close()

if CONFIG['tensorboard']:
    logger.close()