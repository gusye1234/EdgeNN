import world
import utils
import torch
import torch.nn.functional as F
import numpy as np
from pprint import pprint
from world import join, CONFIG
from utils import timer, Path3, set_seed, table_info
from data import loadAFL
from loss import CrossEntropy, EdgeLoss
from tabulate import tabulate
from tensorboardX import SummaryWriter

seed = world.SEED
set_seed(seed)
top_k = world.TOPK
#################################
# data
#################################
dataset = loadAFL(CONFIG['dataset'], split=CONFIG['split'])
#   splitFile=f"{world.CONFIG['dataset']}_split_0.6_0.2_2.npz")
CONFIG['the number of nodes'] = dataset.num_nodes()
CONFIG['the number of classes'] = dataset.num_classes()
CONFIG['the dimension of features'] = dataset['features'].shape[1]
# CONFIG['comment'] = 'original-GCN'
unique_name = utils.uniqueFileFlag()

from model import GCN, GCN_single

net = GCN(CONFIG['the dimension of features'], CONFIG['gcn_hidden'],
            CONFIG['the number of classes'], CONFIG['dropout_rate'])


optimizer = torch.optim.Adam([{'params': net.gc1.parameters(), 'weight_decay': CONFIG['decay']},
                               {'params': net.gc2.parameters(), 'weight_decay': CONFIG['decay']}],lr=CONFIG['lr'])
learning_rate_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                                    factor=CONFIG['decay_factor'],
                                                                    patience=CONFIG['decay_patience'])
earlystop = utils.EarlyStop(CONFIG['stop_patience'],
                            net,
                            Path3(world.LOG, 'checkpoints', f"{unique_name}.pth.tar"))

(net, dataset) = utils.TO(net, dataset, device=world.DEVICE)

print(dataset)
print(utils.dict2table(CONFIG))

labels = dataset['labels']
test = False
if not test:
    for epoch in range(1, CONFIG['epoch']+1):
        # report = {}
        with timer(name='total'):
            net.train()
            train_logits = net(dataset['features'], dataset.adj)
            train_logp = F.log_softmax(train_logits, 1)
            train_loss = F.nll_loss(train_logp[dataset['train mask']], labels[dataset['train mask']])
            train_pred = train_logp.argmax(dim=1)
            train_acc = torch.eq(train_pred[dataset['train mask']], labels[dataset['train mask']]).float().mean().item()

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            net.eval()
            with torch.no_grad():
                val_logits = net(dataset['features'], dataset.adj)
                val_logp = F.log_softmax(val_logits, 1)
                val_loss = F.nll_loss(val_logp[dataset['valid mask']], labels[dataset['valid mask']]).item()
                val_pred = val_logp.argmax(dim=1)
                val_acc = utils.accuracy(val_pred, dataset['labels'], dataset['valid mask'])
                tes_acc = utils.accuracy(val_pred, dataset['labels'], dataset['test mask'])
                # val_acc = torch.eq(val_pred[dataset['valid mask']], labels[dataset['valid mask']]).float().mean().item()

                # tes_acc = torch.eq(val_pred[dataset['test mask']],labels[dataset['test mask']]).float().mean().item()

            report = {
                'train loss': train_loss.item(),
                'train acc' : train_acc,
                'valid loss': val_loss,
                'valid acc' : val_acc
            }
        if not CONFIG['quite']:
            print(f"[{epoch:4}/{CONFIG['epoch']}] : {timer.dict()}"
                f" T loss {report['train loss']:.3f}#"
                f" T acc {report['train acc']:.2f}#"
                f" V loss {report['valid loss']:.3f}#"
                f" V acc {report['valid acc']:.2f}",
                f" E acc {tes_acc}",end='')
        else:
            print(f"[{epoch:4}/{CONFIG['epoch']}] : {timer.dict()}", end='')
        learning_rate_scheduler.step(val_loss)
        if earlystop.step(epoch, report, 'valid acc'):
            break
        print("\r", end='') if CONFIG['quite'] else print()

    #################################
    # Test
    #################################
    torch.save(earlystop.best_model, earlystop.filename)
    print("Save model to", earlystop.filename)
    final_report = earlystop.best_result
    net.load_state_dict(earlystop.best_model)
else:
    final_report = {"test":True}
    print("Test model of",
          Path3(world.LOG, 'checkpoints', f"{unique_name}.pth.tar"))
    net.load_state_dict(
        torch.load(Path3(world.LOG, 'checkpoints', f"{unique_name}.pth.tar")))
with torch.no_grad():
    test_logits = net(dataset['features'], dataset.adj)
    test_logp = F.log_softmax(test_logits, 1)
    precision = F.softmax(test_logits, 1)
    test_loss = F.nll_loss(test_logp[dataset['test mask']], labels[dataset['test mask']]).item()
    test_pred = test_logp.argmax(dim=1)
    test_acc = torch.eq(test_pred[dataset['test mask']], labels[dataset['test mask']]).float().mean().item()
    final_report['test acc'] = test_acc

pprint(final_report)

Overall = {
    "prediction": test_pred.cpu().numpy(),
    "precision": precision.cpu().numpy()
}
import rich
print()
train_mask = dataset['train mask'].cpu()
test_mask = dataset['test mask'].cpu()
test_total = torch.sum(test_mask).item()

for method in ['precision']:
    rich.print(f"[bold yellow]{method}[/bold yellow]")
    rank_table, test_labels = utils.Group_ByPrediction_mask_all(
        Overall, dataset, sortby=method, train_mask=train_mask, test_mask=test_mask, plot=True
    )
    recall, precision, NDCG = utils.topk_metrics(rank_table, top_k)
    pprint(recall)
    pprint(precision)
    pprint(NDCG)
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
pprint(pred_dict)
#################################
# Log
#################################
try:
    handler = open(Path3(world.LOG, 'results', f"{unique_name}.txt"), 'a')
    info = table_info(earlystop.best_epoch, seed)
    handler.write(info)
    handler.write(utils.dict2table(final_report, headers='firstrow'))
    handler.write('\n')
except:
    print("Not able to write", Path3(world.LOG, 'results',
                                     f"{unique_name}.txt"))
finally:
    handler.close()