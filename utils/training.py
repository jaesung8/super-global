import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import typing
import time

import numpy as np
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, Dataset
from torch.optim import SGD, Adam, AdamW, lr_scheduler
import matplotlib.pyplot as plt

from model.matcher import MatchERT
from data.dataset import FeatureDataset, TripletSampler

class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_one_epoch(
        model: nn.Module,
        loader: DataLoader,
        class_loss: nn.Module,
        optimizer: Optimizer,
        scheduler: typing.Tuple[_LRScheduler, bool], 
        epoch: int,
    ) -> None:
    model.train()
    device = next(model.parameters()).device
    to_device = lambda x: x.to(device, non_blocking=True)
    loader_length = len(loader)
    train_losses = AverageMeter()
    train_accs = AverageMeter()
    log_interval = loader_length // 10
    loss_list = []

    start_time = time.time()
    for i, global_feats in enumerate(loader):
        batch_start_time = time.time()
        global_feats = to_device(global_feats)
        p_logits = model(global_feats[0::3], global_feats[1::3])
        n_logits = model(global_feats[0::3], global_feats[2::3])
        logits = torch.cat([p_logits, n_logits], 0)
        bsize = logits.size(0)
        # assert (bsize % 2 == 0)
        labels = logits.new_ones(logits.size()).float()
        labels[(bsize//2):] = 0
        loss = class_loss(logits, labels).mean()
        acc = ((torch.sigmoid(logits) > 0.5).long() == labels.long()).float().mean()

        ##############################################
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()

        train_losses.update(loss.item(), 1)
        train_accs.update(acc, 1)
        

        if i and i % log_interval == 0:
            print(
                f'[Train] Epoch {epoch+1}  Batch {i+1}  '
                f'Loss: {train_losses.val:.5f} ({train_losses.avg:.5f})  ' 
                f'Accuracy {train_accs.val:.3f} ({train_accs.avg:.3f})  '
                f'Time: {time.time() - batch_start_time}s  '
                f'LR: {scheduler[0].get_last_lr()[0]}'
            )
            loss_list.append(train_losses.val)
    print(f'Epoch {epoch + 1} finished {time.time() - start_time}s elapsed')
    if not scheduler[-1]:
        scheduler[0].step()

    return loss_list


class BinaryCrossEntropyWithLogits(nn.Module):
    def __init__(self):
        super(BinaryCrossEntropyWithLogits, self).__init__()

    def forward(self, logits, labels):
        return F.binary_cross_entropy_with_logits(logits, labels, reduction='none')


def train():
    epochs = 100
    batch_size = 2000
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_sampler = TripletSampler(batch_size=batch_size)
    train_set = FeatureDataset()
    train_loader = DataLoader(train_set, batch_sampler=train_sampler, num_workers=8, pin_memory=True)

    model = MatchERT(device=device, d_global=2048, d_model=128, nhead=4, num_encoder_layers=6, 
            dim_feedforward=1024, dropout=0.0, activation='relu', normalize_before=False)
    # if resume is not None:
    #     checkpoint = torch.load(resume, map_location=torch.device('cpu'))
    #     model.load_state_dict(checkpoint['state'], strict=True)
    class_loss = BinaryCrossEntropyWithLogits()

    model.to(device)
    # model = nn.DataParallel(model)
    parameters = [{'params': model.parameters()}]

    loader_length = len(train_loader)
    optimizer = AdamW(parameters, lr=0.0001, weight_decay=0.0004)
    # scheduler = (lr_scheduler.MultiStepLR(optimizer, milestones=[16, 18], gamma=0.1), False)
    scheduler = (lr_scheduler.ExponentialLR(optimizer, gamma=0.96), False)
    # if resume is not None and checkpoint.get('optim', None) is not None:
    #     optimizer.load_state_dict(checkpoint['optim'])
    #     del checkpoint

    # setup partial function to simplify call
    # eval_function = partial(evaluate, model=model, 
    #     cache_nn_inds=cache_nn_inds,
    #     recall=recall_ks, query_loader=loaders.query, gallery_loader=loaders.gallery)

    # setup best validation logger
    # result = eval_function()
    # if callback is not None:
    #     callback.scalars(['l2', 'cosine'], 0, [metrics.recall['l2'][1], metrics.recall['cosine'][1]],
    #                      title='Val Recall@1')
    # pprint(result)
    import os
    from test.config_gnd import config_gnd
    from test.test_utils import test_revisitop

    data_dir = 'revisitop/data/datasets'
    dataset = 'roxford5k'

    q_path = os.path.join(data_dir, f'{dataset}_transformer_q.pt')
    x_path = os.path.join(data_dir, f'{dataset}_transformer_x.pt')
    Q = torch.load(q_path)
    X = torch.load(x_path)
    cfg = config_gnd(dataset,data_dir)
    gnd = cfg['gnd']
    nq = len(gnd)
    Q = torch.tensor(Q).to(device)
    X = torch.tensor(X).to(device)
    sim = torch.matmul(X[:, 1, :], Q[:, 1, :].T) # 6322 70
    ranks = torch.argsort(-sim, axis=0) # 6322 70

    def evaluate(model):
        total_sims = []
        total_eqs = []
        topk_rank_trans = torch.transpose(ranks,1,0)[:,:400] # 70 400
        for i in range(nq):
            rerank_X = X[topk_rank_trans[i]]
            sims = model(Q[i].expand_as(rerank_X), rerank_X)
            # eqs = (torch.sigmoid(sims) > 0.5).long()
            total_sims.append(sims)
            # total_eqs.append(eqs)

        total_sims = torch.stack(total_sims, dim=0)

        # reranks = torch.argsort(-(total_sims + torch.transpose(sim[:400],1,0)), axis=1)
        reranks = torch.argsort(-total_sims, axis=1)

        ranks_transpose = torch.transpose(ranks,1,0)[:,400:]
        rerank_final = []
        for i in range(nq):
            temp_concat = torch.concat([topk_rank_trans[i][reranks[i]], ranks_transpose[i]],0)
            rerank_final.append(temp_concat) # 6322
        ranks = torch.transpose(torch.stack(rerank_final,0),1,0)
        ranks = ranks.data.cpu().numpy()
        ks = [1, 5, 10]
        (mapE, _, _, _), (mapM, _, _, _), (mapH, _, _, _) = test_revisitop(cfg, ks, [ranks, ranks, ranks])

        return mapE, mapM, mapH

    train_loss_list = []
    max_mapH = 0
    for epoch in range(epochs):
        torch.cuda.empty_cache()
        loss_list = train_one_epoch(
            model=model,
            loader=train_loader,
            class_loss=class_loss,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
        )
        train_loss_list.extend(loss_list)
        mapE, mapM, mapH = evaluate(model)
        if mapH > max_mapH:
            max_mapH = mapH
            torch.save(model.state_dict(), 'transformer_model_2048.pt')
            print(f'Max mapH {np.around(mapH*100, decimals=2)}')
        # validation
        # torch.cuda.empty_cache()
        # result = eval_function()
        # print('Validation [{:03d}]'.format(epoch)), pprint(result)
        # ex.log_scalar('val.M_map', result['M_map'], step=epoch + 1)
        # ex.log_scalar('val.H_map', result['H_map'], step=epoch + 1)

        # if (result['M_map'] + result['H_map']) >= (best_val[1]['M_map'] + best_val[1]['H_map']):
        #     print('New best model in epoch %d.'%epoch)
        #     best_val = (epoch + 1, result, deepcopy(model.state_dict()))
        #     torch.save({'state': state_dict_to_cpu(best_val[2]), 'optim': optimizer.state_dict()}, save_name)


    plt.plot(train_loss_list)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig('train_loss.png')

    # logging
    # ex.info['metrics'] = best_val[1]
    # ex.add_artifact(save_name)

    # if callback is not None:
    #     save_name = os.path.join(temp_dir, 'visdom_data.pt')
    #     callback.save(save_name)
    #     ex.add_artifact(save_name)
