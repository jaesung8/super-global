# written by Seongwon Lee (won4113@yonsei.ac.kr)

import os
import time
import pprint

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from test.config_gnd import config_gnd
from test.test_utils import extract_feature, test_revisitop, rerank_ranks_revisitop, extract_cvnet_feature, extract_feature_list
from test.dataset import DataSet

from modules.reranking.MDescAug import MDescAug
from modules.reranking.RerankwMDA import RerankwMDA
from model.matcher import MatchERT


@torch.no_grad()
def test_model(model, data_dir, dataset_list, scale_list, is_rerank, gemp, rgem, sgem, onemeval, depth, logger):
    torch.backends.cudnn.benchmark = False
    model.eval()
    
    state_dict = model.state_dict()
    

    # initialize modules
    MDescAug_obj = MDescAug()
    RerankwMDA_obj = RerankwMDA()



    model.load_state_dict(state_dict)
    for dataset in dataset_list:
        text = '>> {}: Global Retrieval for scale {} with CVNet-Global'.format(dataset, str(scale_list))
        print(text)
        if dataset == 'roxford5k':
            gnd_fn = 'gnd_roxford5k.pkl'
        elif dataset == 'rparis6k':
            gnd_fn = 'gnd_rparis6k.pkl'
        else:
            assert dataset

        q_path = os.path.join(data_dir, f'{dataset}_q.pt')
        x_path = os.path.join(data_dir, f'{dataset}_x.pt')

        if (
            os.path.isfile(q_path)
            and os.path.isfile(x_path)
        ):
            print("load query and db features")
            Q = torch.load(q_path)
            X = torch.load(x_path)
        else:
            print("extract query features")
            Q = extract_feature(model, data_dir, dataset, gnd_fn, "query", [1.0], gemp, rgem, sgem, scale_list)
            print("extract database features")
            X = extract_feature(model, data_dir, dataset, gnd_fn, "db", [1.0], gemp, rgem, sgem, scale_list)

            torch.save(Q, q_path)
            torch.save(X, x_path)

        cfg = config_gnd(dataset,data_dir)
        Q = torch.tensor(Q).cuda()
        X = torch.tensor(X).cuda()
        
        print("perform global feature reranking")
        if onemeval:
            X_expand = torch.load(f"./feats_1m_RN{depth}.pth").cuda()
            X = torch.concat([X,X_expand],0)
        sim = torch.matmul(X, Q.T) # 6322 70
        ranks = torch.argsort(-sim, axis=0) # 6322 70

        plt_ranks = ranks.clone().cpu().numpy()

        gnd = cfg['gnd']
        nq = len(gnd)
        retrieval = []

        for i in range(nq):
            qgnd = np.concatenate([gnd[i]['easy'], gnd[i]['hard']])
            # no positive images, skip from the average
            if qgnd.shape[0] == 0:
                continue
            try:
                qgndj = np.array(gnd[i]['junk'])
            except:
                qgndj = np.empty(0)

            junk = np.arange(plt_ranks.shape[0])[np.in1d(plt_ranks[:,i], qgndj)]
            cur_rank = np.delete(plt_ranks[:, i], junk)
            pos = np.in1d(cur_rank, qgnd)
            retrieval.append(np.concatenate([pos, np.zeros(len(junk), dtype=bool)]))

        retrieval = np.stack(retrieval)
        print(retrieval.shape)
        retrieval = retrieval[:, :400]
        print(
            retrieval[:, :25].sum(), retrieval[:, :50].sum(), retrieval[:, :75].sum(),
            retrieval[:, :100].sum(), retrieval[:, :200].sum(), retrieval[:, :300].sum(), retrieval.sum()
        )
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # ax.plot(retrieval)
        plt.imshow(retrieval[:, :100], aspect='auto', cmap=plt.cm.gray, interpolation='nearest')
        plt.savefig('retrieval.png')
        check_ranks = ranks.clone().data.cpu().numpy()
        ks = [1, 5, 10]
        (mapE, _, _, _), (mapM, _, _, _), (mapH, _, _, _) = test_revisitop(cfg, ks, [check_ranks, check_ranks, check_ranks])

        print('Retrieval results: mAP E: {}, M: {}, H: {}'.format(np.around(mapE*100, decimals=2), np.around(mapM*100, decimals=2), np.around(mapH*100, decimals=2)))

        if is_rerank:
            rerank_dba_final, res_top1000_dba, ranks_trans_1000_pre, x_dba = MDescAug_obj(X, Q, ranks)
            ranks = RerankwMDA_obj(ranks, rerank_dba_final, res_top1000_dba, ranks_trans_1000_pre, x_dba)
        ranks = ranks.data.cpu().numpy()

        plt_ranks = ranks.copy()
        gnd = cfg['gnd']
        nq = len(gnd)
        rerank = []
        for i in range(nq):
            qgnd = np.concatenate([gnd[i]['easy'], gnd[i]['hard']])
            # no positive images, skip from the average
            if qgnd.shape[0] == 0:
                continue
            try:
                qgndj = np.array(gnd[i]['junk'])
            except:
                qgndj = np.empty(0)

            junk = np.arange(plt_ranks.shape[0])[np.in1d(plt_ranks[:,i], qgndj)]
            cur_rank = np.delete(plt_ranks[:, i], junk)
            pos = np.in1d(cur_rank, qgnd)
            rerank.append(np.concatenate([pos, np.zeros(len(junk), dtype=bool)]))

        rerank = np.stack(rerank)
        rerank = rerank[:, :400]
        print(
            rerank[:, :25].sum(), rerank[:, :50].sum(), rerank[:, :75].sum(),
            rerank[:, :100].sum(), rerank[:, :200].sum(), rerank[:, :300].sum(), rerank.sum())
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.imshow(rerank[:, :100], aspect='auto', cmap=plt.cm.gray, interpolation='nearest')
        plt.savefig('rerank.png')


        # revisited evaluation
        ks = [1, 5, 10]
        (mapE, _, _, _), (mapM, _, _, _), (mapH, _, _, _) = test_revisitop(cfg, ks, [ranks, ranks, ranks])

        print('Reranking results: mAP E: {}, M: {}, H: {}'.format(np.around(mapE*100, decimals=2), np.around(mapM*100, decimals=2), np.around(mapH*100, decimals=2)))
        # logger.info('Retrieval results: mAP E: {}, M: {}, H: {}'.format(np.around(mapE*100, decimals=2), np.around(mapM*100, decimals=2), np.around(mapH*100, decimals=2)))


@torch.no_grad()
def test_cvnet(model, data_dir, dataset_list, scale_list, topk_list):
    torch.backends.cudnn.benchmark = False
    model.eval()
    for dataset in dataset_list:
        text = '>> {}: Global Retrieval for scale {} with CVNet-Global'.format(dataset, str(scale_list))
        print(text)
        if dataset == 'roxford5k':
            gnd_fn = 'gnd_roxford5k.pkl'
        elif dataset == 'rparis6k':
            gnd_fn = 'gnd_rparis6k.pkl'
        else:
            assert dataset

        scale_list = [0.7071, 1.0, 1.4142]
        q_path = os.path.join(data_dir, f'{dataset}_cv_q.pt')
        x_path = os.path.join(data_dir, f'{dataset}_cv_x.pt')

        if (
            os.path.isfile(q_path)
            and os.path.isfile(x_path)
        ):
            print("load query and db features")
            Q = torch.load(q_path)
            X = torch.load(x_path)
        else:
            print("extract query features")
            Q = extract_cvnet_feature(model, data_dir, dataset, gnd_fn, "query", scale_list)
            print("extract database features")
            X = extract_cvnet_feature(model, data_dir, dataset, gnd_fn, "db", scale_list)

            torch.save(Q, q_path)
            torch.save(X, x_path)

        cfg = config_gnd(dataset,data_dir)
        # Q = torch.tensor(Q).cuda()
        # X = torch.tensor(X).cuda()

        # print("extract query features")
        # Q = extract_feature(model, data_dir, dataset, gnd_fn, "query", scale_list)
        # print("extract database features")
        # X = extract_feature(model, data_dir, dataset, gnd_fn, "db", scale_list)

        cfg = config_gnd(dataset,data_dir)

        # perform search
        print("perform global retrieval")
        sim = np.dot(X, Q.T)
        ranks = np.argsort(-sim, axis=0)

        plt_ranks = ranks.copy()
        gnd = cfg['gnd']
        nq = len(gnd)
        retrieval = []

        for i in range(nq):
            qgnd = np.concatenate([gnd[i]['easy'], gnd[i]['hard']])
            # no positive images, skip from the average
            if qgnd.shape[0] == 0:
                continue
            try:
                qgndj = np.array(gnd[i]['junk'])
            except:
                qgndj = np.empty(0)

            junk = np.arange(plt_ranks.shape[0])[np.in1d(plt_ranks[:,i], qgndj)]
            cur_rank = np.delete(plt_ranks[:, i], junk)
            pos = np.in1d(cur_rank, qgnd)
            retrieval.append(np.concatenate([pos, np.zeros(len(junk), dtype=bool)]))

        retrieval = np.stack(retrieval)
        retrieval = retrieval[:, :400]
        print(
            retrieval[:, :25].sum(), retrieval[:, :50].sum(), retrieval[:, :75].sum(),
            retrieval[:, :100].sum(), retrieval[:, :200].sum(), retrieval[:, :300].sum(), retrieval.sum()
        )
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # ax.plot(retrieval)
        plt.imshow(retrieval[:, :100], aspect='auto', cmap=plt.cm.gray, interpolation='nearest')
        plt.savefig('retrieval_cv.png')

        # revisited evaluation
        gnd = cfg['gnd']
        ks = [1, 5, 10]
        (mapE, apsE, mprE, prsE), (mapM, apsM, mprM, prsM), (mapH, apsH, mprH, prsH) = test_revisitop(cfg, ks, [ranks, ranks, ranks])

        print('Global retrieval results: mAP E: {}, M: {}, H: {}'.format(np.around(mapE*100, decimals=2), np.around(mapM*100, decimals=2), np.around(mapH*100, decimals=2)))

        print('>> {}: Reranking results with CVNet-Rerank'.format(dataset))

        query_dataset = DataSet(data_dir, dataset, gnd_fn, "query", [1.0])
        db_dataset = DataSet(data_dir, dataset, gnd_fn, "db", [1.0])
        sim_corr_dict = {}
        for topk in topk_list:
            print("current top-k value: ", topk)
            # for i in tqdm(range(int(cfg['nq']))):
            #     im_q = query_dataset.__getitem__(i)[0]
            #     im_q = torch.from_numpy(im_q).cuda().unsqueeze(0)
            #     feat_q = model.extract_featuremap(im_q)

            #     rerank_count = np.zeros(3, dtype=np.uint16)
            #     for j in range(int(cfg['n'])):
            #         if (rerank_count >= topk).sum() == 3:
            #             break

            #         rank_j = ranks[j][i]

            #         if rank_j in gnd[i]['junk']:
            #             continue
            #         elif rank_j in gnd[i]['easy']:
            #             append_j = np.asarray([True, True, False])
            #         elif rank_j in gnd[i]['hard']:
            #             append_j = np.asarray([False, True, True])
            #         else: #negative
            #             append_j = np.asarray([True, True, True])

            #         append_j *= (rerank_count < topk)

            #         if append_j.sum() > 0:
            #             im_k = db_dataset.__getitem__(rank_j)[0]
            #             im_k = torch.from_numpy(im_k).cuda().unsqueeze(0)
            #             feat_k = model.extract_featuremap(im_k)

            #             score = model.extract_score_with_featuremap(feat_q, feat_k).cpu()
            #             sim_corr_dict[(rank_j, i)] = score
            #             rerank_count += append_j
    
            # mix_ratio = 0.5
            # ranks_corr_list = rerank_ranks_revisitop(cfg, topk, ranks, sim, sim_corr_dict, mix_ratio)
            # np.save('ranks_corr_list.npy', ranks_corr_list)
            ranks_corr_list = np.array(np.load('ranks_corr_list.npy'))
            plt_ranks = ranks_corr_list.copy()[1, :, :]
            gnd = cfg['gnd']
            nq = len(gnd)
            rerank = []
            for i in range(nq):
                qgnd = np.concatenate([gnd[i]['easy'], gnd[i]['hard']])
                # no positive images, skip from the average
                if qgnd.shape[0] == 0:
                    continue
                try:
                    qgndj = np.array(gnd[i]['junk'])
                except:
                    qgndj = np.empty(0)

                junk = np.arange(plt_ranks.shape[0])[np.in1d(plt_ranks[:,i], qgndj)]
                cur_rank = np.delete(plt_ranks[:, i], junk)
                pos = np.in1d(cur_rank, qgnd)
                rerank.append(np.concatenate([pos, np.zeros(len(junk), dtype=bool)]))

            rerank = np.stack(rerank)
            rerank = rerank[:, :400]
            print(
                rerank[:, :25].sum(), rerank[:, :50].sum(), rerank[:, :75].sum(),
                rerank[:, :100].sum(), rerank[:, :200].sum(), rerank[:, :300].sum(), rerank.sum())
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plt.imshow(rerank[:, :100], aspect='auto', cmap=plt.cm.gray, interpolation='nearest')
            plt.savefig('rerank_cv.png')


            (mapE_r, apsE_r, mprE_r, prsE_r), (mapM_r, apsM_r, mprM_r, prsM_r), (mapH_r, apsH_r, mprH_r, prsH_r) = test_revisitop(cfg, ks, ranks_corr_list)
            print('Reranking results: mAP E: {}, M: {}, H: {}'.format(np.around(mapE_r*100, decimals=2), np.around(mapM_r*100, decimals=2), np.around(mapH_r*100, decimals=2)))
        
    torch.backends.cudnn.benchmark = True


@torch.no_grad()
def test_transformer_model(model, data_dir, dataset_list, scale_list, gemp, rgem, sgem):
    torch.backends.cudnn.benchmark = False
    model.eval()
    state_dict = model.state_dict()
    model.load_state_dict(state_dict)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    rerank_model = MatchERT(device=device, d_global=2048, d_model=128, nhead=4, num_encoder_layers=6, 
            dim_feedforward=1024, dropout=0.0, activation='relu', normalize_before=False).to(device)
    rerank_model = nn.DataParallel(rerank_model)
    rerank_model.load_state_dict(torch.load('trans_check.pt'))
    rerank_model.eval()

    for dataset in dataset_list:
        text = '>> {}: Global Retrieval for scale {} with CVNet-Global'.format(dataset, str(scale_list))
        print(text)
        if dataset == 'roxford5k':
            gnd_fn = 'gnd_roxford5k.pkl'
        elif dataset == 'rparis6k':
            gnd_fn = 'gnd_rparis6k.pkl'
        else:
            assert dataset

        q_path = os.path.join(data_dir, f'{dataset}_transformer_q.pt')
        x_path = os.path.join(data_dir, f'{dataset}_transformer_x.pt')

        if (
            os.path.isfile(q_path)
            and os.path.isfile(x_path)
        ):
            print("load query and db features")
            Q = torch.load(q_path)
            X = torch.load(x_path)
        else:
            print("extract query features")
            Q = extract_feature_list(model, data_dir, dataset, gnd_fn, "query", [1.0], gemp, rgem, sgem)
            print("extract database features")
            X = extract_feature_list(model, data_dir, dataset, gnd_fn, "db", [1.0], gemp, rgem, sgem)

            torch.save(Q, q_path)
            torch.save(X, x_path)

        cfg = config_gnd(dataset,data_dir)
        Q = torch.tensor(Q).cuda()
        X = torch.tensor(X).cuda()
        
        print("perform global feature reranking")

        sim = torch.matmul(X[:, 1, :], Q[:, 1, :].T) # 6322 70
        ranks = torch.argsort(-sim, axis=0) # 6322 70

        plt_ranks = ranks.clone().cpu().numpy()

        ks = [1, 5, 10]
        (mapE, _, _, _), (mapM, _, _, _), (mapH, _, _, _) = test_revisitop(cfg, ks, [plt_ranks, plt_ranks, plt_ranks])

        print('Retrieval results: mAP E: {}, M: {}, H: {}'.format(np.around(mapE*100, decimals=2), np.around(mapM*100, decimals=2), np.around(mapH*100, decimals=2)))


        gnd = cfg['gnd']
        nq = len(gnd)
        retrieval = []

        for i in range(nq):
            qgnd = np.concatenate([gnd[i]['easy'], gnd[i]['hard']])
            # no positive images, skip from the average
            if qgnd.shape[0] == 0:
                continue
            try:
                qgndj = np.array(gnd[i]['junk'])
            except:
                qgndj = np.empty(0)

            junk = np.arange(plt_ranks.shape[0])[np.in1d(plt_ranks[:,i], qgndj)]
            cur_rank = np.delete(plt_ranks[:, i], junk)
            pos = np.in1d(cur_rank, qgnd)
            retrieval.append(np.concatenate([pos, np.zeros(len(junk), dtype=bool)]))

        retrieval = np.stack(retrieval)
        print(retrieval.shape)
        retrieval = retrieval[:, :400]
        print(retrieval[:, :100].sum(), retrieval[:, :200].sum(), retrieval[:, :300].sum(), retrieval.sum())
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # ax.plot(retrieval)
        plt.imshow(retrieval, aspect='auto', cmap=plt.cm.gray, interpolation='nearest')
        plt.savefig('retrieval_transformer.png')


        # rerank
        total_sims = []
        total_eqs = []
        topk_rank_trans = torch.transpose(ranks,1,0)[:,:400] # 70 400
        for i in range(nq):
            rerank_X = X[topk_rank_trans[i]]
            sims = rerank_model(Q[i].expand_as(rerank_X), rerank_X)
            eqs = (torch.sigmoid(sims) > 0.5).long()
            total_sims.append(sims)
            total_eqs.append(eqs)

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

        plt_ranks = ranks.copy()
        gnd = cfg['gnd']
        nq = len(gnd)
        rerank = []
        for i in range(nq):
            qgnd = np.concatenate([gnd[i]['easy'], gnd[i]['hard']])
            # no positive images, skip from the average
            if qgnd.shape[0] == 0:
                continue
            try:
                qgndj = np.array(gnd[i]['junk'])
            except:
                qgndj = np.empty(0)

            junk = np.arange(plt_ranks.shape[0])[np.in1d(plt_ranks[:,i], qgndj)]
            cur_rank = np.delete(plt_ranks[:, i], junk)
            pos = np.in1d(cur_rank, qgnd)
            rerank.append(np.concatenate([pos, np.zeros(len(junk), dtype=bool)]))

        rerank = np.stack(rerank)
        rerank = rerank[:, :400]
        print(rerank[:, :100].sum(), rerank[:, :200].sum(), rerank[:, :300].sum(), rerank.sum())
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.imshow(rerank, aspect='auto', cmap=plt.cm.gray, interpolation='nearest')
        plt.savefig('rerank_transformer.png')


        # revisited evaluation
        ks = [1, 5, 10]
        (mapE, _, _, _), (mapM, _, _, _), (mapH, _, _, _) = test_revisitop(cfg, ks, [ranks, ranks, ranks])

        print('Rerank results: mAP E: {}, M: {}, H: {}'.format(np.around(mapE*100, decimals=2), np.around(mapM*100, decimals=2), np.around(mapH*100, decimals=2)))
