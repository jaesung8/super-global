import numpy as np
import bisect, torch
import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
import random
import pickle as pkl


class FeatureDataset(Dataset):
    def __init__(self):
        # [124774, 3, 2048]
        self.train_feature = torch.load('gldv2/global_train.pt').cpu()
        self.sample_num = self.train_feature.size(0)

    def __len__(self):
        return self.sample_num * 50

    def __getitem__(self, index):
        return self.train_feature[index % self.sample_num]


class TripletSampler():
    def __init__(self, batch_size):
        self.batch_size = batch_size
        with open('train_pn.pkl', 'rb') as f:
            self.train_pn = pkl.load(f)

        self.num_samples = len(self.train_pn)

    def __iter__(self):
        batch = []
        cands = torch.randperm(self.num_samples).tolist()
        for i in range(50):
            for anchor_idx in cands:
                anchor_dict = self.train_pn[anchor_idx]

                positive_inds = anchor_dict['pos']
                negative_inds = anchor_dict['neg']
                pos_index = random.randint(0, len(positive_inds)-1)
                neg_index = random.randint(0, len(negative_inds)-1)

                batch.append(anchor_idx)
                batch.append(positive_inds[pos_index]) 
                batch.append(negative_inds[neg_index])

                if len(batch) >= self.batch_size:
                    yield batch
                    batch = []
                
        if len(batch) > 0:
            yield batch

    def __len__(self):
        return (self.num_samples * 3 * 50 + self.batch_size - 1) // self.batch_size
