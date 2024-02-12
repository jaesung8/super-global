import os
import torch
import numpy as np

from tqdm import tqdm
import time
import pprint
import pickle as pkl

from PIL import Image
import torch
import matplotlib.pyplot as plt

from test.config_gnd import config_gnd
from test.test_utils import extract_feature, test_revisitop, rerank_ranks_revisitop, extract_cvnet_feature
from test.dataset import DataSet
# import core.transforms as transforms
from torchvision import transforms

from modules.reranking.MDescAug import MDescAug
from modules.reranking.RerankwMDA import RerankwMDA


_MEAN = [0.406, 0.456, 0.485]
_SD = [0.225, 0.224, 0.229]

class DataSet(torch.utils.data.Dataset):
    """Common dataset."""

    def __init__(self):
        super(DataSet, self).__init__()
        self.base_path = 'gldv2'
        with open(os.path.join(self.base_path, 'train_set.pkl'), 'rb') as f:
            self.labels = pkl.load(f)

        print(self.labels)

        self.transform = transforms.Compose(
                [
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _prepare_im(self, im):
        """Prepares the image for network input."""
        # im = im.transpose([2, 0, 1])
        # # [0, 255] -> [0, 1]
        # im = im / 255.0
        # # Color normalization
        # im = transforms.color_norm(im, _MEAN, _SD)
        return self.transform(im)

    def __getitem__(self, index):
        # Load the image
        img = Image.open(os.path.join(self.base_path, self.labels[index]["path"]))
        # im_np = im.astype(np.float32, copy=False)
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.labels)



@torch.no_grad()
def extract_superglobal(model):
    torch.backends.cudnn.benchmark = False
    model.eval()
    
    state_dict = model.state_dict()
    model.load_state_dict(state_dict)

    device = torch.device('cuda:0')

    with torch.no_grad():
        dataset = DataSet()
        # Create a loader
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=96,
            num_workers=4,
        )
        total_features = []
        for im_list in tqdm(loader):
            im_list = im_list.to(device)
            features = model.encoder_q._forward(im_list)
            features = torch.stack(features, dim=1)
            total_features.append(features)

        total_features = torch.concatenate(total_features)
        print(total_features.size())
        torch.save(total_features, './global_train.pt')


def create_topk_set():
    # with open('gldv2/train_set.pkl', 'rb') as f:
    #     labels = pkl.load(f)
    # print(labels)
    # [124774, 3, 2048]
    total_features = torch.load('gldv2/global_train.pt').cpu()
    total_features = total_features[:, 1, :]
    data_length = total_features.size(0)
    
    with open('gldv2/label_count.pkl', 'rb') as f:
        label_counts = pkl.load(f)
    print(len(label_counts))
    print(data_length)
    total_sims = []
    transpose_feature = torch.transpose(total_features, 0, 1).cuda()
    split_num = 10
    chunk_size = (data_length // split_num) + 1
    train_pn = []
    processed_idx = 0
    label_index = 0
    comp_index = 0
    for i in range(split_num):
        print(i)
        feature_chunk = total_features[chunk_size*i:chunk_size*(i+1)]
        sims = torch.matmul(feature_chunk.cuda(), transpose_feature).cpu()
        sims_index = torch.argsort(sims, dim=-1, descending=True)
        print(len(sims_index))

        for j in tqdm(range(len(sims_index))):
            if comp_index + j - processed_idx >= label_counts[label_index]:
                processed_idx += label_counts[label_index]
                label_index += 1
            neg_idx = []
            pos_idx = list(range(processed_idx, processed_idx + label_counts[label_index]))
            for index in sims_index[j]:
                if index not in pos_idx:
                    neg_idx.append(index.item())
                if len(neg_idx) > 99:
                    break
            train_pn.append({
                'pos': pos_idx,
                'neg': neg_idx,
            })
        comp_index += len(sims_index)
        del feature_chunk, sims, sims_index
        torch.cuda.empty_cache()

    # train_pn = []
    # label_index = 0
    # processed_idx = 0
    # split_num = 1000
    # chunk_size = (data_length // split_num) + 1
    # for i in tqdm(range(chunk_size)):
    #     if i - processed_idx >= label_counts[label_index]:
    #         processed_idx += label_counts[label_index]
    #         label_index += 1
    #     sims = torch.matmul(total_features[i], transpose_feature)
    #     sims_index = torch.argsort(sims, dim=-1, descending=True)
    #     neg_idx = []
    #     pos_idx = list(range(processed_idx, processed_idx + label_counts[label_index]))
    #     for index in sims_index:
    #         if index not in pos_idx:
    #             neg_idx.append(index.item())
    #         if len(neg_idx) > 99:
    #             break
    #     train_pn.append({
    #         'pos': pos_idx,
    #         'neg': neg_idx,
    #     })

        # if i - processed_idx >= label_counts[label_index]:
        #     processed_idx += label_counts[label_index]
        #     label_index += 1
        # sims = torch.matmul(total_features[i], transpose_feature)
        # sims_index = torch.argsort(sims, dim=-1, descending=True)
        # neg_idx = []
        # pos_idx = list(range(processed_idx, processed_idx + label_counts[label_index]))
        # for index in sims_index:
        #     if index not in pos_idx:
        #         neg_idx.append(index.item())
        #     if len(neg_idx) > 99:
        #         break
        # train_pn.append({
        #     'pos': pos_idx,
        #     'neg': neg_idx,
        # })
        
    print(len(train_pn))
    with open('train_pn.pkl', 'wb') as tf:
        pkl.dump(train_pn, tf)