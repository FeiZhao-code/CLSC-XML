import datetime
import os
import pickle
from os import path
from os.path import join

import numpy as np
import pandas as pd
import torch
from scipy import sparse

from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from functools import partial
import scipy.sparse as smat
from transformers import AutoTokenizer


class Logger:
    def __init__(self, name):
        self.name = name

    def log(self, text):
        os.makedirs('./log/', exist_ok=True)
        with open(f'./log/{self.name}.txt', 'a') as f:
            f.write(datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S') + text + '\n')


def read_dataset(args):
    trn_js = pd.read_json(join("./dataset/{}".format(args.dataset), 'train_split.jsonl'), lines=True)

    tst_js = pd.read_json(join("./dataset/{}".format(args.dataset), 'test_split.jsonl'), lines=True)
    train_feature = list(trn_js.text)
    test_feature = list(tst_js.text)
    train_label = list(trn_js.label)
    test_label = list(tst_js.label)
    lb_list, new_lb_list, trn_lb_id, tst_lb_id = [], [], [], []
    with open(join("./dataset/{}".format(args.dataset), 'all_labels.txt')) as f:
        for i in f:
            lb_list.append(i.replace('\n', ''))
    with open(join("./dataset/{}".format(args.dataset), 'unseen_labels.txt')) as f:
        for i in f:
            new_lb_list.append(i.replace('\n', ''))
    mask = [1 if i in new_lb_list else 0 for i in lb_list]
    for labels in train_label:
        i_label = [lb_list.index(lb) for lb in labels]
        trn_lb_id.append(i_label)
    for labels in test_label:
        i_label = [lb_list.index(lb) for lb in labels]
        tst_lb_id.append(i_label)

    return lb_list, mask, train_feature, test_feature, trn_lb_id, tst_lb_id


def encode_data(args, tokenizer, train_sents, process, shuffle=False, batch=64):
    X_train = tokenizer.batch_encode_plus(train_sents, padding=True, truncation=True,
                                          max_length=args.max_len, return_tensors='pt')

    if os.path.exists(join("./dataset/{}".format(args.dataset), 'trn_X_Y.npz')):
        if process == "train":
            Y = smat.load_npz(join("./dataset/{}".format(args.dataset), 'trn_X_Y.npz'))

        else:
            Y = smat.load_npz(join("./dataset/{}".format(args.dataset), 'tst_X_Y.npz'))
    Y_sptensor = torch.tensor(Y)
    train_tensor = TensorDataset(X_train['input_ids'], X_train['attention_mask'], Y_sptensor)
    train_loader = DataLoader(train_tensor, num_workers=10, batch_size=batch, shuffle=shuffle)

    return train_loader


def data2tensor(args, text, label, process):
    # load pretrained model tokenizers
    cls_tokenizer = AutoTokenizer.from_pretrained(args.cls_encoder)
    sim_tokenizer = AutoTokenizer.from_pretrained(args.sim_encoder)

    if process == 'eval':
        shuffle = False
    else:
        shuffle = True

    cls_loader = encode_data(args, cls_tokenizer, text, label, process, shuffle=shuffle, batch=args.batch)
    sim_loader = encode_data(args, sim_tokenizer, text, label, process, shuffle=shuffle, batch=args.batch * 2)

    return cls_loader, sim_loader


def do_tokenizer(args, text, tokenizer, text_type):
    out_feat_path = path.join("./output/{}".format(args.dataset), "{}-{}-{}.pkl".format(text_type, args.encoder_type,
                                                                                        20))
    if os.path.exists(out_feat_path):
        with open(out_feat_path, "rb") as fin:
            out_feat = pickle.load(fin)
        return out_feat

    text_features = []
    for (i, xseq) in tqdm(enumerate(text)):
        X = tokenizer(xseq, padding='max_length', truncation=True, return_attention_mask=True, max_length=32,
                      return_tensors='pt')
        cur_inst_dict = {
            'input_ids': X["input_ids"],
            'attention_mask': X["attention_mask"],
        }
        text_features.append(cur_inst_dict)

    with open(out_feat_path, "wb") as fout:
        pickle.dump(text_features, fout, protocol=pickle.HIGHEST_PROTOCOL)
    return text_features


def compute_similarity_matrix(labels):
    dot = torch.matmul(labels, labels.t())
    l2_norm = torch.norm(labels, dim=1)
    sim = dot / torch.matmul(l2_norm.unsqueeze(1), l2_norm.unsqueeze(0))
    return sim


def get_ndcg(prediction, targets, top=5):
    log = 1.0 / np.log2(np.arange(top) + 2)
    dcg = np.zeros((targets.shape[0], 1))

    for i in range(top):
        idx = np.argsort(-prediction, axis=1)[:, i:i + 1].flatten()
        p = sparse.csr_matrix((idx[:, None] == np.arange(prediction.shape[1])).astype(float))
        dcg += p.multiply(targets).sum(axis=-1) * log[i]
    return np.average(dcg / log.cumsum()[np.minimum(targets.sum(axis=-1), top) - 1])


get_n_1 = partial(get_ndcg, top=1)
get_n_3 = partial(get_ndcg, top=3)
get_n_5 = partial(get_ndcg, top=5)
