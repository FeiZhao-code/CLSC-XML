import warnings

import numpy as np
import torch
from scipy import sparse
from sklearn import metrics
from torch.utils.data import TensorDataset
from torchmetrics import Precision
from tqdm import tqdm
from transformers import AutoTokenizer

from src.utils import Logger, do_tokenizer, get_n_1, get_n_3, get_n_5

warnings.filterwarnings('ignore')


def get_lb_tensor(args, lb_feature):
    tokenizer = AutoTokenizer.from_pretrained(args.sim_encoder)
    lb_tensor = do_tokenizer(args, lb_feature, tokenizer, 'sim_label')

    input_ids_list = [f["input_ids"].tolist() for f in lb_tensor]
    all_lb_input_ids = torch.tensor(input_ids_list, dtype=torch.long).squeeze(1)
    #
    attention_mask_list = [f["attention_mask"].tolist() for f in lb_tensor]
    all_lb_attention_mask = torch.tensor(attention_mask_list, dtype=torch.long).squeeze(1)
    lb_data = TensorDataset(all_lb_input_ids, all_lb_attention_mask)
    return lb_data


def evaluate(args, model, eval_cls_loader, sim_output=None):
    print("Evaluating...")
    model.to('cuda')
    model.eval()
    loop = tqdm(enumerate(eval_cls_loader), total=len(eval_cls_loader))

    predicted_labels = []
    target_labels = []

    for step, batch in loop:
        with torch.no_grad():
            labels = batch[2].to_dense().int()
            if sim_output is not None:
                beta = sim_output[step * args.batch:step * args.batch + labels.shape[0], :]
            cls_output, cls_ecd = model(args,
                                        cls_input_ids=batch[0].cuda(),
                                        cls_attention_mask=batch[1].cuda(),
                                        beta=beta
                                        )
            predict = cls_output
            predicted_labels.append(torch.sigmoid(predict).detach().cpu())
            target_labels.append(labels.detach().cpu())

    tol_pre = torch.cat(predicted_labels, dim=0)
    tol_tar = torch.cat(target_labels, dim=0)
    P1 = Precision(top_k=1)(tol_pre, tol_tar).item()
    P3 = Precision(top_k=3)(tol_pre, tol_tar).item()
    P5 = Precision(top_k=5)(tol_pre, tol_tar).item()

    tol_pre = tol_pre.numpy()
    tol_tar = tol_tar.numpy()
    if args.eval:
        print("P1:{},P3:{},P5:{}".format(P1, P3, P5))
        print('nDCG@1,3,5:', get_n_1(tol_pre, sparse.csr_matrix(tol_tar)),
              get_n_3(tol_pre, sparse.csr_matrix(tol_tar)),
              get_n_5(tol_pre, sparse.csr_matrix(tol_tar)))
        micro_f1 = metrics.f1_score(tol_tar, np.where(tol_pre > args.t_f1, 1, 0), average='micro')
        macro_f1 = metrics.f1_score(tol_tar, np.where(tol_pre > args.t_f1, 1, 0), average='macro')
        print("micro_f1:{},macro_f1:{}".format(micro_f1, macro_f1))
    log_str = "P1:{},P3:{},P5:{}".format(P1, P3, P5)
    Logger('log_' + args.exp_name).log(log_str)


def sim_evaluate(args, sim, loader):
    print("SIM Evaluating...")
    sim.to('cuda')
    sim.eval()
    loop = tqdm(enumerate(loader), total=len(loader))
    sim_ins_emb = []

    for step, batch in loop:
        with torch.no_grad():
            sim_outs = sim(args,
                           sim_input_ids=batch[0].cuda(),
                           sim_attention_mask=batch[1].cuda(),
                           )
            sim_ins_emb.append(sim_outs)
    ten_ins_emb = torch.cat(sim_ins_emb, dim=0)
    return ten_ins_emb
