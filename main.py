import argparse
import os
import random

import numpy as np
import torch

from src.evaluate import evaluate, get_lb_tensor, sim_evaluate
from src.model import GZXML, Similarity
from src.train import train
from src.utils import read_dataset, data2tensor, Logger


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, required=False, default='')
    parser.add_argument("--eval", action='store_true', help="Whether to run test.")
    parser.add_argument('--n_labels', type=int, required=False, default=0)
    parser.add_argument('--dataset', type=str, required=False, default='Eurlex-4.3K')
    parser.add_argument("--encoder_type", type=str, default="roberta")
    parser.add_argument("--cls_encoder", type=str, default='roberta-base')
    parser.add_argument("--sim_encoder", type=str, default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--feature_layers', type=int, required=False, default=5)
    parser.add_argument('--dataset_path', type=str, required=False, metavar="DIR")
    parser.add_argument("--cache_dir", default="./dataset", type=str)
    parser.add_argument('--output_data_dir', type=str, required=False, metavar="DIR")
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, required=False, default=0)
    parser.add_argument('--batch', type=int, required=False, default=24)
    parser.add_argument('--eval_batch', type=int, required=False, default=48)
    parser.add_argument('--lr', type=float, required=False)
    parser.add_argument('--dropout', type=float, required=False)
    parser.add_argument('--epochs', type=int, required=False)
    parser.add_argument('--gen_epochs', type=int, required=False)
    parser.add_argument('--vae_epochs', type=int, required=False)
    parser.add_argument('--max_len', type=int, required=False, default=512)
    parser.add_argument('--novel_r', type=float, required=False, default=0.5)
    parser.add_argument('--t_f1', type=float, required=False, default=0.5)
    parser.add_argument('--warmup_steps', type=int, required=False, default=100)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    print(args)

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.exp_name == '':
        args.exp_name = f'{args.dataset}_{args.encoder_type}'
    args.dataset_path = f'./dataset/{args.dataset}'
    args.output_data_dir = f'./output/{args.dataset}'
    os.makedirs(args.output_data_dir, exist_ok=True)
    log_str = " ExpName:{} \nArgs:{}".format(args.exp_name, args)
    Logger('log_' + args.exp_name).log(log_str)

    lb_feature, new_mask, train_feature, test_feature, train_label, test_label = read_dataset(args)
    args.n_labels = len(lb_feature)

    model = GZXML(args)

    if args.eval:
        model.load_state_dict(torch.load(f'output/{args.dataset}/model.bin'))
        eval_cls_loader, eval_sim_loader = data2tensor(args, text=test_feature, label=test_label, process='eval')
        sim = Similarity(args)
        sim.load_state_dict(torch.load(f'output/{args.dataset}/sim.bin'))
        lb_data = get_lb_tensor(args, lb_feature)
        sim.init_lb_embedding(args, lb_data)
        sim_eval_output = sim_evaluate(args, sim, eval_sim_loader)
        evaluate(args, model, eval_cls_loader, sim_eval_output)
        exit(0)

    train_cls_loader, train_sim_loader = data2tensor(args, text=train_feature, label=train_label, process='train')
    train(args, model, train_cls_loader, train_sim_loader, new_mask, lb_feature)
