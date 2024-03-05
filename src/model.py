import math

import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.nn import Parameter
from torch.utils.data import DataLoader

from transformers import AutoModel, AutoConfig
from sentence_transformers.util import cos_sim

from src.utils import compute_similarity_matrix


class GZXML(nn.Module):
    def __init__(self, args):
        super(GZXML, self).__init__()
        self.cls = Classification(args, feature_layers=args.feature_layers, dropout=args.dropout)

        self.sigmoid = nn.Sigmoid()

        self.loss_function = nn.BCEWithLogitsLoss()
        self.lb_embedding = None

    def forward(self, cls_input_ids=None, cls_attention_mask=None, new_feature=None, beta=None):
        cls_outs, encoder_out = self.cls(
            input_ids=cls_input_ids,
            attention_mask=cls_attention_mask,
            new_feature=new_feature
        )
        if beta is not None:
            rho = self.sigmoid(self.fc_y(beta))
            factor1 = 1 / (rho + 1)
            factor2 = 1 - factor1
            cls_outs = self.sigmoid(cls_outs)
            y_pre = factor1 * cls_outs + factor2 * beta
        else:
            y_pre = cls_outs
        return y_pre, encoder_out

    def init_lb_embedding(self, args, lb_data):
        lb_loader = DataLoader(lb_data, num_workers=0, batch_size=args.eval_batch, shuffle=False)

        lb_embedding = []
        with torch.no_grad():
            for step, batch in enumerate(lb_loader):
                lb_outs = self.cls.cls_encoder(
                    input_ids=batch[0].to('cuda'),
                    attention_mask=batch[1].to('cuda')
                )[-1]
                encoder_out = torch.cat([lb_outs[-i][:, 0] for i in range(1, self.feature_layers + 1)], dim=-1)
                # Normalize embeddings
                lb_emb = F.normalize(encoder_out, p=2, dim=1)
                lb_embedding.append(lb_emb.cpu())
        self.lb_embedding = torch.cat(lb_embedding)


class Classification(nn.Module):
    def __init__(self, args, feature_layers, dropout):
        super(Classification, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.feature_layers = feature_layers
        self.fcl1 = nn.Linear(feature_layers * 768, args.n_labels)

        model_config = AutoConfig.from_pretrained(args.cls_encoder)
        model_config.output_hidden_states = True
        self.cls_encoder = AutoModel.from_pretrained(args.cls_encoder, config=model_config)

    def forward(self, input_ids=None, attention_mask=None, new_feature=None):
        if new_feature is not None:
            output = self.fcl(new_feature)
            return output, None
        outs = self.cls_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )[-1]
        # get [cls] hidden states
        encoder_out = torch.cat([outs[-i][:, 0] for i in range(1, self.feature_layers + 1)], dim=-1)
        encoder_out = self.dropout(encoder_out)
        output = self.fcl1(encoder_out)

        return output, encoder_out.detach().cpu()


class Similarity(nn.Module):
    def __init__(self, args):
        super(Similarity, self).__init__()
        self.sim_encoder = AutoModel.from_pretrained(args.sim_encoder)
        self.lb_embedding = None

    def forward(self, args, sim_input_ids, sim_attention_mask, batch_label=None, lb_ids=None, lb_input_ids=None,
                lb_attention_mask=None):
        ins_outs = self.sim_encoder(
            input_ids=sim_input_ids,
            attention_mask=sim_attention_mask,
        )
        if batch_label is None:
            ins_emb = self.mean_pooling(ins_outs, sim_attention_mask)
            ins_emb = F.normalize(ins_emb, p=2, dim=1)
            sim = cos_sim(ins_emb, self.lb_embedding.cuda())
            return sim

        lb_outs = self.sim_encoder(
            input_ids=lb_input_ids,
            attention_mask=lb_attention_mask,
        )
        # # Perform pooling
        ins_emb = self.mean_pooling(ins_outs, sim_attention_mask)
        lb_emb = self.mean_pooling(lb_outs, lb_attention_mask)
        # # Normalize embeddings
        ins_emb = F.normalize(ins_emb, p=2, dim=1)
        lb_emb = F.normalize(lb_emb, p=2, dim=1)

        return ins_emb, lb_emb, lb_ids

    # Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def init_lb_embedding(self, args, lb_data):

        lb_loader = DataLoader(lb_data, num_workers=0, batch_size=args.eval_batch, shuffle=False)

        lb_embedding = []
        self.to(args.device)
        with torch.no_grad():
            for step, batch in enumerate(lb_loader):
                lb_outs = self.sim_encoder(
                    input_ids=batch[0].to('cuda'),
                    attention_mask=batch[1].to('cuda')
                )
                # Perform pooling
                lb_emb = self.mean_pooling(lb_outs, batch[1].to('cuda'))
                # Normalize embeddings
                lb_emb = F.normalize(lb_emb, p=2, dim=1)
                lb_embedding.append(lb_emb.cpu())
        self.lb_embedding = torch.cat(lb_embedding)


class ContrastiveLoss(nn.Module):
    def __init__(self, tem=0.05):
        super(ContrastiveLoss, self).__init__()
        self.tem = tem

    def forward(self, output1, output2, label):
        cos_sin_result = cos_sim(output1, output2) / self.tem
        sim_ii = torch.mul(cos_sin_result, label)

        exp_sim_ij = torch.exp(cos_sin_result)
        softmax_scores = sim_ii - torch.log(exp_sim_ij.sum(1, keepdim=True))
        softmax_scores = torch.mul(softmax_scores, label)
        return -softmax_scores.mean()


class VAE(nn.Module):
    def __init__(self, args):
        super(VAE, self).__init__()
        feature_layers = args.feature_layers
        self.encoder_ins = VAE_Encoder(input_dim=feature_layers * 768, latent_size=64, hidden_size=768)
        self.encoder_lab = VAE_Encoder(input_dim=768, latent_size=64, hidden_size=768)

        self.decoder = VAE_Decoder(input_dim=64, ins_dim=feature_layers * 768, lab_dim=768, hidden_size=768)

        self.reconstruction_criterion = nn.MSELoss(size_average=False)
        self.reparameterize_with_noise = True
        parameters_to_optimize = list(self.parameters())

        parameters_to_optimize += list(self.encoder_ins.parameters())
        parameters_to_optimize += list(self.encoder_lab.parameters())
        parameters_to_optimize += list(self.decoder.parameters())
        self.optimizer = optim.Adam(parameters_to_optimize, lr=0.00015, betas=(0.9, 0.999),
                                    eps=1e-08, weight_decay=0, amsgrad=True)

    def reparameterize(self, mu, logvar):
        if self.reparameterize_with_noise:
            sigma = torch.exp(logvar).cuda()
            eps = torch.FloatTensor(logvar.size()[0], 1).normal_(0, 1).cuda()
            eps = eps.expand(sigma.size())
            return mu.cuda() + sigma * eps
        else:
            return mu

    def forward(self, ins, lb, process='train'):

        if process is 'train':
            mu_ins, logvar_ins = self.encoder_ins(ins)
            z_from_ins = self.reparameterize(mu_ins, logvar_ins)
        else:
            z_from_ins = ins
        mu_lb, logvar_lb = self.encoder_lab(lb)
        z_from_lb = self.reparameterize(mu_lb, logvar_lb)

        ins_from_ins, lb_from_lb = self.decoder(z_from_ins, z_from_lb)
        if process is not 'train':
            return ins_from_ins
        reconstruction_loss = self.reconstruction_criterion(ins_from_ins, ins) \
                              + self.reconstruction_criterion(lb_from_lb, lb)

        KLD = (0.5 * torch.sum(1 + logvar_lb - mu_lb.pow(2) - logvar_lb.exp())) \
              + (0.5 * torch.sum(1 + logvar_ins - mu_ins.pow(2) - logvar_ins.exp()))

        sigma = 1
        sim_lb = compute_similarity_matrix(lb)
        dists = torch.cdist(ins_from_ins, ins_from_ins)
        gauss_kernel = torch.exp(-dists / sigma ** 2)
        sim_loss = torch.sum(torch.abs(gauss_kernel - sim_lb))

        beta = 1
        loss = reconstruction_loss - beta * KLD + sim_loss

        return loss


class VAE_Encoder(nn.Module):
    def __init__(self, input_dim, latent_size, hidden_size):
        super(VAE_Encoder, self).__init__()

        self.feature_encoder = nn.Linear(input_dim, hidden_size)
        self.relu = nn.ReLU()
        self._mu = nn.Linear(in_features=hidden_size, out_features=latent_size)
        self._logvar = nn.Linear(in_features=hidden_size, out_features=latent_size)

    def forward(self, x):
        h = self.feature_encoder(x)
        h = self.relu(h)
        mu = self._mu(h)
        logvar = self._logvar(h)

        return mu, logvar


class VAE_Decoder(nn.Module):

    def __init__(self, input_dim, ins_dim, lab_dim, hidden_size):
        super(VAE_Decoder, self).__init__()

        self.ins_fc = nn.Linear(input_dim, hidden_size)
        self.lab_fc = nn.Linear(input_dim, hidden_size)
        self.ins_out = nn.Linear(hidden_size, ins_dim)
        self.lab_out = nn.Linear(hidden_size, lab_dim)
        self.ins_weight = Parameter(torch.FloatTensor(hidden_size, hidden_size))
        self.lab_weight = Parameter(torch.FloatTensor(hidden_size, hidden_size))
        stdv = 1. / math.sqrt(self.ins_weight.size(1))
        self.ins_weight.data.uniform_(-stdv, stdv)
        self.lab_weight.data.uniform_(-stdv, stdv)

        self.relu = nn.ReLU()

        self.soft_max = nn.Softmax()

    def forward(self, input_ins, input_lab):
        ins = self.ins_fc(input_ins)
        lab = self.lab_fc(input_lab)

        ins = self.relu(ins)
        lab = self.relu(lab)

        att = torch.mm(ins.transpose(0, 1), lab) / np.sqrt(ins.size(1))
        att = self.soft_max(att)

        ins = ins + torch.mm(torch.mm(ins, self.ins_weight), att)
        lab = lab + torch.mm(torch.mm(lab, self.lab_weight), att)

        output_ins = self.ins_out(ins)
        output_lab = self.lab_out(lab)

        return output_ins, output_lab
