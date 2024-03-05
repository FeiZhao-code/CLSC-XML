import pickle
from os import path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer

from src.evaluate import sim_evaluate
from src.model import VAE, Similarity, ContrastiveLoss
from src.utils import do_tokenizer


def train_sim(args, lb_feature, train_sim_loader):
    sim = Similarity(args)
    epochs = 1
    lr = 2e-5

    tokenizer = AutoTokenizer.from_pretrained(args.sim_encoder)
    lb_tensor = do_tokenizer(args, lb_feature, tokenizer, 'sim_label')

    input_ids_list = [f["input_ids"].tolist() for f in lb_tensor]
    all_lb_input_ids = torch.tensor(input_ids_list, dtype=torch.long).squeeze(1)
    #
    attention_mask_list = [f["attention_mask"].tolist() for f in lb_tensor]
    all_lb_attention_mask = torch.tensor(attention_mask_list, dtype=torch.long).squeeze(1)
    lb_data = TensorDataset(all_lb_input_ids, all_lb_attention_mask)

    t_total = len(train_sim_loader) * epochs
    no_decay = ['bias', 'LayerNorm.weight']
    param_optimizer = [[name, para] for name, para in sim.named_parameters() if para.requires_grad]
    optimizer_grouped_parameters = [
        {'params': [param for name, param in param_optimizer if not any(nd in name for nd in no_decay)],
         'weight_decay': 0.01, "lr": lr},
        {'params': [param for name, param in param_optimizer if any(nd in name for nd in no_decay)],
         'weight_decay': 0.0, "lr": lr}
    ]

    optimizer = AdamW(optimizer_grouped_parameters)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)
    # Train!
    print("*****SIM Training *****")
    sim.to(args.device)
    sim.zero_grad()
    sim.train()
    criterion = ContrastiveLoss()

    for epoch in range(int(epochs)):
        train_loss = 0.0
        loop = tqdm(enumerate(train_sim_loader), total=len(train_sim_loader))
        for step, batch in loop:
            Y_labels = batch[2].cuda().to_dense().float()
            lb_ids, lb_input_ids, lb_attention_mask = sim.get_batch_label(batch[2], all_lb_input_ids,
                                                                          all_lb_attention_mask)
            ins_emb, lb_emb, lb_ids = sim(args,
                                          sim_input_ids=batch[0].cuda(),
                                          sim_attention_mask=batch[1].cuda(),
                                          batch_label=batch[2],
                                          lb_ids=lb_ids,
                                          lb_input_ids=lb_input_ids,
                                          lb_attention_mask=lb_attention_mask
                                          )

            label = torch.index_select(Y_labels, 1, torch.tensor(lb_ids).cuda())
            loss = criterion(ins_emb, lb_emb, label)
            train_loss += loss.item()
            loss.backward()
            loop.set_description(f'Epoch [{epoch + 1}/{args.epochs}]')
            loop.set_postfix(loss=[train_loss / (step + 1)],
                             lr="Lr:{}".format(optimizer.state_dict()['param_groups'][0]['lr']))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        sim.init_lb_embedding(args, lb_data)
    sim_output = sim_evaluate(args, sim, train_sim_loader)
    torch.save(sim.state_dict(), f"./output/{args.dataset}/sim.bin")
    return sim_output


def train(args, model, train_cls_loader, train_sim_loader, new_mask, lb_feature):

    t_total = len(train_cls_loader) * args.epochs

    no_decay = ['bias', 'layer_norm.bias', 'layer_norm.weight']
    param_optimizer = [[name, para] for name, para in model.named_parameters() if para.requires_grad]
    optimizer_grouped_parameters = [
        {'params': [param for name, param in param_optimizer if not any(nd in name for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [param for name, param in param_optimizer if any(nd in name for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)
    # Train!
    print("***** Training *****")
    model.to(args.device)
    model.zero_grad()
    model.train()

    tokenizer = AutoTokenizer.from_pretrained(args.cls_encoder)
    lb_tensor = do_tokenizer(args, lb_feature, tokenizer, 'cls_label')

    input_ids_list = [f["input_ids"].tolist() for f in lb_tensor]
    lb_input_ids = torch.tensor(input_ids_list, dtype=torch.long).squeeze(1)
    attention_mask_list = [f["attention_mask"].tolist() for f in lb_tensor]
    lb_attention_mask = torch.tensor(attention_mask_list, dtype=torch.long).squeeze(1)

    vae = VAE(args)

    sim_output = train_sim(args, lb_feature, train_sim_loader)
    for epoch in range(int(args.epochs)):

        train_loss = 0.0
        new_train_loss = 0.0
        all_ins_emb = []
        all_label = []

        loop = tqdm(enumerate(train_cls_loader), total=len(train_cls_loader))
        for step, batch in loop:
            Y_labels = batch[2].cuda().to_dense().float()

            beta = sim_output[step * args.batch:step * args.batch + Y_labels.shape[0], :]
            cls_output, encoder_out = model(args,
                                            cls_input_ids=batch[0].cuda(),
                                            cls_attention_mask=batch[1].cuda(),
                                            beta=beta
                                            )

            all_ins_emb.append(encoder_out)
            all_label.append(batch[2])
            predict = cls_output
            loss = model.loss_function(predict, Y_labels)
            train_loss += loss.item()
            loss.backward()
            loop.set_description(f'Epoch [{epoch + 1}/{args.epochs}]')
            loop.set_postfix(loss=[train_loss / (step + 1)],
                             lr="Lr:{}".format(optimizer.state_dict()['param_groups'][0]['lr']))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        lb_data = TensorDataset(lb_input_ids, lb_attention_mask)
        model.init_lb_embedding(args, lb_data)
        ten_ins_emb = torch.cat(all_ins_emb, dim=0)
        ten_lab = torch.cat(all_label, dim=0)
        sum_ins_lab = torch.mm(ten_lab.float(), model.lb_embedding)
        #
        train_vae(args, vae, TensorDataset(ten_ins_emb, sum_ins_lab))
        if epoch >= args.gen_epochs:
            index = torch.tensor(np.where(np.array(new_mask) == 1)[0])
            new_feature = gen_ins2nlb(args, torch.index_select(model.lb_embedding, 0, index))
            new_lb_path = path.join("./output/{}".format(args.dataset), "new_lb_tensor.pkl")
            with open(new_lb_path, "rb") as fin:
                new_lb = pickle.load(fin)

            new_tensor = TensorDataset(new_feature, new_lb)
            new_loader = DataLoader(new_tensor, batch_size=args.batch, shuffle=True)

            new_loop = tqdm(enumerate(new_loader), total=len(new_loader))
            for step, batch in new_loop:
                new_cls_outs = model(args, new_feature=batch[0].cuda())
                new_loss = model.loss_function(new_cls_outs[0], batch[1].cuda().float())
                new_train_loss += new_loss.item()
                new_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

    torch.save(model.state_dict(), f"./output/{args.dataset}/model.bin")


def gen_ins2nlb(args, new_lb_emb):
    vae = VAE(args)
    vae.load_state_dict(torch.load(f'output/{args.dataset}/vae.bin'))
    vae.to(args.device)
    z = torch.normal(mean=0., std=1., size=(new_lb_emb.shape[0], 64))
    eval_tensor = TensorDataset(z, new_lb_emb)
    dataloader = DataLoader(eval_tensor, num_workers=3, batch_size=args.eval_batch, shuffle=False)

    print("Generating instances for novel label ~~~")
    vae.to('cuda')
    vae.eval()
    all_ins2nlb = []
    loop = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, batch in loop:
        with torch.no_grad():
            out = vae(batch[0].cuda(), batch[1].cuda(), process='gen')
            all_ins2nlb.append(out)
    ten_all_ins2nlb = torch.cat(all_ins2nlb, dim=0)

    return ten_all_ins2nlb


def train_vae(args, vae, train_tensor):
    dataloader = DataLoader(train_tensor, batch_size=args.batch * 4, shuffle=True,
                            num_workers=3)
    vae.to(args.device)
    vae.zero_grad()
    vae.train()
    vae.reparameterize_with_noise = True

    print('Train for reconstruction ~~~')
    for epoch in range(0, args.vae_epochs):  # args.vae_epochs):
        vae.current_epoch = epoch
        loop = tqdm(enumerate(dataloader), total=len(dataloader))
        vae_loss = 0.0
        for step, batch in loop:
            ins, lb = batch[0].cuda(), batch[1].cuda()
            loss = vae(ins, lb)
            vae_loss += loss.item()
            if step % 50 == 0:
                loop.set_description(f'Epoch [{epoch + 1}/{args.vae_epochs}]')
                loop.set_postfix(loss=[vae_loss / (step + 1)])
            loss.backward()
            vae.optimizer.step()
            vae.optimizer.zero_grad()
    torch.save(vae.state_dict(), f"./output/{args.dataset}/vae.bin")
