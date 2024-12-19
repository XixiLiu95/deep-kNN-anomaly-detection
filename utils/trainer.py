import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorboardX import SummaryWriter
from test import  test_single_model_cleaner_KNN_merge_bank
import torch.nn.functional as F
from models import EntropyLossEncap,  get_model, get_loader, get_loader_siamese, compute_features 
 
from torch.nn.functional import normalize
 



def train_module(cfgs, opt, out_dir):
    Model = cfgs["Model"]
    network = Model["network"]
    mp = Model["mp"]
    ls = Model["ls"]
    mem_dim = Model["mem_dim"]
    shrink_thres = Model["shrink_thres"]
 

    Data = cfgs["Data"]
    dataset = Data["dataset"]
    img_size = Data["img_size"]
    data_ratio = opt.data_ratio
    outlier_ratio = opt.outlier_ratio

    Solver = cfgs["Solver"]
    bs = Solver["bs"]
    lr = Solver["lr"]
    weight_decay = Solver["weight_decay"]
    num_epoch = opt.epoch

    if opt.mode == "a":
        train_loader = get_loader(dataset=dataset, dtype="train", bs=bs, img_size=img_size, workers=1,
                                   data_ratio=data_ratio, outlier_ratio=outlier_ratio)
        test_loader = get_loader(dataset=dataset, dtype="train", bs=bs, img_size=img_size, workers=1,
                                   data_ratio=data_ratio, outlier_ratio=outlier_ratio)
    elif opt.mode in ["train","AE", "siamese", "barlow", "siamese_CLRv2"] :
        train_loader = get_loader_siamese(network=network, dataset=dataset, dtype="train", bs=bs, img_size=img_size, workers=1,
                                  data_ratio=data_ratio, outlier_ratio=outlier_ratio,without_pre_trained=opt.without_pre_trained)
        test_loader = get_loader_siamese(network=network,dataset=dataset, dtype="test", bs=1, img_size=img_size, workers=1,without_pre_trained=opt.without_pre_trained)
    else:  # b
        print("right dataloader for mode b, which is AE or AE-U")
        train_loader = get_loader(dataset=dataset, dtype="train", bs=bs, img_size=img_size, workers=1,
                                   data_ratio=data_ratio, outlier_ratio=outlier_ratio)
        test_loader = get_loader(dataset=dataset, dtype="test", bs=bs, img_size=img_size, workers=1,
                                   data_ratio=data_ratio, outlier_ratio=outlier_ratio)
     

    model = get_model(network=network, mp=mp, ls=ls, img_size=img_size, mem_dim=mem_dim, shrink_thres=shrink_thres,opt=opt)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=weight_decay)
    writer = SummaryWriter(os.path.join(out_dir, "log"))
    if network in ["AE", "AE-U"]:
        model = AE_trainer(model, train_loader, test_loader, optimizer, num_epoch, writer, cfgs, opt, out_dir)
    elif network in ["siamese_CLRv2","barlow"]:
        model = Siamese_CLR_trainer(model, train_loader, test_loader, optimizer, num_epoch, writer, cfgs, opt,out_dir )
    elif network == "MemAE":
        model = MemAE_trainer(model, train_loader, test_loader, optimizer, num_epoch, writer, cfgs, opt,out_dir)
    writer.close()
    print()
    
     

def Siamese_CLR_trainer(model, train_loader, test_loader, optimizer, num_epoch, writer, cfgs, opt, out_dir):
    t0 = time.time()
     
    data_ratio = opt.data_ratio
    outlier_ratio = opt.outlier_ratio
    
    k = opt.k
    for e in range(num_epoch):
        l1s, l2s = [], []
        model.train() # the output is latent code 
        for i, images in enumerate(train_loader):
            x1 = images[0][0].cuda()
            x2 = images[0][1].cuda()
            x1.requires_grad = False 
            x2.requires_grad = False
            z1 = model(x1)
            z2 = model(x2)
            p1,p2 = model.predictor(z1), model.predictor(z2)
            loss = cosin_similarity(p1,z2)/2 + cosin_similarity(p2,z1)/2
            l1s.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
         
        l1s = np.mean(l1s)
        writer.add_scalar('rec_err', l1s, e)
    #t = time.time()-t0
    #print("total training time {:.2f}s".format( t))
     
     
        if (e+1) % 20 == 0:
            t = time.time() - t0
            t0 = time.time()
            train_feature  = compute_features(train_loader, model)
            auc, ap, y_true, y_score = test_single_model_cleaner_KNN_merge_bank(model=model, test_loader=test_loader, train_feature=train_feature, K=k)
            writer.add_scalar('AUC', auc, e)
            writer.add_scalar('AP', ap, e)
            print("Mode {}. Epoch[{}/{}]  Time:{:.2f}s  AUC:{:.3f}  AP:{:.3f}   "
                  "Rec_err:{:.5f}".format(opt.mode, e, num_epoch, t, auc, ap, l1s))    
            model_path =   out_dir 
            if not os.path.exists(model_path):
                os.makedirs(model_path)   
            model_name = os.path.join(model_path, "{}_normal_{}_outlier_stable_epoch_{}.pth".format(data_ratio, outlier_ratio, e))
            torch.save(model.state_dict(), model_name)
            print("{} is saved in {}!".format(model_name,model_path))
     
    return model



def AE_trainer(model, train_loader, test_loader, optimizer, num_epoch, writer, cfgs, opt,out_dir):
    t0 = time.time()
    data_ratio = opt.data_ratio
    outlier_ratio = opt.outlier_ratio
    for e in range(num_epoch):
        l1s, l2s = [], []
        model.train()
        for (x, _, _) in train_loader:
            x = x.cuda()
            x.requires_grad = False
            if cfgs["Model"]["network"] == "AE":
                out = model(x)
                rec_err = (out - x) ** 2
                loss = rec_err.mean()
                l1s.append(loss.item())
            else:  # AE-U
                mean, logvar = model(x)
                rec_err = (mean - x) ** 2
                loss1 = torch.mean(torch.exp(-logvar) * rec_err)
                loss2 = torch.mean(logvar)
                loss = loss1 + loss2
                l1s.append(rec_err.mean().item())
                l2s.append(loss2.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        l1s = np.mean(l1s)
        l2s = np.mean(l2s) if len(l2s) > 0 else 0
        writer.add_scalar('rec_err', l1s, e)
        writer.add_scalar('logvars', l2s, e)

        if (e+1) % 10 == 0:    
            model_path =  out_dir 
            if not os.path.exists(model_path):
                os.makedirs(model_path)   
            model_name = os.path.join(model_path, "{}_normal_{}_outlier_stable_epoch_{}.pth".format(data_ratio, outlier_ratio, e))
             
            torch.save(model.state_dict(), model_name)
            print("{} is saved in {}!".format(model_name,model_path))
    return model

def cosin_similarity (p,z):
    z = z.detach()
    p =  normalize(p, dim =1)
    z =  normalize(z, dim =1)
    return -(p*z).sum(dim=1).mean()

def MemAE_trainer(model, train_loader, test_loader, optimizer, num_epoch, writer, cfgs, opt):
    criterion_entropy = EntropyLossEncap()
    entropy_loss_weight = cfgs["Model"]["entropy_loss_weight"]
    t0 = time.time()
    for e in range(num_epoch):
        l1s = []
        ent_ls = []
        model.train()
        for (x, _, _) in train_loader:
            x = x.cuda()
            x.requires_grad = False
            out = model(x)
            rec = out['output']
            att_w = out['att']

            rec_err = (rec - x) ** 2
            loss1 = rec_err.mean()
            entropy_loss = criterion_entropy(att_w)
            loss = loss1 + entropy_loss_weight * entropy_loss

            l1s.append(rec_err.mean().item())
            ent_ls.append(entropy_loss.mean().item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        l1s = np.mean(l1s)
        ent_ls = np.mean(ent_ls)
        writer.add_scalar('rec_err', l1s, e)
        writer.add_scalar('entropy_loss', ent_ls, e)
        if e % 25 == 0:
            t = time.time() - t0
            t0 = time.time()

            if opt.mode == "b":
                auc, ap = test_single_model(model=model, test_loader=test_loader, cfgs=cfgs)
                writer.add_scalar('AUC', auc, e)
                writer.add_scalar('AP', ap, e)
                print("Mode {}. Epoch[{}/{}]  Time:{:.2f}s  AUC:{:.3f}  AP:{:.3f}   "
                      "Rec_err:{:.5f}   Entropy_loss:{:.5f}".format(opt.mode, e, num_epoch, t, auc, ap, l1s, ent_ls))
            else:
                print("Mode {}. Epoch[{}/{}]  Time:{:.2f}s  "
                      "Rec_err:{:.5f}   Entropy_loss:{:.5f}".format(opt.mode, e, num_epoch, t, l1s, ent_ls))

    return model
