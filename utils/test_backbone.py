import torch
from sklearn import metrics
from tqdm import tqdm
import numpy as np
import os
from models import get_model, get_loader, get_loader_siamese 
import matplotlib.pyplot as plt
 
from models.utils import negative_cosin_similarity,compute_features, compute_features_simple
import torch.nn.functional as F 
from torch.nn.functional import normalize
import faiss
 


def anomaly_score_histogram(y_score, y_true, anomaly_score, out_dir, f_name):
    plt.rcParams.update({'font.size': 14})
    plt.cla()
    plt.hist(y_score[y_true == 0], bins=100, density=True, color='blue', alpha=0.5, label="Normal")
    plt.hist(y_score[y_true == 1], bins=100, density=True, color='red', alpha=0.5, label="Abnormal")
    plt.xlabel(anomaly_score)
    plt.ylabel("Frequency")
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.yticks([0, 2, 4, 6, 8, 10])
    plt.xlim(0, 1)
    plt.ylim(0, 11)
    plt.legend()
    plt.savefig('{}/{}.pdf'.format(out_dir, f_name))

def makedirs(dataset, opt, network):
    k = opt.k
    if not os.path.exists(f'results/{dataset}/only_pre_trained_{network}/5_augmenttaions/{k}_nearest'):
             os.makedirs(f'results/{dataset}/only_pre_trained_{network}/5_augmentations/{k}_nearest')
     

def test_rec(cfgs, opt):

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
     

    if network in ["siamese_CLRv2_KNN", "barlow", "ReSSL", "siamese","simCLRv2"]:
        k = opt.k
        ### Extracted training feature
        train_loader = get_loader_siamese(network=network, dataset=dataset, dtype="train", bs=100, img_size=img_size, workers=1,
                                       data_ratio=data_ratio, outlier_ratio=outlier_ratio,without_pre_trained=opt.without_pre_trained )
        test_loader = get_loader_siamese(network=network, dataset=dataset, dtype="test", bs=1, img_size=img_size, workers=1,
                                   data_ratio=data_ratio, outlier_ratio=outlier_ratio,without_pre_trained=opt.without_pre_trained)
     
        print("=> Testing ... ")
        model = get_model(network=network, mp=mp, ls=ls, img_size=img_size, mem_dim=mem_dim, shrink_thres=shrink_thres,opt=opt)
        train_feature  = compute_features_simple(train_loader, model)
        print("test with only backbone")
        auc, ap, y_true, y_score =  test_single_model_cleaner_KNN_merge_bank(model=model, test_loader=test_loader, train_feature=train_feature, K=k)
         
        print(" AUC:{:.3f}  AP:{:.3f}".format(auc, ap))
        makedirs(dataset, opt, network)
        ### saves scores 
        with open(f'results/{dataset}/only_pre_trained_{network}/5_augmentations/{k}_nearest.txt', 'w') as f:
                        f.write(f'{round(auc*100,2)}, {round(ap*100,2)}')
           
        
 
def test_single_model_simCLRv2(model, test_loader, train_feature=None, K=5):
    # code is partially adpated from https://github.com/deeplearning-wisc/knn-ood/blob/master/run_cifar.py
    ####KNN###
    normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)
    prepos_feat = lambda x: np.ascontiguousarray(normalizer(x))# Last Layer only
    normalized_train_feature1 = prepos_feat(train_feature[0])
    normalized_train_feature2 = prepos_feat(train_feature[1])
    index1 = faiss.IndexFlatL2(normalized_train_feature1.shape[1])
    index1.add(normalized_train_feature1)
    index2 = faiss.IndexFlatL2(normalized_train_feature2.shape[1])
    index2.add(normalized_train_feature2)
    model.eval()
    with torch.no_grad():
        y_score, y_true = [], []
        for i, images in enumerate(test_loader): # need to double check the testloader
                    x1 = images[0][0].cuda()
                    x2 = images[0][1].cuda()
                    label = images[1]  
                    z1 = model(x1)
                    z2 = model(x2)
                    p1,p2 = z1, z2
                    p1= p1.cpu().detach().numpy()
                    p2= p2.cpu().detach().numpy()
                    p1 = prepos_feat(p1)
                    p2 = prepos_feat(p2)
                    D1, _ = index1.search(p1, K)
                    D2, _ = index2.search(p2, K)
                    #res = 1/K**2*(np.sum(D1[:,])*np.sum(D2[:,]))
                    res = D1[:,-1]*D2[:,-1]
                    y_true.append(label.cpu())
                    y_score.append(res)
        
        y_true = np.concatenate(y_true)
        y_score = np.array(y_score)
        #y_score = np.concatenate(y_score)
        auc = metrics.roc_auc_score(y_true, y_score)
        ap = metrics.average_precision_score(y_true, y_score)
        return auc, ap, y_true, y_score

def test_single_model_cleaner_KNN_merge_bank(model, test_loader, train_feature=None, K=5, normalization=True):
    # code is partially adpated from https://github.com/deeplearning-wisc/knn-ood/blob/master/run_cifar.py
    ####KNN###
    if normalization:
        normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)
        prepos_feat = lambda x: np.ascontiguousarray(normalizer(x))# Last Layer only
        normalized_train_feature1 = prepos_feat(train_feature[0]) ## number of samples x feature size
        normalized_train_feature2 = prepos_feat(train_feature[1])    
         
       
    else:
        normalized_train_feature1 =  train_feature[0]
        normalized_train_feature2 =  train_feature[1]

    normalized_train_feature = np.concatenate((normalized_train_feature1, normalized_train_feature2),axis=0)
    index = faiss.IndexFlatL2(normalized_train_feature.shape[1])
    index.add(normalized_train_feature)
    model.eval()
    with torch.no_grad():
        y_score, y_true = [], []
        for i, images in enumerate(test_loader): # need to double check the testloader
                    x1 = images[0][0].cuda()
                    x2 = images[0][1].cuda()
                     
                    
                    label = images[1]  
                    z1 = model(x1)
                    z2 = model(x2)

                    p1= z1.cpu().detach().numpy()
                    p2= z2.cpu().detach().numpy()
                      
                    if normalization:
                        p1 = prepos_feat(p1)
                        p2 = prepos_feat(p2)
                        

                    D1, _ = index.search(p1, K)
                    D2, _ = index.search(p2, K)
                     
                    res = np.power(D1[:,-1]*D2[:,-1],1/5)
                    y_true.append(label.cpu())
                    y_score.append(res)
        
        y_true = np.concatenate(y_true)
        y_score = np.array(y_score)
       
        auc = metrics.roc_auc_score(y_true, y_score)
        ap = metrics.average_precision_score(y_true, y_score)
        return auc, ap, y_true, y_score

def test_single_model_KNN_merge_bank(model, test_loader, train_feature=None, K=5, normalization=True):
    # code is partially adpated from https://github.com/deeplearning-wisc/knn-ood/blob/master/run_cifar.py
    ####KNN###
    if normalization:
        normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)
        prepos_feat = lambda x: np.ascontiguousarray(normalizer(x))# Last Layer only
        normalized_train_feature1 = prepos_feat(train_feature[0]) ## number of samples x feature size
        normalized_train_feature2 = prepos_feat(train_feature[1])
         
    else:
        normalized_train_feature1 =  train_feature[0]
        normalized_train_feature2 =  train_feature[1]

    #index1 = faiss.IndexFlatL2(normalized_train_feature1.shape[1])
    #index1.add(normalized_train_feature1)
    #index2 = faiss.IndexFlatL2(normalized_train_feature2.shape[1])
    #index2.add(normalized_train_feature2)
    normalized_train_feature = np.concatenate((normalized_train_feature1, normalized_train_feature2),axis=0)
    index = faiss.IndexFlatL2(normalized_train_feature.shape[1])
    index.add(normalized_train_feature)
    model.eval()
    with torch.no_grad():
        y_score, y_true = [], []
        for i, images in enumerate(test_loader): # need to double check the testloader
                    x1 = images[0][0].cuda()
                    x2 = images[0][1].cuda()
                    label = images[1]  
                    z1 = model(x1)
                    z2 = model(x2)
                     
                    p1= z1.cpu().detach().numpy()
                    p2= z2.cpu().detach().numpy()
                    if normalization:
                        p1 = prepos_feat(p1)
                        p2 = prepos_feat(p2)
                    D1, _ = index.search(p1, K)
                    D2, _ = index.search(p2, K)
                    res = np.power(D1[:,-1]*D2[:,-1],1/2)
                    #res1 =  np.power(np.prod(D1[:,]), 1/K) # geometric mean.
                    #res2 = np.power(np.prod(D2[:,]), 1/K)  # geometric mean.
                    #res = np.min([res1, res2]) 
                    #res = D1[:,-1]*D2[:,-1]
                    y_true.append(label.cpu())
                    y_score.append(res)
        
        y_true = np.concatenate(y_true)
        y_score = np.array(y_score)
        #y_score = np.concatenate(y_score)
        auc = metrics.roc_auc_score(y_true, y_score)
        ap = metrics.average_precision_score(y_true, y_score)
        return auc, ap, y_true, y_score


 

 