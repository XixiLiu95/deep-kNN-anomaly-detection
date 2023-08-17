from models import AE, MemAE, Siamese,  Siamese_CLR
from anomaly_data import AnomalyDetectionDataset
from torchvision import transforms
import torchvision
from torch.utils import data
from torch.nn.functional import normalize
import os
import torch
from simclr import SimCLR
from models.resnet import get_resnet, name_to_params
import numpy as np
 
def compute_features(dataloader, model):
  
    print('Compute features')
    model.eval()
    with torch.no_grad():
        features1 = []
        features2 = []
        
        for i, images in enumerate(dataloader):
            
            x1 = images[0][0].cuda()
            x2 = images[0][1].cuda()
           
            x1.requires_grad = False 
            x2.requires_grad = False
             
            z1 = model(x1)  
            p1= model.predictor(z1).cpu().numpy()
            z2 = model(x2)
            p2 = model.predictor(z2).cpu().numpy()
            
            features1.append(p1)
            features2.append(p2)
             
        features1 = np.row_stack(features1)
        features2 = np.row_stack(features2)
        
    return (features1, features2)

 

def compute_features_simple_2(dataloader, model):
  
    print('Compute features')
    model.eval()
    with torch.no_grad():
        features1 = []
        features2 = []

        for i, images in enumerate(dataloader):
            
            x1 = images[0][0].cuda()
            x2 = images[0][1].cuda()
 
            x1.requires_grad = False 
            x2.requires_grad = False
 
            z1 = model(x1).cpu().numpy() 
            z2 = model(x2).cpu().numpy()

            features1.append(z1)
            features2.append(z2)
         
    features1 = np.row_stack(features1)
    features2 = np.row_stack(features2)
 
    return (features1, features2)

def compute_features_simple_5(dataloader, model):
  
    print('Compute features')
    model.eval()
    with torch.no_grad():
        features1 = []
        features2 = []
        features3 = []
        features4 = []
        features5 = []

        for i, images in enumerate(dataloader):
            
            x1 = images[0][0].cuda()
            x2 = images[0][1].cuda()
            x3 = images[0][2].cuda()
            x4 = images[0][3].cuda()
            x5 = images[0][4].cuda()

            x1.requires_grad = False 
            x2.requires_grad = False
            x3.requires_grad = False 
            x4.requires_grad = False
            x5.requires_grad = False 
#
            z1 = model(x1).cpu().numpy() 
            z2 = model(x2).cpu().numpy()
            z3 = model(x3).cpu().numpy()
            z4 = model(x4).cpu().numpy()
            z5 = model(x5).cpu().numpy()

            features1.append(z1)
            features2.append(z2)
            features3.append(z3)
            features4.append(z4)
            features5.append(z5)
             

    features1 = np.row_stack(features1)
    features2 = np.row_stack(features2)
    features3 = np.row_stack(features3)
    features4 = np.row_stack(features4)
    features5 = np.row_stack(features5)

    return (features1, features2, features3, features4, features5)

def get_model(network, mp=None, ls=None, img_size=None, mem_dim=None, shrink_thres=0.0,opt=None):
    if network == "AE":
        model = AE(latent_size=ls, multiplier=mp, unc=False, img_size=img_size)
    elif network in ["siamese_CLRv2"]:
        if not opt.without_pre_trained:
            pth_path = "models/r50_1x_sk0.pth"
            model, _ = get_resnet(*name_to_params(pth_path))
            model.load_state_dict(torch.load(pth_path)['resnet'])
        else:
            print(" without pre-trained simCLRv2")
            model=None
        model = Siamese_CLR(latent_size=ls, img_size=img_size,model=model)
    elif network in ["barlow"]:
        if not opt.without_pre_trained:
            model =  torchvision.models.resnet50()
            pth_path = "models/barlow_resnet50_.pth"
            model.load_state_dict(torch.load(pth_path), strict=False)
            model.fc = torch.nn.Identity()
        else:
            print(" without pre-trained barlow")
            model=None
        model = Siamese_CLR(latent_size=ls, img_size=img_size,model=model)
    elif network == "AE-U":
        model = AE(latent_size=ls, multiplier=mp, unc=True, img_size=img_size)
    elif network == "MemAE":
        model = MemAE(latent_size=ls, multiplier=mp, img_size=img_size, mem_dim=mem_dim, shrink_thres=shrink_thres)
    else:
        raise Exception("Invalid Model Name!")

    model.cuda()
    return model


def get_loader(dataset, dtype, bs, img_size=512, workers=1, data_ratio=1, outlier_ratio=0.):
    DATA_PATH = '/home/xixi/Downloads/DDAD-main/Datasets/'
    transform = transforms.Compose([
         
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    if dataset == 'rsna':
        path = os.path.join(DATA_PATH, 'rsna-pneumonia-detection-challenge')

    elif dataset == 'vin':
        path = os.path.join(DATA_PATH, "vincxr")
    else:
        raise Exception("Invalid dataset: {}".format(dataset))

    dset = AnomalyDetectionDataset(main_path=path, transform=transform, mode=dtype, img_size=img_size,
                                   data_ratio=data_ratio, ar=outlier_ratio)

    train_flag = True if dtype == 'train' else False
    dataloader = data.DataLoader(dset, bs, shuffle=train_flag,
                                 drop_last=train_flag, num_workers=workers, pin_memory=False)

    return dataloader

def get_loader_siamese(network, dataset, dtype, bs, img_size, workers=1, data_ratio=1, outlier_ratio=0.,without_pre_trained=False):
    DATA_PATH = '/home/xixi/Downloads/DDAD-main/Datasets/'
    
    print("without_pre_trained is", without_pre_trained)
    if not without_pre_trained:
        aug = [
             transforms.Resize(256),
             transforms.RandomCrop(img_size),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize((0.5,), (0.5,)),
             lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x]
    else:
        aug = [
             transforms.Resize(256),
             transforms.RandomCrop(img_size),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize((0.5,), (0.5,))]
         
    if dataset == 'rsna':
        path = os.path.join(DATA_PATH, 'rsna-pneumonia-detection-challenge')

    elif dataset == 'vin':
        path = os.path.join(DATA_PATH, "vincxr")
    else:
        raise Exception("Invalid dataset: {}".format(dataset))

    if network in ["siamese", "siamese_CLRv2", "barlow"]:
        dset = AnomalyDetectionDataset(main_path=path, transform=TwoCropsTransform(transforms.Compose(aug), transforms.Compose(aug)), mode=dtype, img_size=img_size,
                                   data_ratio=data_ratio, ar=outlier_ratio)
        #dset = AnomalyDetectionDataset(main_path=path, transform=MultipleCropsTransform(transforms.Compose(weak_aug), transforms.Compose(weak_aug)), mode=dtype, img_size=img_size,
        #                          data_ratio=data_ratio, ar=outlier_ratio)

    else:
        transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])
        dset = AnomalyDetectionDataset(main_path=path, transform= transform,  mode=dtype, img_size=img_size,
                                  data_ratio=data_ratio, ar=outlier_ratio)

    train_flag = True if dtype == 'train' else False
    dataloader = data.DataLoader(dset, bs, shuffle=train_flag,
                                 drop_last=train_flag, num_workers=workers, pin_memory=False)

    return dataloader


 
# https://github.com/facebookresearch/simsiam/blob/a7bc1772896d0dad0806c51f0bb6f3b16d290468/main_simsiam.py#L225
class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, str_transform, weak_transform):
        self.str_transform = str_transform
        self.weak_transform = weak_transform

    def __call__(self, x):
        q = self.str_transform(x)
        k = self.weak_transform(x)
        return [q, k]
    
class MultipleCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self,  str_transform, weak_transform):
        self.str_transform = str_transform
        self.weak_transform = weak_transform
          
         

    def __call__(self, x):
        q = self.weak_transform(x)
        k = self.weak_transform(x)
        l = self.weak_transform(x)
        m = self.weak_transform(x)
        n = self.weak_transform(x)
        
        return [q, k,l,m,n]

class TenCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self,  str_transform, weak_transform):
        self.str_transform = str_transform
        self.weak_transform = weak_transform
          
         

    def __call__(self, x):
        q = self.weak_transform(x)
        k = self.weak_transform(x)
        l = self.weak_transform(x)
        m = self.weak_transform(x)
        n = self.weak_transform(x)
        qq = self.weak_transform(x)
        kk = self.weak_transform(x)
        ll = self.weak_transform(x)
        mm = self.weak_transform(x)
        nn = self.weak_transform(x)
         
        
        return [q, k,l,m,n, qq, kk, ll, mm, nn]
















def negative_cosin_similarity (p,z):
    
    p =  normalize(p, dim =1)
    z =  normalize(z, dim =1)
    return -(p*z).sum(dim=1).mean()


def similarity(p,z):
    """
    p: 1 x Feature_Size 
    z: BS x Feature_Size 
    """
    z = z.detach()
    p =  normalize(p, dim =1)
    p = p.expand(z.shape[0],p.shape[1])
    z =  normalize(z, dim =1)
    return -(p*z).sum(dim=1).mean()


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')
    
 

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.clim(0,1)

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names,fontsize=18)
        plt.yticks(tick_marks, target_names,fontsize=18)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=18)
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black",fontsize=18)


    #plt.tight_layout()
    plt.ylabel('True label' )
    plt.xlabel('Predicted label')
    plt.savefig(f'all_plots/new_vin_confusion_matrix.pdf', dpi=600)
    print("accuracy", accuracy, "misclassification", misclass)
    plt.show()



def plot_confusion_matrix_version2(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')
    
    with plt.style.context(['science', 'ieee']):
        fig, ax = plt.subplots(nrows=1, ncols=1,  sharex='col', sharey='row')
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.set(title=title, xlabel= "Predicted label", ylabel="True label")
        ax.title.set_fontsize(10)
        ax.xaxis.label.set_fontsize(10)
        ax.yaxis.label.set_fontsize(10)
        plt.colorbar()
        plt.clim(0,1000)
         
        if target_names is not None:
            tick_marks = np.arange(len(target_names))
            plt.xticks(tick_marks, target_names,fontsize=10)
            plt.yticks(tick_marks, target_names,fontsize=10)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


        thresh = cm.max() / 1.5 if normalize else cm.max() / 2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black", fontsize=10)
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black",fontsize=10)
    

        #plt.tight_layout()
         
        plt.savefig(f'all_plots/new_vin_confusion_matrix.pdf', dpi=600)
        print("accuracy", accuracy, "misclassification", misclass)
        plt.show()
