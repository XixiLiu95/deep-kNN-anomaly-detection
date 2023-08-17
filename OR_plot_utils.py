import matplotlib.pyplot as plt
import numpy as np


 
 
##### For different data ratio #####
outlier_ratio = np.array([0.0, 0.01, 0.02, 0.05, 0.07, 0.10])
data_ratio = 0.5
epoch = 99
K = np.array([1, 5, 10, 20])
dataset = "vin" # ["rsna", "vin"]
AUROC_values = np.zeros([len(K), len(outlier_ratio)])
AP_values = np.zeros([len(K), len(outlier_ratio)])
 
idx =0
k=1
index =0
network1 = "siamese_CLRv2_KNN"
network2= "barlow"
aug = 5
#for k in K:
#    index =0
#    for m in outlier_ratio:
#        respath = f'results/{dataset}/{k}_nearest/{data_ratio}_normal_{m}_outlier_epoch_{epoch}.txt'
#        with open(respath) as f: 
#            row_element = [float(a) for a in f.read().split(',')]
#        AUROC_values[idx, index] =  row_element[0]/100 
#        index = index +1
#    idx =idx+1
#      

for m in outlier_ratio:
     
    #respath1 = f'results/{dataset}/{k}_nearest/{m}_normal_{outlier_ratio}_outlier_epoch_{epoch}.txt'
    respath1 = f'results/{aug}_augmentations/{dataset}/{network1}/geometric_mean/{k}_nearest/{data_ratio}_normal_{m}_outlier_epoch_{epoch}.txt'
    with open(respath1) as f: 
                    row_element1 = [float(a) for a in f.read().split(',')]
    respath2 = f'results/{aug}_augmentations/{dataset}/{network2}/geometric_mean/{k}_nearest/{data_ratio}_normal_{m}_outlier_epoch_{epoch}.txt'
    with open(respath2) as f: 
                    row_element2 = [float(a) for a in f.read().split(',')]
    AUROC_values[0, index] =  row_element1[0]/100 
    AUROC_values[1, index] =  row_element2[0]/100 
     
    index = index +1

with plt.style.context(['science', 'ieee']):
    fig, ax = plt.subplots(nrows=1, ncols=1,  sharex='col', sharey='row')
         
    ax.plot(outlier_ratio, AUROC_values[0,:], '*-', color='darkorange', label= 'SimCLRv2* (Ours)', lw=0.2, markersize=1)
    ax.plot(outlier_ratio, AUROC_values[1,:], '*-', color='dodgerblue', label= 'Barlow* (Ours)', lw=0.2, markersize=1)
    #ax.plot(outlier_ratio, AUROC_values[1,:], '*-', color='dodgerblue', label= 'k $=5$', lw=0.2, markersize=3)
    #ax.plot(outlier_ratio, AUROC_values[2,:] , '*-', color= 'chocolate', label= 'k$=10$', lw=0.2, markersize=3)
    #ax.plot(outlier_ratio, AUROC_values[3,:] , '*-', color= 'olivedrab', label= 'k$=20$', lw=0.2, markersize=3)
    #ax.axhline(y = 0.559, ls="--",color = 'purple', label = 'AE $($OR$=0)$', linewidth=0.5)
     
    #ax.axhline(y = 0.873, ls="--",color = 'olive', label = 'DDAD-AE-U $($AR$=0)$', linewidth=0.5)
    #ax.axhline(y = 0.798, ls="--",color = 'salmon', label = 'f-AnoGAN $($AR$=0)$', linewidth=0.5)
    ax.axhline(y = 0.763, ls="--",color = 'olive', label = 'DDAD-AE-U $($AR$=0)$', linewidth=0.5)
    ax.axhline(y = 0.743, ls="--",color = 'salmon', label = 'f-AnoGAN $($AR$=0)$', linewidth=0.5)
    #ax.plot(outlier_ratio, AUROC_values[ 3,:], '*-', color= 'dodgerblue', label= 'OpenImage-O', lw=0.2, markersize=3)
    for index in range(len(outlier_ratio)):
        ax.text(outlier_ratio[index],  AUROC_values[0,index], round(AUROC_values[0, index],3), size=6)
        ax.text(outlier_ratio[index],  AUROC_values[1,index], round(AUROC_values[1, index],3), size=6)
    ax.legend(fontsize=5)
    ax.set_title('VinBigData')
    #ax.set_title('RSNA')
    ax.set_xlabel('Anomaly Ratio(AR)' )
    ax.set_ylabel('AUROC')
    plt.ylim([0.7, 1.0])
    plt.xlim([-0.005, 0.11])
    plt.show()
    fig.savefig(f'all_plots/{dataset}_OR_AUROC.pdf', dpi=600)
    plt.close()
    #fig, ax = plt.subplots(nrows=1, ncols=1,  sharex='col', sharey='row')
    #ax.plot(M, FPR95_values[ 0,:], '*-', color='darkorange', label= 'Imagenet-O', lw=0.2, markersize=3)
    #ax.plot(M, FPR95_values[ 1,:] , '*-', color= 'chocolate', label= 'iNaturalist', lw=0.2, markersize=3)
    #ax.plot(M, FPR95_values[ 2,:] , '*-', color= 'olivedrab', label= 'Texture', lw=0.2, markersize=3)
    #ax.plot(M, FPR95_values[ 3,:], '*-', color= 'dodgerblue', label= 'OpenImage-O', lw=0.2, markersize=3)
    #ax.legend(fontsize=6)
    #ax.set_title('ResNet-50-D')
    #ax.set_xlabel('Top $M$ Classes' )
    #ax.set_ylabel('FPR95' )
    #plt.ylim([30, 100])
    #plt.show()
    #fig.savefig(f'all_plots/{arch}_M_FPR95.pdf', dpi=600)