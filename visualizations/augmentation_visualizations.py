import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data_ratio = 0.5
outlier_ratio = 0.0
epoch = 99
augs = [2,5,10]
dataset = "vin" # ["rsna", "vin"]
AUROC_values = np.zeros([2, len(augs)])
AP_values = np.zeros([2, len(augs)])
k=1
index =0
network1 = "siamese_CLRv2_KNN"
network2= "barlow"
 
for m in augs:
     
    #respath1 = f'results/{dataset}/{k}_nearest/{m}_normal_{outlier_ratio}_outlier_epoch_{epoch}.txt'
    respath1 = f'results/{m}_augmentations/{dataset}/{network1}/geometric_mean/{k}_nearest/{data_ratio}_normal_{outlier_ratio}_outlier_epoch_{epoch}.txt'
    with open(respath1) as f: 
                    row_element1 = [float(a) for a in f.read().split(',')]
    respath2 = f'results/{m}_augmentations/{dataset}/{network2}/geometric_mean/{k}_nearest/{data_ratio}_normal_{outlier_ratio}_outlier_epoch_{epoch}.txt'
    with open(respath2) as f: 
                    row_element2 = [float(a) for a in f.read().split(',')]
    AUROC_values[0, index] =  row_element1[0]/100 
    AUROC_values[1, index] =  row_element2[0]/100 
    AP_values[0, index] =  row_element1[1]/100 
    AP_values[1, index] =  row_element2[1]/100 
     
    index = index +1

AUROC_data = pd.DataFrame({
"$aug=2$ ": AUROC_values[:,0],
#"$\gamma=0.2$ ": AUROC_values[:,1],
"$aug=5$ ":  AUROC_values[:,1],
#"$\gamma=0.4$ ": AUROC_values[:,3],
#"$\gamma=0.5$ ": AUROC_values[:,2],
#"$\gamma=0.6$ ": AUROC_values[:,5],
#"$\gamma=0.7$ ": AUROC_values[:,6],
#"$\gamma=0.8$ ": AUROC_values[:,7],
"$aug=10$ ": AUROC_values[:,2]},
#"$\gamma=1$ "  : AUROC_values[:,9]},
index=["simCLRv2* (Ours)", "Barlow* (Ours)"])

FPR95_data = pd.DataFrame({
"$aug=2$ ": AP_values[:,0],
#"$\gamma=0.2$ ": FPR95_values[:,1],
"$aug=5$ ": AP_values[:,1],
#"$\gamma=0.4$ ": FPR95_values[:,3],
#"$\gamma=0.5$ ": FPR95_values[:,4],
#"$\gamma=0.6$ ": FPR95_values[:,5],
#"$\gamma=0.7$ ": FPR95_values[:,6],
#"$\gamma=0.8$ ": FPR95_values[:,7],
"$aug=10$ ": AP_values[:,2]},
#"$\gamma=1$ ":   FPR95_values[:,9]},
index=["simCLRv2* (Ours)", "Barlow* (Ours)"])





with plt.style.context(['science', 'ieee']):
     
    ax = AUROC_data.plot.bar(rot=0,color=['royalblue','cornflowerblue','dodgerblue'])
    ax.legend(fontsize=6)
    ax.set_title('VinBigData')
    #ax.set_title('VinBigData')
    ax.set_ylabel('AUROC' )
    plt.ylim([0.7, 1])
    plt.savefig(f'all_plots/{dataset}_aug_AUROC.pdf', dpi=600)
    plt.show()
     
    plt.close()
    
    
    ax = FPR95_data.plot.bar(rot=0,color=['royalblue','cornflowerblue','dodgerblue'])
    ax.legend(fontsize=6)
    ax.set_title('VinBigData')
    ax.set_ylabel('Average Precision' )
    plt.ylim([0.7, 1])
    plt.savefig(f'all_plots/{dataset}_aug_AP.pdf', dpi=600)
 
    plt.show()
   