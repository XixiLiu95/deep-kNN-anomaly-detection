import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

 
 
 
K = np.array([1, 2, 5, 10, 20])
dataset = "rsna" # ["rsna", "vin"]
epoch  = 99
AUROC_values = np.zeros([2, len(K)])
FPR95_values = np.zeros([2, len(K)])
 
 
index =0
for k in K:
    
    respath = f'results/{dataset}/{k}_nearest/0.5_normal_0.0_outlier_epoch_{epoch}.txt'
    with open(respath) as f: 
                row_element = [float(a) for a in f.read().split(',')]
    AUROC_values[0, index] = row_element[0]/100
    FPR95_values[0, index] = row_element[1]/100
    index +=1

with plt.style.context(['science', 'ieee']):
    fig, ax = plt.subplots(nrows=1, ncols=1,  sharex='col', sharey='row')
         
    
    plot1=ax.plot(K, AUROC_values[0,:], '*-', color='dodgerblue', label= 'AUROC', lw=0.2, markersize=3)
    ax2=ax.twinx()
    plot2= ax2.plot(K, FPR95_values[0,:], '*-', color='chocolate', label= 'AP', lw=0.2, markersize=3)
     
    for index in range(len(K)):
        if index ==1:
            ax.text(K[index],  AUROC_values[0,index], round(AUROC_values[0,index],3), size=8, horizontalalignment='center')
        else:
            ax.text(K[index],  AUROC_values[0,index], round(AUROC_values[0,index],3), size=8, horizontalalignment='right')
    for index in range(len(K)):
        ax2.text(K[index],  FPR95_values[0,index], round(FPR95_values[0,index],3), size=8)
    labs = [l.get_label() for l in plot1+plot2]
    ax.legend(plot1+plot2,labs,fontsize=8, loc=0)
    #ax2.legend(loc=0)
    ax.set_title('RSNA')
    ax.set_xlabel('$k$-Nearest-Neighbor' )
    ax.set_ylabel('AUROC' )
    ax2.set_ylabel('AP' )
    ax.set_ylim([0.87, 0.88])
    ax2.set_ylim([0.84, 0.89])
    plt.xlim([-2 ,24])
    plt.show()
    fig.savefig(f'all_plots/new_{dataset}_K_sensitivity_AUROC.pdf', dpi=600)
    plt.close()