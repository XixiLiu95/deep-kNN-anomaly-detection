import matplotlib.pyplot as plt
import numpy as np


 
 
##### For different data ratio #####
data_ratio = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
outlier_ratio = 0.0
epoch = 99
dataset = "rsna" # ["rsna", "vin"]
AUROC_values = np.zeros([2, len(data_ratio)])
AP_values = np.zeros([2, len(data_ratio)])
k=1
index =0
network1 = "siamese_CLRv2_KNN"
network2= "barlow"
aug = 5
for m in data_ratio:
     
    #respath1 = f'results/{dataset}/{k}_nearest/{m}_normal_{outlier_ratio}_outlier_epoch_{epoch}.txt'
    respath1 = f'results/{aug}_augmentations/{dataset}/{network1}/geometric_mean/{k}_nearest/{m}_normal_{outlier_ratio}_outlier_epoch_{epoch}.txt'
    with open(respath1) as f: 
                    row_element1 = [float(a) for a in f.read().split(',')]
    respath2 = f'results/{aug}_augmentations/{dataset}/{network2}/geometric_mean/{k}_nearest/{m}_normal_{outlier_ratio}_outlier_epoch_{epoch}.txt'
    with open(respath2) as f: 
                    row_element2 = [float(a) for a in f.read().split(',')]
    AUROC_values[0, index] =  row_element1[1]/100 
    AUROC_values[1, index] =  row_element2[1]/100 
     
    index = index +1
      


with plt.style.context(['science', 'ieee']):
    fig, ax = plt.subplots(nrows=1, ncols=1,  sharex='col', sharey='row')
         
    ax.plot(data_ratio, AUROC_values[0,:], '*-', color='darkorange', label= 'SimCLRv2*(Ours)', lw=0.2, markersize=1)
    ax.plot(data_ratio, AUROC_values[1,:] , '*-', color= 'dodgerblue', label= 'Barlow*(Ours)', lw=0.2, markersize=1)
    #ax.plot(M, AUROC_values[ 2,:] , '*-', color= 'olivedrab', label= 'Texture', lw=0.2, markersize=3)
    #ax.plot(data_ratio, AUROC_values[ 1,:], '*-', color= 'dodgerblue', label= 'baseline-AE', lw=0.2, markersize=3)
    for index in range(len(data_ratio)):
            #if index ==0:
            #    ax.text(data_ratio[index],  AUROC_values[0,index], round(AUROC_values[0,index],3), size=6, horizontalalignment='right')
            #    #ax.text(data_ratio[index],  AUROC_values[1,index], round(AUROC_values[1,index],3), size=6, horizontalalignment='left')
            #else:
                ax.text(data_ratio[index],  AUROC_values[0,index], round(AUROC_values[0,index],3), size=6, horizontalalignment='left')
                ax.text(data_ratio[index],  AUROC_values[1,index], round(AUROC_values[1,index],3), size=6, horizontalalignment='left')
    #ax.axhline(y = 0.763, ls="--",color = 'salmon', label = 'f-AnoGAN $($DR$=0.5)$', linewidth=0.5)
    #ax.axhline(y = 0.743, ls="--",color = 'olive', label = 'DDAD-AE-U $($DR$=0.5)$', linewidth=0.5)
    #ax.scatter( 0.49, 0.763, marker='P',c = 'salmon', label = 'f-AnoGAN $($DR$=0.49)$',s=1)
    #ax.scatter( 0.49, 0.743, marker='X',c = 'olive', label = 'DDAD-AE-U $($DR$=0.49)$', s=1)
    ax.legend(fontsize=6)
    ax.set_title('RSNA')
    #ax.set_title('VinBigData')
    ax.set_xlabel('Data Ratio(DR)' )
    ax.set_ylabel('Average Precision' )
    plt.ylim([0.7, 1.0])
    plt.xlim([0, 1])
    plt.show()
    fig.savefig(f'all_plots/{aug}_augmentations_{dataset}_DR_AP.pdf', dpi=600)
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