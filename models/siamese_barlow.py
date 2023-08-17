import torch.nn as nn
import torchvision.models as models
import torch
# some of code is borrowed from https://github.com/facebookresearch/barlowtwins/blob/8e8d284ca0bc02f88b92328e53f9b901e86b4a3c/main.py#L180

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class Siamese_CLR_barlow(nn.Module):
    def __init__(self, latent_size, multiplier=1,  model=None, lambd=0.005):
        super(Siamese_CLR_barlow, self).__init__()
        
        self.pred =  nn.Sequential(nn.Linear(latent_size,  int(latent_size/2)),
                                   nn.Linear(int(latent_size/2),  int(latent_size/4)),
                                   nn.Linear(int(latent_size/4),  int(latent_size/16)))
                                                             
        self.lambd = lambd
                                  
                                    
        self.encoder = model
        for param in self.encoder.parameters():
            param.requires_grad = False

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(int(latent_size/16), affine=False)

         

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    
    def forward(self, x1, x2):
        lat_rep_1 = self.feature(x1)
        lat_rep_2 = self.feature(x2)
        z1 = self.pred(lat_rep_1)
        z2 = self.pred(lat_rep_2)
        # empirical mean 
        #empirical_mean = torch.mean(z1-z2, dim=0)
        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2) 
        #loss = torch.log(torch.diagonal(c)).sum() + ((1+empirical_mean.pow_(2))/(2*torch.diagonal(c).pow_(2))).sum()-0.5
        # sum the cross-correlation matrix between all gpus
        c.div_(z1.shape[0])
        #temp1 = torch.diagonal(c).sum()
        #temp2 = torch.diagonal(c).pow_(2).sum()
        #on_diag = temp1 / torch.sqrt(temp2*z1.shape[0])
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = -on_diag + self.lambd * off_diag

        return loss

    def feature(self, x):
        lat_rep = self.encoder(x)
        return lat_rep

    def predictor(self, lat_rep=None): 
        out = self.pred(lat_rep)
        return out