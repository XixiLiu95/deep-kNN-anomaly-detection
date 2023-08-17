import torch.nn as nn
import torchvision.models as models
class Siamese_CLR(nn.Module):
    def __init__(self, latent_size, multiplier=1, img_size=64, model=None):
        super(Siamese_CLR, self).__init__()
        
        self.fm = img_size // 16
        self.adaptor =  nn.Linear(latent_size,  int(latent_size/2))
        self.pred =  nn.Linear(int(latent_size/2),  int(latent_size/2))
        self.model = model
        if self.model is None:
                self.mp = multiplier
                self.encoder = nn.Sequential(
                    nn.Conv2d(1, int(16 * multiplier), 4, 2, 1, bias=False),
                    nn.BatchNorm2d(int(16 * multiplier)),
                    nn.ReLU(True),
                    nn.Conv2d(int(16 * multiplier),
                              int(32 * multiplier), 4, 2, 1, bias=False),
                    nn.BatchNorm2d(int(32 * multiplier)),
                    nn.ReLU(True),
                    nn.Conv2d(int(32 * multiplier),
                              int(64 * multiplier), 4, 2, 1, bias=False),
                    nn.BatchNorm2d(int(64 * multiplier)),
                    nn.ReLU(True),
                    nn.Conv2d(int(64 * multiplier),
                              int(64 * multiplier), 4, 2, 1, bias=False),
                    nn.BatchNorm2d(int(64 * multiplier)),
                    nn.ReLU(True)
                )
                self.linear_enc = nn.Sequential(
                    nn.Linear(int(64 * multiplier) * self.fm * self.fm, 2048),
                    nn.BatchNorm1d(2048),
                    nn.ReLU(True),
                    #nn.Linear(2048, latent_size),
                )

        else:                                                       
            self.encoder = model
            #self.adaptor = nn.Identity() # this is added for zero-shot anomaly detection, i.e. test with raw feature extracted from backbones.

            for param in self.encoder.parameters():
                param.requires_grad = False

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
    
    
    def forward(self, x):
        lat_rep = self.feature(x)
         
        return lat_rep

    def feature(self, x):
        if  self.model is None:
            lat_rep = self.encoder(x)
            lat_rep = lat_rep.view(lat_rep.size(0), -1)
            lat_rep = self.linear_enc(lat_rep)
            lat_rep = self.adaptor(lat_rep)
        else:
            lat_rep = self.encoder(x)
            lat_rep = self.adaptor(lat_rep)

        return lat_rep

    def predictor(self, lat_rep=None): 
        out = self.pred(lat_rep)
        return out
