import yaml
from trainer import *
from test import  test_rec
#from test_backbone import test_rec # for the case that we only use the pre-trained backbones as the feature extractor.
from argparse import ArgumentParser
import random
#torch.backends.cudnn.benchmark = True
seed=1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', dest='config', type=str, default="cfgs/RSNA_siamese.yaml")  # config file
    parser.add_argument('--mode', dest='mode', type=str, default="train")
    parser.add_argument('--epoch', dest='epoch', type=int, default=100)
    parser.add_argument('--data_ratio', dest='data_ratio', type=float, default=0.5)
    parser.add_argument('--outlier_ratio', dest='outlier_ratio', type=float, default=0.)
    parser.add_argument('--k', dest='k', type=int, default=1)
    parser.add_argument('--with_siamese', dest='with_siamese', help='use siasiam network',  action="store_true")
    parser.add_argument('--without_pre_trained', dest='without_pre_trained', help='do not add pre-trained simCLRv2 as enoder', action="store_true" )
    parser.add_argument('--normalization', dest='normalization', help='feature normalization', action="store_true" )
    parser.add_argument('--geometric_mean', dest='geometric_mean', action="store_true" )
    parser.add_argument('--augmentation', dest='augmentation', type =int, default=5)
    

    
    opt = parser.parse_args()

    with open(opt.config, "r") as f:
        cfgs = yaml.safe_load(f)

    torch.cuda.set_device(cfgs["Exp"]["gpu"])

    out_dir = cfgs["Exp"]["out_dir"]
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if opt.mode =="train":
        train_module(cfgs, opt, out_dir)
    elif opt.mode == "test":
        test_rec(cfgs, opt)
    else:
        raise Exception("Invalid mode: {}".format(opt.mode))