from models.entropy_loss import EntropyLossEncap
from models.autoencoder import AE
from models.memory_ae import MemAE
from models.siamese import Siamese
 
from models.siamese_CLR import Siamese_CLR
from models.siamese_barlow import Siamese_CLR_barlow
from models.utils import get_model, get_loader, get_loader_siamese,  negative_cosin_similarity, compute_features
from models.resnet import get_resnet, name_to_params
 