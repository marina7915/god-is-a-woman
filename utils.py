import torch
import random, os
import numpy as np
from torch import nn

class Definitions:
    optimizers = {
        "Adam": torch.optim.Adam,
        "SGD": torch.optim.SGD,
        "AdamW": torch.optim.AdamW
    }
    schedulers = {
        "CosineAnnealingLR": torch.optim.lr_scheduler.CosineAnnealingLR,
        "ExponentialLR": torch.optim.lr_scheduler.ExponentialLR,
        "ReduceLROnPlateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "OneCycleLR": torch.optim.lr_scheduler.OneCycleLR,
        "CosineAnnealingWarmRestarts": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    }
    activations = {
        "relu": nn.ReLU(),
        "leaky_relu": nn.LeakyReLU(negative_slope=0.01),
        "elu": nn.ELU(alpha=1.0),
        "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh(),
        "softplus": nn.Softplus(),
    }

def fix_seed(seed: int):

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True