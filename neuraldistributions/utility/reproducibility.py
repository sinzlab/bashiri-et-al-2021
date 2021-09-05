import numpy as np
import torch


def set_random_seed(seed):
    """
    Sets all random seeds
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def print_kwargs(orig_fun):
    """
    prints the keyword arguments passed to the function
    """
    def wrapper(*args, **kwargs):
        for k, v in kwargs.items():
            print("{} = {}".format(k, v))
        return orig_fun(*args, **kwargs)
    
    return wrapper