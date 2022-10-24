import torch as t
import numpy as np
import random

ML_PATH = 'federatedml.nn'
HOMOMODELMETA = "HomoNNMeta"
HOMOMODELPARAM = "HomoNNParam"


def global_seed(seed):
    # set all random seeds
    # set random seed of torch
    t.manual_seed(seed)
    t.cuda.manual_seed_all(seed)
    t.backends.cudnn.deterministic = True
    # np & random
    np.random.seed(seed)
    random.seed(seed)


def get_homo_model_dict(param, meta):
    return {HOMOMODELPARAM: param,  # param
            HOMOMODELMETA: meta}  # meta


def get_homo_param_meta(model_dict):
    return model_dict.get(HOMOMODELPARAM), model_dict.get(HOMOMODELMETA)


def parent_attr_check(obj, attr):
    if not hasattr(obj, attr):
        raise ValueError('{} is None, please make sure that you call the __init__ function of the super class')
        
