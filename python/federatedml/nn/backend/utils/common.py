import torch as t
import numpy as np
import random

ML_PATH = 'federatedml.nn'
HOMOMODELMETA = "HomoNNMeta"
HOMOMODELPARAM = "HomoNNParam"


def global_seed(seed):

    # set random seed of torch
    t.manual_seed(seed)
    t.cuda.manual_seed_all(seed)
    t.backends.cudnn.deterministic = True


def get_homo_model_dict(param, meta):
    return {HOMOMODELPARAM: param,  # param
            HOMOMODELMETA: meta}  # meta


def get_homo_param_meta(model_dict):
    return model_dict.get(HOMOMODELPARAM), model_dict.get(HOMOMODELMETA)
