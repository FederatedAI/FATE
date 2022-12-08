import torch as t
import numpy as np
import tempfile

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


# read model from model bytes
def recover_model_bytes(model_bytes):
    with tempfile.TemporaryFile() as f:
        f.write(model_bytes)
        f.seek(0)
        model_dict = t.load(f)

    return model_dict


def get_torch_model_bytes(model_dict):

    with tempfile.TemporaryFile() as f:
        t.save(model_dict, f)
        f.seek(0)
        model_saved_bytes = f.read()

        return model_saved_bytes
