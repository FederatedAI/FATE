import numpy as np
import pandas as pd
import torch
from fate.arch import tensor


def arith_method(lhs, rhs, op):
    if isinstance(rhs, pd.Series):
        rhs = tensor.tensor(torch.tensor(rhs.tolist(), dtype=getattr(torch, str(rhs.dtype))))
    elif isinstance(rhs, (int, float, np.int, np.int32, np.int64, np.float, np.float32, np.float64)):
        pass
    elif hasattr(rhs, "values"):
        rhs = rhs.values
    else:
        raise ValueError(f"{op.__name__} between DataFrame and {type(rhs)} is not supported")

    return op(lhs, rhs)








