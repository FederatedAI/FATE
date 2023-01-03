import pandas as pd


def stat_method(df, stat_func, *args, index=None, **kwargs) -> pd.Series:
    stat_ret = getattr(df, stat_func)(*args, **kwargs)
    dtype = str(stat_ret.dtype.to_torch_dtype()).split(".", -1)[-1]
    if not kwargs.get("axis", 0):
        if index:
            return pd.Series(stat_ret, index=index, dtype=dtype)
        else:
            return pd.Series(stat_ret, dtype=dtype)
    else:
        return pd.Series(stat_ret, dtype=dtype)
