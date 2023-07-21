from fate.arch.dataframe import DataFrame
import numpy as np
import pandas as pd
import torch as t


def _process_dataframe(df):
    result_dict = {}
    
    for column in df.columns:
        unique_values = df[column].unique()
        sorted_values = sorted(unique_values)
        result_dict[column] = sorted_values
    
    return result_dict


def binning(data: DataFrame, max_bin=32):

    quantile = [i / max_bin for i in range(1, max_bin)]
    quantile_values = data.quantile(quantile)
    result_dict = _process_dataframe(quantile_values)

    return result_dict