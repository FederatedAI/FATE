import functools
import numpy as np
import pandas as pd
import torch
from fate.arch import tensor
from ._json_schema import build_schema, parse_schema
from .._dataframe import DataFrame
from ..storage import Index, ValueStore


def serialize(ctx, data):
    """
    index, match_id, label, weight, values
    """
    # TODO: tensor does not provide method to get raw values, so we use to local first
    data = data.to_local()
    global_ranks = data.index.global_ranks
    schema = build_schema(data, global_ranks)

    tensors = [data.label, data.weight]
    tensor_concat = None
    for tensor in tensors:
        if not tensor:
            continue
        tensor = tensor.storage.data
        if tensor_concat is None:
            tensor_concat = tensor
        else:
            tensor_concat = torch.concat([tensor_concat, tensor], -1)

    # TODO: modify here before releasing
    value_concat = None
    if data.values is not None:
        if isinstance(data.values, pd.DataFrame):
            value_concat = data.values.to_numpy()
        else:
            value_concat = data.values.storage.data.numpy()

        if tensor_concat is not None:
            value_concat = np.concatenate([tensor_concat.numpy(), value_concat], axis=-1)
    else:
        value_concat = tensor_concat

    if value_concat is not None:
        tensor_concat = ctx.computing.parallelize(
            [value_concat.tolist()],
            include_key=False,
            partition=1
        )
    """
    data only has index
    """
    if tensor_concat is None:
        serialize_data = data.index.mapValues(lambda pd_index: pd_index.tolist())
    else:
        def _flatten(index: pd.Index, t: list):
            index = index.tolist()
            # t = t.tolist()
            flatten_ret = []
            for _id, _tensor in zip(index, t):
                flatten_ret.append([_id] + _tensor)

            return flatten_ret

        serialize_data = data.index.to_local().values.join(tensor_concat, _flatten)

    serialize_data.schema = schema
    data_dict = dict(data=list(serialize_data.collect()),
                     schema=schema)
    return data_dict
    # return serialize_data


def deserialize(ctx, data):
    local_data = data["data"]
    schema = data["schema"]
    data = ctx.computing.parallelize(
            local_data,
            include_key=True,
            partition=1
        )
    data.schema = schema

    recovery_schema, global_ranks, column_info = parse_schema(data.schema)

    def _recovery_index(value):
        """
        TODO: index should provider deserialize method, implement it here for convenient
        """
        start_index = column_info["index"]["start_idx"]
        indexes = [v[start_index] for v in value]

        return pd.Index(indexes)

    def _recovery_tensor(value, tensor_info=None):
        start_index = tensor_info["start_idx"]
        end_index = tensor_info["end_idx"]
        dtype = tensor_info["type"]

        ret_tensor = []
        for v in value:
            ret_tensor.append(v[start_index: end_index + 1])

        return torch.tensor(ret_tensor, dtype=getattr(torch, dtype))

    def _recovery_distributed_value_store(value, value_info, header):
        start_index = value_info["start_idx"]
        end_index = value_info["end_idx"]

        filter_value = []
        for v in value:
            filter_value.append(v[start_index : end_index + 1])

        df = pd.DataFrame(filter_value, columns=header)

        return df


    def _to_distributed_tensor(tensor_list):
        return tensor.distributed_tensor(
            ctx, tensor_list, partitions=len(tensor_list)
        )

    ret_dict = dict()
    ret_dict["index"] = Index(ctx=ctx,
                              distributed_index=data.mapValues(_recovery_index),
                              global_ranks=global_ranks)

    tensor_keywords = ["weight", "label", "values"]
    for keyword in tensor_keywords:
        if keyword in column_info:
            if keyword == "values" and column_info["values"]["source"] == "pd.dataframe":
                continue
            _recovery_func = functools.partial(
                _recovery_tensor,
                tensor_info=column_info[keyword]
            )
            tensors = [tensor for key, tensor in sorted(list(data.mapValues(_recovery_func).collect()))]
            ret_dict[keyword] = _to_distributed_tensor(tensors)

    if "values" in column_info and column_info["values"]["source"] == "pd.dataframe":
        _recovery_df_func = functools.partial(
            _recovery_distributed_value_store,
            value_info=column_info["values"],
            header=recovery_schema["header"]
        )
        ret_dict["values"] = ValueStore(
            ctx,
            data.mapValues(_recovery_df_func),
            recovery_schema["header"]
        )

    return DataFrame(ctx, recovery_schema, **ret_dict)
