#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import functools

import numpy as np
import pandas as pd
import torch
from fate.arch import tensor
from fate.arch.context.io.data import df

from .._dataframe import DataFrame
from ..storage import Index, ValueStore
from ._json_schema import build_schema, parse_schema


def _serialize_distributed(ctx, data):
    """
    index, match_id, label, weight, values
    """
    # TODO: tensor does not provide method to get raw values directly, so we use .storages.blocks first
    schema = build_schema(data)

    tensors = [data.label, data.weight]
    tensor_concat = None
    for t in tensors:
        if not t:
            continue

        """
        distributed tensor
        """
        t = t.storage.blocks
        if tensor_concat is None:
            tensor_concat = t
        else:
            tensor_concat = tensor_concat.join(tensor, lambda t1, t2: torch.concat([t1, t2], -1))

    # TODO: modify here before releasing
    if data.values is not None:
        if isinstance(data.values, ValueStore):
            value_concat = data.values.values
            if tensor_concat is not None:
                value_concat = tensor_concat.join(
                    value_concat, lambda t1, t2: np.concatenate([t1.to_local().data.numpy(), t2.to_numpy()], axis=-1)
                )
        else:
            value_concat = data.values.storage.blocks.mapValues(lambda t: t.to_local().data)
            if tensor_concat is not None:
                value_concat = tensor_concat.join(
                    value_concat, lambda t1, t2: np.concatenate([t1.to_local().data.numpy(), t2.numpy()], axis=-1)
                )

    else:
        value_concat = tensor_concat
        if value_concat is not None:
            value_concat = value_concat.mapValues(lambda t: t.to_local.data.numpy())

    tensor_concat = value_concat

    index = Index.aggregate(data.index.values)
    if tensor_concat is None:
        """
        data only has index
        """
        serialize_data = index
    else:

        def _flatten(index: list, t):
            flatten_ret = []
            for (_id, block_index), _t in zip(index, t):
                flatten_ret.append([_id] + _t.tolist())

            return flatten_ret

        serialize_data = index.join(tensor_concat, _flatten)

    serialize_data.schema = schema
    return serialize_data


def serialize(ctx, data):
    if isinstance(data, df.Dataframe):
        data = data.data

    return _serialize_distributed(ctx, data)


def deserialize(ctx, data):
    recovery_schema, global_ranks, block_partition_mapping, column_info = parse_schema(data.schema)

    def _recovery_index(kvs):
        """
        TODO: index should provider deserialize method, implement it here for convenient
        """
        start_index = column_info["index"]["start_idx"]
        indexes = []
        for key, values in kvs:
            for offset, v in enumerate(values):
                indexes.append((v[start_index], (key, offset)))

        return indexes

    def _recovery_tensor(value, tensor_info=None):
        start_index = tensor_info["start_idx"]
        end_index = tensor_info["end_idx"]
        dtype = tensor_info["type"]

        ret_tensor = []
        for v in value:
            ret_tensor.append(v[start_index : end_index + 1])

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
        return tensor.distributed_tensor(ctx, tensor_list, partitions=len(tensor_list))

    ret_dict = dict()
    ret_dict["index"] = Index(
        ctx=ctx,
        distributed_index=data.mapPartitions(_recovery_index, use_previous_behavior=False),
        block_partition_mapping=block_partition_mapping,
        global_ranks=global_ranks,
    )

    tensor_keywords = ["weight", "label", "values"]
    for keyword in tensor_keywords:
        if keyword in column_info:
            if keyword == "values" and column_info["values"]["source"] == "fate.dataframe.value_store":
                continue
            _recovery_func = functools.partial(_recovery_tensor, tensor_info=column_info[keyword])
            tensors = [tensor for key, tensor in sorted(list(data.mapValues(_recovery_func).collect()))]
            ret_dict[keyword] = _to_distributed_tensor(tensors)

    if "values" in column_info and column_info["values"]["source"] == "fate.dataframe.value_store":
        _recovery_df_func = functools.partial(
            _recovery_distributed_value_store, value_info=column_info["values"], header=recovery_schema["header"]
        )
        ret_dict["values"] = ValueStore(ctx, data.mapValues(_recovery_df_func), recovery_schema["header"])

    return DataFrame(ctx, recovery_schema, **ret_dict)
