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
from fate.arch.context.io.data import df

from .._dataframe import DataFrame
from ._json_schema import build_schema, parse_schema
from ..manager import DataManager


def _serialize(ctx, data):
    # test
    # data = data + 1
    # data = data * 2
    # data = data - 3
    # data = 1 - data
    # data = data / 10
    # data.label
    # data.weight
    # data = data["x1"] * data["x0"]
    # data = data[["x0", "x1"]] * 3
    # pd_df = data.as_pd_df()
    # print (pd_df)
    # empty_df = data.create_frame(with_label=False, with_weight=False)
    # empty_df["x20"] = 1.0
    # df = data.create_frame()
    # df["g"] = 100
    # df["h"] = 200
    # new_df = DataFrame.hstack([data, df])
    # print(data.drop(data).shape)
    # print (new_df.as_pd_df())
    # print(DataFrame.hstack([data, empty_df]).as_pd_df())
    # print(DataFrame.vstack([data, data * 3]).as_pd_df())
    # data["x20"] = 1.0
    # data["x21"] = [1, 2]
    # data[["x22", "x23"]] = [3, 4]
    # data["x23"] = data["x0"]
    # data["x1"] = 1.0
    # data["x2"] = [1, 2]
    # data[["x3", "x4"]] = [3, 4]
    # data["x5"] = data["x0"]
    # apply_df = data.apply_row(lambda value: [1, {1:2, 2:3}])
    # print(apply_df.as_pd_df())
    # print(data.sigmoid().as_pd_df())
    """
    index, match_id, label, weight, values
    """
    # TODO: tensor does not provide method to get raw values directly, so we use .storages.blocks first
    schema = build_schema(data)

    from ..ops._transformer import transform_block_to_list
    serialize_data = transform_block_to_list(data.block_table, data.data_manager)
    serialize_data.schema = schema
    return serialize_data


def serialize(ctx, data):
    if isinstance(data, df.Dataframe):
        data = data.data

    return _serialize(ctx, data)


def deserialize(ctx, data):
    fields, partition_order_mappings = parse_schema(data.schema)

    data_manager = DataManager.deserialize(fields)
    from ..ops._transformer import transform_list_to_block

    block_table = transform_list_to_block(data, data_manager)

    return DataFrame(ctx, block_table, partition_order_mappings, data_manager)
