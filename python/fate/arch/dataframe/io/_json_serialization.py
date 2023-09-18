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

from .._dataframe import DataFrame
from ..manager import DataManager
from ._json_schema import build_schema, parse_schema


def _serialize(ctx, data):
    """
    index, match_id, label, weight, values
    """
    schema = build_schema(data)

    from ..ops._transformer import transform_block_table_to_list

    serialize_data = transform_block_table_to_list(data.block_table, data.data_manager)
    serialize_data.schema = schema
    return serialize_data


def serialize(ctx, data):
    return _serialize(ctx, data)


def deserialize(ctx, data):
    schema_meta, partition_order_mappings = parse_schema(data.schema)

    data_manager = DataManager.deserialize(schema_meta)

    site_name = ctx.local.name
    data_manager.fill_anonymous_site_name(site_name)

    from ..ops._transformer import transform_list_to_block_table

    block_table = transform_list_to_block_table(data, data_manager)

    return DataFrame(ctx, block_table, partition_order_mappings, data_manager)
