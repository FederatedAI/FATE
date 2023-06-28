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
#
from .._dataframe import DataFrame
from ..ops._transformer import transform_block_to_list


def collect_data(df: DataFrame, num=100):
    data = []
    fields_loc = df.data_manager.get_fields_loc()
    for block_id, blocks in df.block_table.collect():
        data_list = transform_block_to_list(blocks, fields_loc)

        if len(data) + len(data_list) <= num:
            data.extend(data_list)
        else:
            data.extend(data_list[:num])

        num -= len(data_list)

        if num <= 0:
            break

    return data
