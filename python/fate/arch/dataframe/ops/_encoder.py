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
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from .._dataframe import DataFrame
from ..manager import BlockType


def get_dummies(df: "DataFrame", dtype="int32"):
    data_manager = df.data_manager
    block_indexes = data_manager.infer_operable_blocks()
    field_names = data_manager.infer_operable_field_names()

    if len(field_names) != 1:
        raise ValueError(f"get_dummies only support single column, but {len(field_names)} columns are found.")

    categories = _get_categories(df.block_table, block_indexes)[0][0]
    dst_field_names = ["_".join(map(str, [field_names[0], c])) for c in categories]
    dst_data_manager = data_manager.duplicate()
    dst_data_manager.pop_blocks(block_indexes)
    dst_data_manager.append_columns(dst_field_names, block_types=BlockType.get_block_type(dtype))

    block_table = _one_hot_encode(df.block_table, block_indexes, dst_data_manager, [[categories]], dtype=dtype)

    return DataFrame(
        df._ctx,
        block_table,
        partition_order_mappings=df.partition_order_mappings,
        data_manager=dst_data_manager
    )


def _get_categories(block_table, block_indexes):
    block_index_set = set(block_indexes)

    def _mapper(blocks):
        categories_ = []
        for bid, block in enumerate(blocks):
            if bid not in block_index_set:
                continue

            enc = OneHotEncoder()
            cate_block = enc.fit(block).categories_
            categories_.append([set(cate) for cate in cate_block])

        return categories_

    def _reducer(categories1_, categories2_):
        categories_ = []
        for cate_block1, cate_block2 in zip(categories1_, categories2_):
            cate_block = [cate1 | cate2 for cate1, cate2 in zip(cate_block1, cate_block2)]
            categories_.append(cate_block)

        return categories_

    categories = block_table.mapValues(_mapper).reduce(_reducer)

    categories = [[sorted(cate) for cate in cate_block]
                  for cate_block in categories]

    return categories


def _one_hot_encode(block_table, block_indexes, data_manager, categories, dtype):
    categories = [np.array(category) for category in categories]
    block_index_set = set(block_indexes)

    def _encode(blocks):
        ret_blocks = []
        enc_blocks = []
        idx = 0
        for bid, block in enumerate(blocks):
            if bid not in block_index_set:
                ret_blocks.append(block)
                continue

            enc = OneHotEncoder(dtype=dtype)
            enc.fit([[1]])  # one hot encoder need to fit first.
            enc.categories_ = categories[idx]
            idx += 1
            enc_blocks.append(enc.transform(block).toarray())

        ret_blocks.append(data_manager.blocks[-1].convert_block(np.hstack(enc_blocks)))

        return ret_blocks

    return block_table.mapValues(_encode)
