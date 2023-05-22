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

import pandas as pd
from typing import List, Tuple
import torch
from fate.arch import tensor
import numpy as np
from ..manager.data_manager import DataManager


def transform_to_tensor(block_table,
                        data_manager: "DataManager", dtype=None):
    def _merge_blocks(src_blocks, bids=None, fields=None):
        if len(bids) == 1:
            bid = bids[0]
            t = src_blocks[bid]
        else:
            i = 0
            tensors = []
            while i < len(fields):
                bid = fields[i][0]
                indexes = [fields[i][1]]
                j = i + 1
                while j < len(fields) and bid == fields[j][0] and indexes[-1] + 1 == fields[j][1]:
                    indexes.append(fields[j][1])
                    j += 1

                tensors.append(src_blocks[bid][:, indexes])
                i = j

            t = torch.hstack(tensors)

        if dtype:
            t = t.type(getattr(torch, t))

        return t

    block_indexes = data_manager.infer_operable_blocks()
    for block_id in block_indexes:
        if not data_manager.get_block(block_id).is_numeric:
            raise ValueError("Transform to distributed tensor should ensure every field is numeric")

    field_names = data_manager.infer_operable_field_names()
    fields_loc = data_manager.loc_block(field_names)

    _merged_func = functools.partial(
        _merge_blocks,
        bids=block_indexes,
        fields=fields_loc
    )
    merged_table = block_table.mapValues(_merged_func)

    shape_table = merged_table.mapValues(lambda v: v.shape)
    shapes = [shape_obj for k, shape_obj in sorted(shape_table.collect())]

    return tensor.DTensor.from_sharding_table(merged_table,
                                              shapes=shapes)


def transform_block_to_list(block_table, data_manager):
    fields_loc = data_manager.get_fields_loc()

    def _to_list(src_blocks):
        i = 0
        dst_list = None
        lines = 0
        while i < len(fields_loc):
            bid = fields_loc[i][0]
            if isinstance(src_blocks[bid], pd.Index):
                if not dst_list:
                    lines = len(src_blocks[bid])
                    dst_list = [[] for i in range(lines)]

                for j in range(lines):
                    dst_list[j].append(src_blocks[bid][j])

                i += 1
            else:
                """
                pd.values or tensor
                """
                indexes = [fields_loc[i][1]]
                j = i + 1
                while j < len(fields_loc) and fields_loc[j] == fields_loc[j - 1]:
                    indexes.append(fields_loc[j][1])
                    j += 1

                if isinstance(src_blocks[bid], np.ndarray):
                    for line_id, row_value in enumerate(src_blocks[bid][:, indexes]):
                        dst_list[line_id].extend(row_value.tolist())
                else:
                    try:
                        for line_id, row_value in enumerate(src_blocks[bid][:, indexes].tolist()):
                            dst_list[line_id].extend(row_value)
                    except Exception as e:
                        assert 1 == 2, (e, type(src_blocks[bid]), indexes)

                i = j

        return dst_list

    return block_table.mapValues(_to_list)


def transform_list_to_block(table, data_manager):
    from ..manager.block_manager import BlockType

    def _to_block(values):
        convert_blocks = []

        lines = len(values)
        for block_schema in data_manager.blocks:
            if block_schema.block_type == BlockType.index and len(block_schema.field_indexes) == 1:
                col_idx = block_schema.field_indexes[0]
                block_content = [values[i][col_idx] for i in range(lines)]
            else:
                block_content = []
                for i in range(lines):
                    buf = []
                    for col_idx in block_schema.field_indexes:
                        buf.append(values[i][col_idx])
                    block_content.append(buf)

            convert_blocks.append(block_schema.convert_block(block_content))

        return convert_blocks

    return table.mapValues(_to_block)


def transform_list_block_to_frame_block(block_table, data_manager):
    def _to_frame_block(blocks):
        convert_blocks = []
        for idx, block_schema in enumerate(data_manager.blocks):
            block_content = [block[idx] for block in blocks]
            convert_blocks.append(block_schema.convert_block(block_content))

        return convert_blocks

    return block_table.mapValues(_to_frame_block)


def transform_to_pandas_dataframe(block_table, data_manager):
    fields_loc = data_manager.get_fields_loc()

    def _flatten(blocks):
        flatten_ret = []
        lines = len(blocks[0])

        for lid in range(lines):
            row = [[] for i in range(len(fields_loc))]
            for field_id, (bid, offset) in enumerate(fields_loc):
                if isinstance(blocks[bid], np.ndarray):
                    row[field_id] = blocks[bid][lid][offset]
                elif isinstance(blocks[bid], torch.Tensor):
                    row[field_id] = blocks[bid][lid][offset].item()
                else:
                    row[field_id] = blocks[bid][lid]

            flatten_ret.append(row)

        return flatten_ret

    flatten_table = block_table.mapValues(_flatten)

    flatten_obj = []
    for k, v in flatten_table.collect():
        if not flatten_obj:
            flatten_obj = v
        else:
            flatten_obj.extend(v)

    fields = [data_manager.get_field_name(idx) for idx in range(len(fields_loc))]
    pd_df = pd.DataFrame(flatten_obj, columns=fields, dtype=object)
    pd_df.set_index(data_manager.schema.sample_id_name)

    for name in fields[1:]:
        dtype =  data_manager.get_field_type_by_name(name)
        if dtype in ["int32", "float32", "int64", "float64"]:
            pd_df[name] = pd_df[name].astype(dtype)

    return pd_df
