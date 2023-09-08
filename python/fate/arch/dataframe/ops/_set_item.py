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
from .._dataframe import DataFrame
from ..manager.block_manager import BlockType
from ..manager.data_manager import DataManager
from fate.arch.tensor import DTensor
from fate.arch.tensor.phe._tensor import PHETensor


def set_item(df: "DataFrame", keys, items, state):
    """
    state: 1 - keys are all new
           2 - keys are all old
    """
    if state == 1:
        _set_new_item(df, keys, items)
    else:
        _set_old_item(df, keys, items)


def set_label_or_weight(df: "DataFrame", item: "DataFrame", key_type="label"):
    if not isinstance(item, DataFrame):
        raise ValueError(f"To set label or weight, make sure rhs type={type(df)}")

    data_manager = df.data_manager
    other_data_manager = item.data_manager
    other_field_names = other_data_manager.infer_operable_field_names()
    other_block_id = other_data_manager.loc_block(other_field_names[0], with_offset=False)

    if len(other_field_names) > 1:
        raise ValueError(f"Too many columns of rhs, only one is supported")

    other_block_type = other_data_manager.blocks[other_block_id].block_type
    if (name := getattr(df.schema, f"{key_type}_name")) is not None:
        block_id = data_manager.loc_block(name, with_offset=False)
        block_table = df.block_table.join(item.block_table,
                                          lambda blocks1, blocks2:
                                          [block if bid != block_id else blocks2[other_block_id]
                                           for bid, block in enumerate(blocks1)]
        )
        if data_manager.blocks[block_id].block_type < other_block_type:
            data_manager.blocks[block_id].convert_block_type(other_block_type)
    else:
        data_manager.add_label_or_weight(key_type=key_type,
                                         name=other_field_names[0],
                                         block_type=other_block_type)

        block_table = df.block_table.join(item.block_table,
                                          lambda blocks1, blocks2: blocks1 + [blocks2[other_block_id]])

    df.block_table = block_table
    df.data_manager = data_manager


def _set_new_item(df: "DataFrame", keys, items):
    def _append_single(blocks, item, col_len, bid=None, dm: DataManager=None):
        lines = len(blocks[0])
        ret_blocks = [block for block in blocks]
        ret_blocks.append(dm.blocks[bid].convert_block([[item for _ in range(col_len)] for idx in range(lines)]))

        return ret_blocks

    def _append_multi(blocks, item_list, bid_list=None, dm: DataManager=None):
        lines = len(blocks[0])
        ret_blocks = [block for block in blocks]
        for bid, item in zip(bid_list, item_list):
            ret_blocks.append(dm.blocks[bid].convert_block([[item] for _ in range(lines)]))

        return ret_blocks

    def _append_df(l_blocks, r_blocks, r_blocks_loc=None, dm=None):
        ret_blocks = [block for block in l_blocks]
        l_bid = len(ret_blocks)
        for bid, offset in r_blocks_loc:
            if dm.blocks[l_bid].is_phe_tensor():
                ret_blocks.append(r_blocks[bid])
            elif r_blocks[bid].shape[1] == 1:
                ret_blocks.append(r_blocks[bid])
            else:
                ret_blocks.append(r_blocks[bid][:, [offset]])
            l_bid += 1

        return ret_blocks

    def _append_tensor(l_blocks, r_tensor, bid_list=None, dm: DataManager = None):
        ret_blocks = [block for block in l_blocks]
        for offset, bid in enumerate(bid_list):
            ret_blocks.append(dm.blocks[bid].convert_block(r_tensor[:, offset: offset+1]))

        return ret_blocks

    def _append_phe_tensor(l_blocks, r_tensor):
        ret_blocks = [block for block in l_blocks]
        ret_blocks.append(r_tensor._data)

        return ret_blocks

    data_manager = df.data_manager.duplicate()
    if isinstance(items, (bool, int, float, str, np.int32, np.float32, np.int64, np.float64, np.bool_)):
        bids = data_manager.append_columns(keys, BlockType.get_block_type(items))
        _append_func = functools.partial(_append_single, item=items, col_len=len(keys), bid=bids[0], dm=data_manager)
        block_table = df.block_table.mapValues(_append_func)

    elif isinstance(items, list):
        if len(keys) != len(items):
            if len(keys) > 1:
                raise ValueError("Must have equal len keys and value when setting with an iterable")
            bids = data_manager.append_columns(keys, BlockType.get_block_type("object"))
            _append_func = functools.partial(_append_single, item=items, col_len=len(keys),
                                             bid=bids[0], dm=data_manager)
        else:
            bids = data_manager.append_columns(keys, [BlockType.get_block_type(items[i]) for i in range(len(keys))])
            _append_func = functools.partial(_append_multi, item_list=items, bid_list=bids, dm=data_manager)
        block_table = df.block_table.mapValues(_append_func)
    elif isinstance(items, DataFrame):
        other_dm = items.data_manager
        operable_fields = other_dm.infer_operable_field_names()
        operable_blocks_loc = other_dm.loc_block(operable_fields)
        block_types = [other_dm.blocks[bid].block_type for bid, _ in operable_blocks_loc]
        if len(keys) != len(operable_fields):
            raise ValueError("Setitem with rhs=DataFrame must have equal len keys")
        data_manager.append_columns(keys, block_types)

        l = len(keys)
        for idx, (other_block_id, _) in enumerate(operable_blocks_loc):
            if data_manager.blocks[-l + idx].is_phe_tensor():
                other_block = other_dm.blocks[other_block_id]
                data_manager.blocks[-l + idx].set_extra_kwargs(pk=other_block._pk,
                                                               evaluator=other_block._evaluator,
                                                               coder=other_block._coder,
                                                               dtype=other_block._dtype,
                                                               device=other_block._device)

        _append_func = functools.partial(_append_df, r_blocks_loc=operable_blocks_loc, dm=data_manager)
        block_table = df.block_table.join(items.block_table, _append_func)
    elif isinstance(items, DTensor):
        meta_data = items.shardings._data.mapValues(
            lambda v: (v.pk, v.evaluator, v.coder, v.dtype) if isinstance(v, PHETensor) else None
        ).first()[1]

        if isinstance(meta_data, tuple):
            block_type = BlockType.phe_tensor
            if len(keys) != 1:
                raise ValueError("to set item of PHETensor, lhs should has only one columns.")
            data_manager.append_columns(keys, block_type)
            data_manager.blocks[-1].set_extra_kwargs(pk=meta_data[0], evaluator=meta_data[1], coder=meta_data[2],
                                                     dtype=meta_data[3], device=items.device)
            _append_func = functools.partial(_append_phe_tensor)
            block_table = df.block_table.join(items.shardings._data, _append_func)
        else:
            block_type = BlockType.get_block_type(items.dtype)
            if len(keys) != items.shape[1]:
                raise ValueError("Setitem with rhs=DTensor must have equal len keys")
            bids = data_manager.append_columns(keys, block_type)
            _append_func = functools.partial(_append_tensor, bid_list=bids, dm=data_manager)
            block_table = df.block_table.join(items.shardings._data, _append_func)
    else:
        raise ValueError(f"Seiitem with rhs_type={type(items)} is not supported")

    df.block_table = block_table
    df.data_manager = data_manager


def _set_old_item(df: "DataFrame", keys, items):
    def _replace_single(blocks, item=None, narrow_loc=None, dst_bids=None, dm: DataManager=None):
        ret_blocks = [block for block in blocks]
        for i in range(len(ret_blocks), dm.block_num):
            ret_blocks.append([])

        for bid, offsets in narrow_loc:
            ret_blocks[bid] = ret_blocks[bid][:, offsets]

        lines = len(blocks[0])
        for dst_bid in dst_bids:
            ret_blocks[dst_bid] = dm.blocks[dst_bid].convert_block([[item] for idx in range(lines)])

        return ret_blocks

    def _replace_multi(blocks, item_list=None, narrow_loc=None, dst_bids=None, dm: DataManager = None):
        ret_blocks = [block for block in blocks]
        for i in range(len(ret_blocks), dm.block_num):
            ret_blocks.append([])

        for bid, offsets in narrow_loc:
            ret_blocks[bid] = ret_blocks[bid][:, offsets]

        lines = len(blocks[0])
        for dst_bid, item in zip(dst_bids, item_list):
            ret_blocks[dst_bid] = dm.blocks[dst_bid].convert_block([[item] for idx in range(lines)])

        return ret_blocks

    def _replace_df(l_blocks, r_blocks, narrow_loc=None, dst_bids=None, r_blocks_loc=None, dm: DataManager=None):
        ret_blocks = [block for block in l_blocks]
        for i in range(len(ret_blocks), dm.block_num):
            ret_blocks.append([])

        for bid, offsets in narrow_loc:
            ret_blocks[bid] = ret_blocks[bid][:, offsets]

        for dst_bid, (r_bid, offset) in zip(dst_bids, r_blocks_loc):
            ret_blocks[dst_bid] = r_blocks[r_bid][:, [offset]]

        return ret_blocks

    def _replace_tensor(blocks, r_tensor, narrow_loc=None, dst_bids=None, dm: DataManager = None):
        ret_blocks = [block for block in blocks]
        for i in range(len(ret_blocks), dm.block_num):
            ret_blocks.append([])

        for bid, offsets in narrow_loc:
            ret_blocks[bid] = ret_blocks[bid][:, offsets]

        for offset, dst_bid in enumerate(dst_bids):
            ret_blocks[dst_bid] = dm.blocks[dst_bid].convert_block(r_tensor[:, offset : offset + 1])

        return ret_blocks

    data_manager = df.data_manager.duplicate()
    if isinstance(items, (bool, int, float, str, np.int32, np.float32, np.int64, np.float64, np.bool_)):
        narrow_blocks, dst_blocks = data_manager.split_columns(keys, BlockType.get_block_type(items))
        replace_func = functools.partial(_replace_single, item=items, narrow_loc=narrow_blocks,
                                         dst_bids=dst_blocks, dm=data_manager)
        block_table = df.block_table.mapValues(replace_func)
    elif isinstance(items, list):
        if len(keys) != len(items):
            if len(keys) > 1:
                raise ValueError("Must have equal len keys and value when setting with an iterable")
            narrow_blocks, dst_blocks = data_manager.split_columns(keys, BlockType.get_block_type("object"))
            replace_func = functools.partial(_replace_single, item=items[0], narrow_loc=narrow_blocks,
                                             dst_bids=dst_blocks, dm=data_manager)
        else:
            narrow_blocks, dst_blocks = data_manager.split_columns(keys,
                                                                   [BlockType.get_block_type(item) for item in items])
            replace_func = functools.partial(_replace_multi, item_list=items, narrow_loc=narrow_blocks,
                                             dst_bids=dst_blocks, dm=data_manager)

        block_table = df.block_table.mapValues(replace_func)
    elif isinstance(items, DataFrame):
        other_dm = items.data_manager
        operable_fields = other_dm.infer_operable_field_names()
        operable_blocks_loc = other_dm.loc_block(operable_fields)
        block_types = [other_dm.blocks[bid].block_type for bid, _ in operable_blocks_loc]
        if len(keys) != len(operable_fields):
            raise ValueError("Setitem with rhs=DataFrame must have equal len keys")
        narrow_blocks, dst_blocks = data_manager.split_columns(keys, block_types)
        replace_func = functools.partial(_replace_df, narrow_loc=narrow_blocks, dst_bids=dst_blocks,
                                         r_blocks_loc=operable_blocks_loc, dm=data_manager)
        block_table = df.block_table.join(items.block_table, replace_func)
    elif isinstance(items, DTensor):
        if len(keys) != items.shape[1]:
            raise ValueError("Setitem with rhs=DTensor must have equal len keys")
        block_type = BlockType.get_block_type(items.dtype)
        narrow_blocks, dst_blocks = data_manager.split_columns(keys, block_type)
        replace_func = functools.partial(_replace_tensor, narrow_loc=narrow_blocks,
                                         dst_bids=dst_blocks, dm=data_manager)
        block_table = df.block_table.join(items.shardings._data, replace_func)

    else:
        raise ValueError(f"Seiitem with rhs_type={type(items)} is not supported")

    df.block_table = block_table
    df.data_manager = data_manager
