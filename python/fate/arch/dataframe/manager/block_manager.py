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

import bisect
import copy
import json
from enum import Enum
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from fate.arch.tensor.phe._tensor import PHETensor

from .schema_manager import SchemaManager


class BlockType(str, Enum):
    int32 = "int32"
    int64 = "int64"
    float32 = "float32"
    float64 = "float64"
    bool = "bool"
    index = "index"
    phe_tensor = "phe_tensor"
    np_object = "np_object"

    @staticmethod
    def promote_types(l_type: "BlockType", r_type: "BlockType"):
        if l_type < r_type:
            return r_type
        else:
            return l_type

    def __lt__(self, other):
        if self == other:
            return False

        if self == BlockType.bool:
            return other != BlockType.bool

        if self == BlockType.index:
            raise ValueError("Can not compare index types")

        if self == BlockType.np_object:
            return False

        if other == BlockType.np_object:
            return True

        if self == BlockType.int32:
            return other not in [BlockType.bool, BlockType.int32, BlockType]

        if self == BlockType.int64:
            return other not in [BlockType.bool, BlockType.int32, BlockType.int64]

        if self == BlockType.float32:
            return other in [BlockType.float64, BlockType.phe_tensor, BlockType.np_object]

        if self == BlockType.float64:
            return other in [BlockType.phe_tensor, BlockType.np_object]

        return False

    def __gt__(self, other):
        if self == other:
            return False

        return other < self

    @staticmethod
    def get_block_type(data_type):
        if isinstance(data_type, PHETensor) or type(data_type) == PHETensor:
            return BlockType.phe_tensor
        if hasattr(data_type, "dtype"):
            data_type = data_type.dtype
        if hasattr(data_type, "name"):
            data_type = data_type.name
        if isinstance(data_type, str):
            try:
                data_type = BlockType(data_type)
            except ValueError:
                data_type = "np_object"
            return BlockType(data_type)
        elif isinstance(data_type, (bool, np.bool_)) or data_type == torch.bool:
            return BlockType.bool
        elif isinstance(data_type, np.int64) or data_type == torch.int64:
            return BlockType.int64
        elif isinstance(data_type, (int, np.int32)) or data_type == torch.int32:
            return BlockType.int32
        elif isinstance(data_type, (float, np.float64)) or data_type == torch.float64:
            return BlockType.float64
        elif isinstance(data_type, np.float32) or data_type == torch.float32:
            return BlockType.float32
        else:
            return BlockType.np_object

    @staticmethod
    def is_tensor(block_type):
        return block_type in [BlockType.bool, BlockType.int32, BlockType.int64, BlockType.float32, BlockType.float64]

    @staticmethod
    def is_float(block_type):
        return block_type in [BlockType.float32, BlockType.float64]

    @staticmethod
    def is_integer(block_type):
        return block_type in [BlockType.int32, BlockType.int64]

    @staticmethod
    def is_arr(block_value):
        if isinstance(block_value, (torch.Tensor, np.ndarray)) and block_value.shape:
            return True
        return isinstance(block_value, list)


class Block(object):
    def __init__(self, field_indexes, block_type=None, should_compress=True):
        self._block_type = block_type
        self._field_indexes = field_indexes
        self._should_compress = should_compress

        self._field_index_mapping = dict(zip(field_indexes, range(len(field_indexes))))

    @property
    def block_type(self):
        return self._block_type

    @property
    def field_indexes(self):
        return self._field_indexes

    @field_indexes.setter
    def field_indexes(self, field_indexes: Union[list, set]):
        self._field_indexes = field_indexes
        self._field_index_mapping = dict(zip(field_indexes, range(len(field_indexes))))

    @property
    def should_compress(self):
        return self._should_compress

    @should_compress.setter
    def should_compress(self, should_compress):
        self._should_compress = should_compress

    @property
    def is_single(self):
        return len(self._field_indexes) == 1

    def get_field_offset(self, idx):
        return self._field_index_mapping[idx]

    def reset_field_indexes(self, dst_field_indexes):
        field_indexes = [dst_field_indexes[src_field_index] for src_field_index in self._field_indexes]
        self._field_index_mapping = dict(zip(field_indexes, range(len(field_indexes))))
        self._field_indexes = field_indexes

    def derive_block(self, field_indexes) -> Tuple["Block", bool, list]:
        """
        assume that sub field indexes always in self._field_indexes

        return: BlockObject, RetrievalIndexInOriBlock: list
        """
        src_field_indexes, dst_field_indexes = [], []
        field_indexes = sorted(field_indexes, key=lambda v: v[1])
        for src_field_index, dst_field_index in field_indexes:
            src_field_indexes.append(src_field_index)
            dst_field_indexes.append(dst_field_index)

        new_block = copy.deepcopy(self)
        new_block.field_indexes = dst_field_indexes
        # new_block = type(self)(dst_field_indexes)
        new_block.should_compress = self._should_compress

        # TODO: can be optimize as sub_field_indexes is ordered, but this is not a bottle neck
        changed = True
        if len(src_field_indexes) == len(self._field_indexes):
            is_monotonous = True
            for i in range(1, len(src_field_indexes)):
                if src_field_indexes[i] < src_field_indexes[i - 1]:
                    is_monotonous = False

            if is_monotonous:
                retrieval_indexes = [i for i in range(len(self._field_indexes))]
                changed = False
            else:
                retrieval_indexes = [bisect.bisect_left(self._field_indexes, col) for col in src_field_indexes]
        else:
            retrieval_indexes = [bisect.bisect_left(self._field_indexes, col) for col in src_field_indexes]

        return new_block, changed, retrieval_indexes

    def __str__(self):
        field_indexes_format = ",".join(map(str, self._field_indexes))
        return f"block_type:{self._block_type}, fields=={field_indexes_format}"

    def is_numeric(self):
        return self._block_type in {BlockType.int32, BlockType.int64, BlockType.float32, BlockType.float64}

    def is_phe_tensor(self):
        return self._block_type == BlockType.phe_tensor

    def to_dict(self):
        return dict(
            block_type=json.dumps(self._block_type),
            field_indexes=self._field_indexes,
            should_compress=self._should_compress,
        )

    @staticmethod
    def from_dict(s_dict):
        block_type = json.loads(s_dict["block_type"])
        field_indexes = s_dict["field_indexes"]
        should_compress = s_dict["should_compress"]
        block = Block.get_block_by_type(block_type)
        return block(field_indexes, should_compress=should_compress)

    @staticmethod
    def get_block_by_type(block_type):
        if not isinstance(block_type, BlockType):
            block_type = BlockType.get_block_type(block_type)

        if block_type == block_type.int32:
            return Int32Block
        elif block_type == block_type.int64:
            return Int64Block
        elif block_type == block_type.float32:
            return Float32Block
        elif block_type == block_type.float64:
            return Float64Block
        elif block_type == block_type.bool:
            return BoolBlock
        elif block_type == block_type.index:
            return IndexBlock
        elif block_type == block_type.phe_tensor:
            return PHETensorBlock
        else:
            return NPObjectBlock

    @staticmethod
    def convert_block(block):
        raise NotImplemented

    def convert_block_type(self, block_type):
        converted_block = self.get_block_by_type(block_type)(self._field_indexes, block_type, self._should_compress)

        return converted_block

    # @classmethod
    # def retrieval_row(cls, block, indexes):
    #     if isinstance(block, CiphertextVector):
    #         return block.slice_indexes(indexes)
    #     elif isinstance(block, pd.Index):
    #         if isinstance(indexes, list):
    #             return block[indexes]
    #         else:
    #             return pd.Index(block[indexes])
    #     else:
    #         return block[indexes]

    @classmethod
    def transform_block_to_list(cls, block):
        return block.tolist()

    # @classmethod
    # def transform_row_to_raw(cls, block, index):
    #     if isinstance(block, pd.Index):
    #         return block[index]
    #     elif isinstance(block, CiphertextVector):
    #         return block.slice_indexes([index])
    #     else:
    #         return block[index].tolist()

    @classmethod
    def vstack(cls, blocks):
        ret = blocks[0]
        if isinstance(ret, pd.Index):
            for block in blocks[1:]:
                ret = ret.append(block)
        elif isinstance(ret, torch.Tensor):
            ret = torch.vstack(blocks)
        elif isinstance(ret, np.ndarray):
            ret = np.vstack(blocks)
        else:
            raise ValueError(f"Not implemented block vstack for type {type(ret)}")

        return ret


class Int32Block(Block):
    def __init__(self, *args, **kwargs):
        super(Int32Block, self).__init__(*args, **kwargs)
        self._block_type = BlockType.int32

    @staticmethod
    def convert_block(block):
        if isinstance(block, torch.Tensor):
            if block.dtype == torch.int32:
                return block
            else:
                return block.to(torch.int32)
        try:
            return torch.tensor(block, dtype=torch.int32)
        except ValueError:
            return torch.tensor(np.array(block, dtype="int32"), dtype=torch.int32)

    @property
    def dtype(self):
        return torch.int32


class Int64Block(Block):
    def __init__(self, *args, **kwargs):
        super(Int64Block, self).__init__(*args, **kwargs)
        self._block_type = BlockType.int64

    @staticmethod
    def convert_block(block):
        if isinstance(block, torch.Tensor):
            if block.dtype == torch.int64:
                return block
            else:
                return block.to(torch.int64)
        try:
            return torch.tensor(block, dtype=torch.int64)
        except ValueError:
            return torch.tensor(np.array(block, dtype="int64"), dtype=torch.int64)

    @property
    def dtype(self):
        return torch.int64


class Float32Block(Block):
    def __init__(self, *args, **kwargs):
        super(Float32Block, self).__init__(*args, **kwargs)
        self._block_type = BlockType.float32

    @staticmethod
    def convert_block(block):
        if isinstance(block, torch.Tensor):
            if block.dtype == torch.float32:
                return block
            else:
                return block.to(torch.float32)
        try:
            return torch.tensor(block, dtype=torch.float32)
        except ValueError:
            return torch.tensor(np.array(block, dtype="float32"), dtype=torch.float32)

    @property
    def dtype(self):
        return torch.float32


class Float64Block(Block):
    def __init__(self, *args, **kwargs):
        super(Float64Block, self).__init__(*args, **kwargs)
        self._block_type = BlockType.float64

    @staticmethod
    def convert_block(block):
        if isinstance(block, torch.Tensor):
            if block.dtype == torch.float64:
                return block
            else:
                return block.to(torch.float64)
        try:
            return torch.tensor(block, dtype=torch.float64)
        except ValueError:
            return torch.tensor(np.array(block, dtype="float64"), dtype=torch.float64)

    @property
    def dtype(self):
        return torch.float64


class BoolBlock(Block):
    def __init__(self, *args, **kwargs):
        super(BoolBlock, self).__init__(*args, **kwargs)
        self._block_type = BlockType.bool

    @staticmethod
    def convert_block(block):
        if isinstance(block, torch.Tensor):
            if block.dtype == torch.bool:
                return block
            else:
                return block.to(torch.bool)
        try:
            return torch.tensor(block, dtype=torch.bool)
        except ValueError:
            return torch.tensor(np.array(block, dtype="bool"), dtype=torch.bool)

    @property
    def dtype(self):
        return torch.bool


class IndexBlock(Block):
    def __init__(self, *args, **kwargs):
        super(IndexBlock, self).__init__(*args, **kwargs)
        self._block_type = BlockType.index

    @staticmethod
    def convert_block(block):
        return pd.Index(block, dtype=str)

    @property
    def dtype(self):
        return np.dtype("O")


class PHETensorBlock(Block):
    def __init__(self, *args, **kwargs):
        kwargs["should_compress"] = False

        super(PHETensorBlock, self).__init__(*args, **kwargs)
        self._block_type = BlockType.phe_tensor
        self._pk = None
        self._evaluator = None
        self._coder = None
        self._dtype = None
        self._device = None

    def set_extra_kwargs(self, pk, evaluator, coder, dtype, device):
        self._pk = pk
        self._evaluator = evaluator
        self._coder = coder
        self._dtype = dtype
        self._device = device

    def convert_block(self, block):
        if isinstance(block, list):
            block = self._evaluator.cat(block)
        return block

    def convert_to_phe_tensor(self, block, shape):
        if isinstance(block, PHETensor):
            return block

        if isinstance(block, list):
            block = block[0].cat(block[1:])

        return PHETensor(
            pk=self._pk,
            evaluator=self._evaluator,
            coder=self._coder,
            shape=shape,
            data=block,
            device=self._device,
            dtype=self._dtype,
        )

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return self._dtype


class NPObjectBlock(Block):
    def __init__(self, *args, **kwargs):
        super(NPObjectBlock, self).__init__(*args, **kwargs)
        self._block_type = BlockType.np_object

    @staticmethod
    def convert_block(block):
        return np.array(block, dtype=object)

    @property
    def dtype(self):
        return np.dtype("O")


class BlockManager(object):
    def __init__(self):
        """
        block manager managers the block structure of each partition, distributed always
        Please note that we only compress numeric or bool or object type, not compress index type yet

        _blocks: list of Blocks, each element contains the attrs: axis_set
        """
        self._blocks = []
        self._field_block_mapping = dict()

    def initialize_blocks(self, schema_manager: SchemaManager):
        """
        sample_id
        match_id
        label
        weight
        fields
        """
        schema = schema_manager.schema
        sample_id_type = schema_manager.get_field_types(name=schema.sample_id_name)

        self._blocks.append(
            Block.get_block_by_type(sample_id_type)(
                [schema_manager.get_field_offset(schema.sample_id_name)], should_compress=False
            )
        )

        if schema.match_id_name:
            dtype = schema_manager.get_field_types(name=schema.match_id_name)
            self._blocks.append(
                Block.get_block_by_type(dtype)(
                    [schema_manager.get_field_offset(schema.match_id_name)], should_compress=False
                )
            )

        if schema.label_name:
            dtype = schema_manager.get_field_types(name=schema.label_name)
            self._blocks.append(
                Block.get_block_by_type(dtype)(
                    [schema_manager.get_field_offset(schema.label_name)], should_compress=False
                )
            )

        if schema.weight_name:
            dtype = schema_manager.get_field_types(name=schema.weight_name)
            self._blocks.append(
                Block.get_block_by_type(dtype)(
                    [schema_manager.get_field_offset(schema.weight_name)], should_compress=False
                )
            )

        for column_name in schema.columns:
            dtype = schema_manager.get_field_types(name=column_name)
            self._blocks.append(
                Block.get_block_by_type(dtype)([schema_manager.get_field_offset(column_name)], should_compress=True)
            )

        new_blocks, _1, _2 = self.compress()

        self.reset_blocks(new_blocks)

    def append_fields(self, field_indexes, block_types, should_compress=True):
        block_num = len(self._blocks)
        block_ids = []
        if isinstance(block_types, list):
            for offset, (field_index, block_type) in enumerate(zip(field_indexes, block_types)):
                block = Block.get_block_by_type(block_type)
                self._blocks.append(block(field_indexes=[field_index], should_compress=should_compress))
                self._field_block_mapping[field_index] = (block_num + offset, 0)
                block_ids.append(block_num + offset)
        else:
            block = Block.get_block_by_type(block_types)
            self._blocks.append(block(field_indexes=field_indexes, should_compress=should_compress))
            block_ids.append(block_num)
            for offset, field_index in enumerate(field_indexes):
                self._field_block_mapping[field_index] = (block_num, offset)

        return block_ids

    def pop_blocks(self, block_indexes: List[int]):
        block_index_set = set(block_indexes)
        blocks = []
        field_block_mapping = dict()

        for bid, block in enumerate(self._blocks):
            if bid not in block_index_set:
                blocks.append(block)

        self._blocks = blocks

    def split_fields(self, field_indexes, block_types):
        field_sets = set(field_indexes)
        block_field_maps = dict()
        for idx, field_index in enumerate(field_indexes):
            block_id, offset = self.loc_block(field_index, with_offset=True)
            if block_id not in block_field_maps:
                block_field_maps[block_id] = []

            block_type = block_types[idx].value if isinstance(block_types, list) else block_types.value
            block_field_maps[block_id].append([field_index, offset, block_type])

        cur_block_num = len(self._blocks)
        narrow_blocks = []
        for block_id, field_with_offset_list in block_field_maps.items():
            if len(self._blocks[block_id].field_indexes) == len(field_with_offset_list):
                if len(field_with_offset_list) == 1:
                    self._blocks[block_id] = Block.get_block_by_type(block_type)(
                        self._blocks[block_id].field_indexes, should_compress=self._blocks[block_id].should_compress
                    )
                else:
                    should_compress = self._blocks[block_id].should_compress
                    for idx, (field, offset, block_type) in enumerate(field_with_offset_list):
                        if not idx:
                            self._blocks[block_id] = Block.get_block_by_type(block_type)(
                                [field], should_compress=should_compress
                            )
                            self._field_block_mapping[field] = (block_id, 0)
                        else:
                            self._blocks.append(
                                Block.get_block_by_type(block_type)([field], should_compress=should_compress)
                            )
                            self._field_block_mapping[field] = (cur_block_num, 0)
                            cur_block_num += 1
            else:
                narrow_field_indexes = []
                narrow_field_offsets = []
                for offset, field in enumerate(self._blocks[block_id].field_indexes):
                    if field not in field_sets:
                        narrow_field_indexes.append(field)
                        narrow_field_offsets.append(offset)

                narrow_blocks.append((block_id, narrow_field_offsets))

                self._blocks[block_id] = Block.get_block_by_type(self._blocks[block_id].block_type)(
                    narrow_field_indexes, should_compress=self._blocks[block_id].should_compress
                )
                for offset, narrow_field in enumerate(narrow_field_indexes):
                    self._field_block_mapping[narrow_field] = (block_id, offset)

                for field, offset, block_type in field_with_offset_list:
                    self._blocks.append(
                        Block.get_block_by_type(block_type)(
                            [field], should_compress=self._blocks[block_id].should_compress
                        )
                    )
                    self._field_block_mapping[field] = (cur_block_num, 0)
                    cur_block_num += 1

        dst_blocks = [self._field_block_mapping[field][0] for field in field_indexes]

        return narrow_blocks, dst_blocks

    @property
    def blocks(self):
        return self._blocks

    @blocks.setter
    def blocks(self, blocks):
        self._blocks = blocks

    @property
    def field_block_mapping(self):
        return self._field_block_mapping

    @field_block_mapping.setter
    def field_block_mapping(self, field_block_mapping):
        self._field_block_mapping = field_block_mapping

    def reset_block_field_indexes(self, field_index_changes: Dict[int, int]):
        field_block_mapping = dict()
        for bid in range(len(self._blocks)):
            self._blocks[bid].reset_field_indexes(field_index_changes)
            for offset, field_index in enumerate(self._blocks[bid].field_indexes):
                field_block_mapping[field_index] = (bid, offset)

        self._field_block_mapping = field_block_mapping

    def duplicate(self):
        dup_block_manager = BlockManager()
        dup_block_manager.blocks = copy.deepcopy(self._blocks)
        dup_block_manager.field_block_mapping = copy.deepcopy(self._field_block_mapping)

        return dup_block_manager

    def get_numeric_block(self):
        numeric_blocks = []
        for _blk in self._blocks:
            if _blk.is_numeric:
                numeric_blocks.append(_blk)

        return numeric_blocks

    def loc_block(self, field_index, with_offset=True) -> Union[Tuple[int, int], int]:
        if with_offset:
            return self._field_block_mapping[field_index]
        else:
            return self._field_block_mapping[field_index][0]

    def compress(self):
        compressible_blocks = dict()

        has_compressed = False
        for block_id, block in enumerate(self._blocks):
            if block.should_compress:
                has_compressed = True

            if block.block_type not in compressible_blocks:
                compressible_blocks[block.block_type] = []
            compressible_blocks[block.block_type].append((block_id, block))

        if not has_compressed:
            return self._blocks, [], []

        new_blocks, to_compress_block_loc = [], []
        non_compressed_block_changes = dict()
        for block_type, block_list in compressible_blocks.items():
            _blocks = []
            _block_ids = []
            for block_id, block in block_list:
                if block.should_compress:
                    _blocks.append(block)
                    _block_ids.append(block_id)
                else:
                    non_compressed_block_changes[block_id] = len(new_blocks)
                    new_blocks.append(block)

            if len(_blocks) > 1:
                dst_block_id = len(new_blocks)
                block_loc = []
                """
                merge all field_indexes, use set instead of merge sort to avoid O(n * len(_blocks))
                """
                field_indexes_set = set()
                for block in _blocks:
                    field_indexes_set |= set(block.field_indexes)
                new_blocks.append(
                    Block.get_block_by_type(block_type)(sorted(list(field_indexes_set)), should_compress=True)
                )

                dst_loc_mappings = dict(zip(new_blocks[-1].field_indexes, range(len(new_blocks[-1].field_indexes))))
                for block_id, block in zip(_block_ids, _blocks):
                    new_field_indexes = [dst_loc_mappings[bid] for bid in block.field_indexes]
                    block_loc.append((block_id, new_field_indexes))

                to_compress_block_loc.append((dst_block_id, block_loc))

            elif _blocks:
                non_compressed_block_changes[_block_ids[0]] = len(new_blocks)
                new_blocks.append(_blocks[0])

        return new_blocks, to_compress_block_loc, non_compressed_block_changes

    def reset_blocks(self, blocks):
        self._blocks = blocks
        self._field_block_mapping = dict()
        for bid, blocks in enumerate(self._blocks):
            for idx in blocks.field_indexes:
                self._field_block_mapping[idx] = (bid, blocks.get_field_offset(idx))

    def apply(
        self,
    ):
        """
        make some fields to some other type

        """

    def to_dict(self):
        """
        deserialize
        """
        return dict(blocks=[blk.to_dict() for blk in self._blocks])

    @staticmethod
    def from_dict(s_dict):
        blocks = [Block.from_dict(block_s_dict) for block_s_dict in s_dict["blocks"]]
        bm = BlockManager()
        bm.blocks = blocks

        return blocks

    def derive_new_block_manager(self, indexes: list) -> Tuple["BlockManager", List[Tuple[int, int, bool, List]]]:
        """
        derive a new block manager filter by indexes

        return:  list, each element in order:
                 (src_block, src_block_dst_block, block_changed, old_block_indexes)
        """
        block_manager = BlockManager()
        block_index_mapping = dict()

        indexes = sorted(indexes)
        for src_field_index, dst_field_index in indexes:
            bid = self.loc_block(src_field_index, with_offset=False)
            if bid not in block_index_mapping:
                block_index_mapping[bid] = []
            block_index_mapping[bid].append((src_field_index, dst_field_index))

        derived_blocks = []
        blocks_loc = []

        new_block_id = 0
        for bid, field_indexes in block_index_mapping.items():
            block, block_changed, retrieval_indexes = self._blocks[bid].derive_block(field_indexes)
            derived_blocks.append(block)
            blocks_loc.append((bid, new_block_id, block_changed, retrieval_indexes))
            new_block_id += 1

        block_manager.reset_blocks(derived_blocks)

        return block_manager, blocks_loc
