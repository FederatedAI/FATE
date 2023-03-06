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
import json
import pandas as pd
import torch
from enum import Enum
from typing import Union, Tuple, List
from .schema_manager import SchemaManager
from fate.arch import tensor


class BlockType(str, Enum):
    int32 = "int32"
    int64 = "int64"
    float32 = "float32"
    float64 = "float64"
    bool = "bool"
    index = "index"
    pandas_object = "pd_obj"


class Block(object):
    def __init__(self, column_indexes, block_type=None, should_compress=True):
        self._block_type = block_type
        self._column_indexes = column_indexes
        self._should_compress = should_compress

        self._column_index_mapping = dict(zip(column_indexes, range(len(column_indexes))))

    @property
    def block_type(self):
        return self._block_type

    @property
    def column_indexes(self):
        return self._column_indexes

    @column_indexes.setter
    def column_indexes(self, column_indexes: Union[list, set]):
        self._column_indexes = column_indexes


    @property
    def should_compress(self):
        return self._should_compress

    @should_compress.setter
    def should_compress(self, should_compress):
        self._should_compress = should_compress

    @property
    def is_single_column(self):
        return len(self._column_indexes) == 1

    def get_column_offset(self, idx):
        return self._column_index_mapping[idx]

    def derive_block(self, sub_column_indexes) -> Tuple["Block", bool, list]:
        """
        assume that sub column indexes always in self._column_indexes

        return: BlockObject, RetrievalIndexInOriBlock: list
        """
        new_block = type(self)(sub_column_indexes)
        new_block.should_compress = self._should_compress

        # TODO: can be optimize as sub_column_indexes is ordered, but this is not a bottle neck
        changed = True
        if len(sub_column_indexes) == len(self._column_indexes):
            retrieval_indexes = [i for i in range(len(self._column_indexes))]
            changed = False
        else:
            retrieval_indexes = [bisect.bisect_left(self._column_indexes, col) for col in sub_column_indexes]

        return new_block, changed, retrieval_indexes

    def __str__(self):
        column_indexes_format = ",".join(map(str, self._column_indexes))
        return f"block_type:{self._block_type}, columns=={column_indexes_format}"

    def is_numeric(self):
        return self._block_type in {
            BlockType.int32, BlockType.int64,
            BlockType.float32, BlockType.float64
        }

    def to_dict(self):
        return dict(
            block_type= json.dumps(self._block_type),
            column_indexes=self._column_indexes,
            should_compress=self._should_compress
        )

    @staticmethod
    def from_dict(s_dict):
        block_type = json.loads(s_dict["block_type"])
        column_indexes = s_dict["column_indexes"]
        should_compress = s_dict["should_compress"]
        block = Block.get_block_by_type(block_type)
        return block(column_indexes, should_compress=should_compress)

    @staticmethod
    def get_block_by_type(block_type: str):
        if block_type == "int32":
            return Int32Block
        elif block_type == "int64":
            return Int64Block
        elif block_type == "float32":
            return Float32Block
        elif block_type == "float64":
            return Float64Block
        elif block_type == "bool":
            return BoolBlock
        elif block_type == "index":
            return IndexBlock
        else:
            return PandasObjectBlock

    @staticmethod
    def convert_block(block):
        raise NotImplemented


class Int32Block(Block):
    def __init__(self, *args, **kwargs):
        super(Int32Block, self).__init__(*args, **kwargs)
        self._block_type = BlockType.int32

    @staticmethod
    def convert_block(block):
        return  tensor.tensor(torch.tensor(block, dtype=torch.int32))


class Int64Block(Block):
    def __init__(self, *args, **kwargs):
        super(Int64Block, self).__init__(*args, **kwargs)
        self._block_type = BlockType.int64

    @staticmethod
    def convert_block(block):
        return  tensor.tensor(torch.tensor(block, dtype=torch.int64))


class Float32Block(Block):
    def __init__(self, *args, **kwargs):
        super(Float32Block, self).__init__(*args, **kwargs)
        self._block_type = BlockType.float32

    @staticmethod
    def convert_block(block):
        return  tensor.tensor(torch.tensor(block, dtype=torch.float32))


class Float64Block(Block):
    def __init__(self, *args, **kwargs):
        super(Float64Block, self).__init__(*args, **kwargs)
        self._block_type = BlockType.float64

    @staticmethod
    def convert_block(block):
        return  tensor.tensor(torch.tensor(block, dtype=torch.float64))


class BoolBlock(Block):
    def __init__(self, *args, **kwargs):
        super(BoolBlock, self).__init__(*args, **kwargs)
        self._block_type = BlockType.bool

    @staticmethod
    def convert_block(block):
        return  tensor.tensor(torch.tensor(block, dtype=torch.bool))


class IndexBlock(Block):
    def __init__(self, *args, **kwargs):
        super(IndexBlock, self).__init__(*args, **kwargs)
        self._block_type = BlockType.index

    @staticmethod
    def convert_block(block):
        return pd.Index(block, dtype=str)


class PandasObjectBlock(Block):
    def __init__(self, *args, **kwargs):
        super(PandasObjectBlock, self).__init__(*args, **kwargs)
        self._block_type = BlockType.pandas_object

    @staticmethod
    def convert_block(block):
        return pd.DataFrame(block, dtype=object)


class BlockManager(object):
    def __init__(self):
        """
        block manager managers the block structure of each partition, distributed always
        Please note that we only compress numeric or bool or object type, not compress index type yet

        _blocks: list of Blocks, each element contains the attrs: axis_set
        """
        self._blocks = []
        self._column_block_mapping = dict()

    def initialize_blocks(self, schema_manager: SchemaManager):
        """
        sample_id
        match_id
        label
        weight
        columns
        """
        schema = schema_manager.schema
        sample_id_type = schema_manager.get_column_types(name=schema.sample_id_name)

        self._blocks.append(Block.get_block_by_type(sample_id_type)(
            [schema_manager.get_column_index(schema.sample_id_name)], should_compress=False))

        if schema.match_id_name:
            dtype = schema_manager.get_column_types(name=schema.match_id_name)
            self._blocks.append(Block.get_block_by_type(dtype)(
                [schema_manager.get_column_index(schema.match_id_name)], should_compress=False))

        if schema.label_name:
            dtype = schema_manager.get_column_types(name=schema.label_name)
            self._blocks.append(Block.get_block_by_type(dtype)(
                [schema_manager.get_column_index(schema.label_name)], should_compress=False))

        if schema.weight_name:
            dtype = schema_manager.get_column_types(name=schema.weight_name)
            self._blocks.append(Block.get_block_by_type(dtype)(
                [schema_manager.get_column_index(schema.weight_name)], should_compress=False))

        for column_name in schema.columns:
            dtype = schema_manager.get_column_types(name=column_name)
            self._blocks.append(Block.get_block_by_type(dtype)(
                [schema_manager.get_column_index(column_name)], should_compress=True))

        new_blocks, to_compress_blocks = self.compress()

        self.reset_blocks(new_blocks)

    @property
    def blocks(self):
        return self._blocks

    @blocks.setter
    def blocks(self, blocks):
        self._blocks = blocks

    @property
    def column_block_mapping(self):
        return self._column_block_mapping

    def get_numeric_block(self):
        numeric_blocks = []
        for _blk in self._blocks:
            if _blk.is_numeric:
                numeric_blocks.append(_blk)

        return numeric_blocks

    def get_block_id(self, column_index) -> Tuple[int, int]:
        return self._column_block_mapping[column_index]

    def get_block(self, block_id):
        return self._blocks[block_id]

    def compress(self):
        compressible_blocks = dict()

        has_compressed = False
        for block_id, block in enumerate(self._blocks):
            if block.should_compress:
                has_compressed = True

            if block.block_type not in compressible_blocks:
                compressible_blocks[block.block_type] = []
            compressible_blocks[block.block_type].append(block)

        if not has_compressed:
            return self._blocks, []

        new_blocks, to_compress_blocks = [], []
        for block_type, block_list in compressible_blocks.items():
            _blocks = []
            for block in block_list:
                if block.should_compress:
                    _blocks.append(block)
                else:
                    new_blocks.append(block)

            if len(_blocks) > 1:
                to_compress_blocks.append(_blocks)
                """
                merge all column_indexes, use set instead of merge sort to avoid O(n * len(_blocks))
                """
                column_indexes_set = set()
                for block in _blocks:
                    column_indexes_set |= set(block.column_indexes)
                new_blocks.append(
                    Block.get_block_by_type(block_type)(sorted(list(column_indexes_set)), should_compress=True)
                )
            elif _blocks:
                new_blocks.append(_blocks[0])

        return new_blocks, to_compress_blocks

    def reset_blocks(self, blocks):
        self._blocks = blocks
        self._column_block_mapping = dict()
        for bid, blocks in enumerate(self._blocks):
            for idx in blocks.column_indexes:
                self._column_block_mapping[idx] = (bid, blocks.get_column_offset(idx))

    def apply(self, ):
        """
        make some columns to some other type

        """

    def to_dict(self):
        """
        deserialize
        """
        return dict(
            blocks=[
                blk.to_dict() for blk in self._blocks
            ]
        )

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
        for col_index in indexes:
            bid = self._column_block_mapping[col_index]
            if bid not in col_index:
                block_index_mapping[bid] = []
            block_index_mapping[bid] = col_index

        derived_blocks = []
        block_retrieval_indexes = []

        new_block_id = 0
        for bid, col_indexes in block_index_mapping.items():
            block, block_changed, retrieval_indexes = self._blocks[bid].derive_block(col_indexes)
            derived_blocks.append(block)
            block_retrieval_indexes.append((bid, new_block_id, block_changed, retrieval_indexes))
            new_block_id += 1

        block_manager.reset_blocks(derived_blocks)

        return block_manager, block_retrieval_indexes
