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
import pandas as pd


def extract_columns(block_table, retrieval_block_indexes):
    """
    retrieval_block_indexes: list, each element: (src_block_id, dst_block_id, changed=True/False, block_indexes)
    """
    def _extract_columns(src_blocks):
        extract_blocks = [None] * len(retrieval_block_indexes)

        for src_block_id, dst_block_id, is_changed, block_column_indexes in retrieval_block_indexes:
            block = src_blocks[src_block_id]
            if is_changed:
                """
                multiple columns, maybe pandas or fate.arch.tensor object
                """
                if isinstance(block, pd.DataFrame):
                    extract_blocks[dst_block_id] = block.iloc[:, block_column_indexes]
                else:
                    extract_blocks[dst_block_id] = block[:, block_column_indexes]
            else:
                extract_blocks[dst_block_id] = block

        return extract_blocks

    extract_table = block_table.mapValues(_extract_columns)

    return extract_table
