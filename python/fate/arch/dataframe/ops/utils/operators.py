import numpy as np
from fate.arch.computing import is_table


def binary_operate(lhs, rhs, op, block_indexes, rhs_block_id=None):
    block_index_set = set(block_indexes)
    if isinstance(rhs, list):
        op_ret = lhs.mapValues(
            lambda blocks:
            [
                op(blocks[bid], rhs[bid]) if bid in block_index_set else blocks[bid]
                for bid in range(len(blocks))
            ]
        )
    elif isinstance(rhs, (bool, int, float, np.int32, np.float32, np.int64, np.float64, np.bool_)):
        op_ret = lhs.mapValues(
            lambda blocks:
            [
                op(blocks[bid], rhs) if bid in block_index_set else blocks[bid]
                for bid in range(len(blocks))
             ]
        )
    elif is_table(rhs):
        op_ret = lhs.join(rhs,
            lambda blocks1, blocks2:
            [
                op(blocks1[bid], blocks2[rhs_block_id]) if bid in block_index_set else blocks1[bid]
                for bid in range(len(blocks1))
            ]
        )
    else:
        raise ValueError(f"Not implement type between dataframe nad {type(rhs)}")

    return op_ret


def unary_operate(lhs, op, block_indexes):
    block_index_set = set(block_indexes)
    op_ret = lhs.mapValues(
        lambda blocks:
        [
            op(blocks[bid]) if bid in block_index_set
            else blocks[bid]
            for bid in range(len(blocks))
        ]
    )

    return op_ret
