
from typing import Any, overload


def matmul(left, right):
    ...

    """
    If both arguments are 2-D they are multiplied like conventional matrices.
    If either argument is N-D, N > 2, it is treated as a stack of matrices residing in the last two indexes and broadcast accordingly.
    If the first argument is 1-D, it is promoted to a matrix by prepending a 1 to its dimensions. After matrix multiplication the prepended 1 is removed.
    If the second argument is 1-D, it is promoted to a matrix by appending a 1 to its dimensions. After matrix multiplication the appended 1 is removed.
    (x, a, 1) (x, 1, b) -> (x, a, b)
    """
    lshape = self.shape
    rshape = other.shape
    if (s1 := lshape[-1]) != (s2 := rshape[-2:][0]):
        raise ValueError(
            "matmul: Input operand 1 has a mismatch in its core dimension 0,"
            f"signature (n?,k),(k,m?)->(n?,m?) (size {s1} is different from {s2})"
        )
    bs_shape = torch.broadcast_shapes(lshape[:-2], rshape[:-2])

    if other_d_axis := get_distributed_axis(other) is None:
        # last axis is distributed
        if self._d_axis == len(lshape) - 1:
            # (..., ?) x (...)
            raise ValueError(
                f"matmul: can't matmul distributed tensor with non distributed tensor with operand 0 last dim distributed"
            )

        else:
            # (..., d, ?, ...) x (...)
            def _map_func(block):
                return matmul(block, other)

            output_blocks_table = self._blocks_table.mapValues(_map_func)
    else:
        if self._d_axis == len(lshape) - 2:
            # (..., d, ?) x (..., ?, ...)
            raise ValueError(
                f"matmul: can't matmul distributed tensor with distributed tensor with operand 0 `-2 dim distributed`"
            )
        if self._d_axis == len(lshape) -1 and len(rshape) == 1:
            self._blocks_table.join(other._block_table).mapValues().reduce()
            # (..., d) x (d)
            ...
        if self._d_axis == len(lshape) -1 and len(rshape) - 2 == other_d_axis:
            # (..., d) x (..., d, ?)
            ...

        if len(lshape) - self._d_axis == len(rshape) - other_d_axis:
            # (..., d, a1, ..., ak) x (..., d, b1, ..., bk)
            ...

        raise ValueError(...)
