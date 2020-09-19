cimport numpy as np
from libc.stdint cimport uint64_t

import numpy as np

def _nms_gpu_post(np.ndarray[np.uint64_t, ndim=1] mask,
                  int n_bbox,
                  int threads_per_block,
                  int col_blocks
                  ):
    cdef:
        int i, j, nblock, index
        uint64_t inblock
        int n_selection = 0
        uint64_t one_ull = 1
        np.ndarray[np.int32_t, ndim=1] selection
        np.ndarray[np.uint64_t, ndim=1] remv

    selection = np.zeros((n_bbox,), dtype=np.int32)
    remv = np.zeros((col_blocks,), dtype=np.uint64)

    for i in range(n_bbox):
        nblock = i // threads_per_block
        inblock = i % threads_per_block

        if not (remv[nblock] & one_ull << inblock):
            selection[n_selection] = i
            n_selection += 1

            index = i * col_blocks
            for j in range(nblock, col_blocks):
                remv[j] |= mask[index + j]
    return selection, n_selection
