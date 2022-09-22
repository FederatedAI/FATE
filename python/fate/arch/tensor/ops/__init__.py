def broadcast_matmul(matrix, bc_matrix):
    return matrix.blocks.mapValues(lambda cb: cb @ bc_matrix)
