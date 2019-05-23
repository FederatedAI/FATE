def get_mul_op_def_example():
    # This is an example of defining multiplication operation beaver triples
    mul_op_def, ops = create_mul_op_def(num_overlap_samples=20,
                                        num_non_overlap_samples_guest=80,
                                        guest_input_dim=32,
                                        host_input_dim=28,
                                        hidden_dim=24)
    return mul_op_def


def create_mul_op_def(num_overlap_samples, num_non_overlap_samples_guest, guest_input_dim, host_input_dim, hidden_dim):
    # num_samples_host = num_overlap_samples + num_non_overlap_samples_host
    num_samples_guest = num_overlap_samples + num_non_overlap_samples_guest
    ops = []
    mul_op_def = dict()

    #
    # mul operations for host
    #

    op_id = "mul_op_0"
    mul_op_def[op_id] = dict()
    mul_op_def[op_id]["X_shape"] = (num_overlap_samples, 1, hidden_dim)
    mul_op_def[op_id]["Y_shape"] = (num_overlap_samples, hidden_dim, hidden_dim)
    mul_op_def[op_id]["batch_size"] = 0
    mul_op_def[op_id]["is_constant"] = True
    mul_op_def[op_id]["mul_type"] = "matmul"
    mul_op_def[op_id]["batch_axis"] = 0
    ops.append(op_id)

    op_id = "mul_op_1_for_host"
    mul_op_def[op_id] = dict()
    mul_op_def[op_id]["X_shape"] = (num_overlap_samples, host_input_dim, hidden_dim)
    mul_op_def[op_id]["Y_shape"] = (num_overlap_samples, host_input_dim, hidden_dim)
    mul_op_def[op_id]["batch_size"] = 0
    mul_op_def[op_id]["is_constant"] = True
    mul_op_def[op_id]["mul_type"] = "multiply"
    mul_op_def[op_id]["batch_axis"] = 0
    ops.append(op_id)

    op_id = "mul_op_1_for_guest"
    mul_op_def[op_id] = dict()
    mul_op_def[op_id]["X_shape"] = (num_samples_guest, guest_input_dim, hidden_dim)
    mul_op_def[op_id]["Y_shape"] = (num_samples_guest, guest_input_dim, hidden_dim)
    mul_op_def[op_id]["batch_size"] = 0
    mul_op_def[op_id]["is_constant"] = True
    mul_op_def[op_id]["mul_type"] = "multiply"
    mul_op_def[op_id]["batch_axis"] = 0
    ops.append(op_id)

    op_id = "mul_op_2"
    mul_op_def[op_id] = dict()
    mul_op_def[op_id]["X_shape"] = (num_overlap_samples, hidden_dim)
    mul_op_def[op_id]["Y_shape"] = (num_overlap_samples, hidden_dim)
    mul_op_def[op_id]["batch_size"] = 0
    mul_op_def[op_id]["is_constant"] = True
    mul_op_def[op_id]["mul_type"] = "multiply"
    mul_op_def[op_id]["batch_axis"] = 0
    ops.append(op_id)

    op_id = "mul_op_3"
    mul_op_def[op_id] = dict()
    mul_op_def[op_id]["X_shape"] = (num_overlap_samples, 1, hidden_dim)
    mul_op_def[op_id]["Y_shape"] = (num_overlap_samples, hidden_dim, hidden_dim)
    mul_op_def[op_id]["batch_size"] = 0
    mul_op_def[op_id]["is_constant"] = True
    mul_op_def[op_id]["mul_type"] = "matmul"
    mul_op_def[op_id]["batch_axis"] = 0
    ops.append(op_id)

    op_id = "mul_op_4"
    mul_op_def[op_id] = dict()
    mul_op_def[op_id]["X_shape"] = (num_overlap_samples, hidden_dim)
    mul_op_def[op_id]["Y_shape"] = (num_overlap_samples, hidden_dim)
    mul_op_def[op_id]["batch_size"] = 0
    mul_op_def[op_id]["is_constant"] = True
    mul_op_def[op_id]["mul_type"] = "multiply"
    mul_op_def[op_id]["batch_axis"] = 0
    ops.append(op_id)

    op_id = "mul_op_5"
    mul_op_def[op_id] = dict()
    mul_op_def[op_id]["X_shape"] = (num_non_overlap_samples_guest, hidden_dim)
    mul_op_def[op_id]["Y_shape"] = (num_non_overlap_samples_guest, hidden_dim)
    mul_op_def[op_id]["batch_size"] = 0
    mul_op_def[op_id]["is_constant"] = True
    mul_op_def[op_id]["mul_type"] = "multiply"
    mul_op_def[op_id]["batch_axis"] = 0
    ops.append(op_id)

    op_id = "mul_op_6"
    mul_op_def[op_id] = dict()
    mul_op_def[op_id]["X_shape"] = (num_overlap_samples, hidden_dim)
    mul_op_def[op_id]["Y_shape"] = (num_overlap_samples, hidden_dim)
    mul_op_def[op_id]["batch_size"] = 0
    mul_op_def[op_id]["is_constant"] = True
    mul_op_def[op_id]["mul_type"] = "multiply"
    mul_op_def[op_id]["batch_axis"] = 0
    ops.append(op_id)

    op_id = "mul_op_7"
    mul_op_def[op_id] = dict()
    mul_op_def[op_id]["X_shape"] = (num_samples_guest, hidden_dim, hidden_dim)
    mul_op_def[op_id]["Y_shape"] = (num_samples_guest, hidden_dim, hidden_dim)
    mul_op_def[op_id]["batch_size"] = 0
    mul_op_def[op_id]["is_constant"] = True
    mul_op_def[op_id]["mul_type"] = "multiply"
    mul_op_def[op_id]["batch_axis"] = 0
    ops.append(op_id)

    op_id = "mul_op_8"
    mul_op_def[op_id] = dict()
    mul_op_def[op_id]["X_shape"] = (num_samples_guest, hidden_dim)
    mul_op_def[op_id]["Y_shape"] = (num_samples_guest, hidden_dim)
    mul_op_def[op_id]["batch_size"] = 0
    mul_op_def[op_id]["is_constant"] = True
    mul_op_def[op_id]["mul_type"] = "multiply"
    mul_op_def[op_id]["batch_axis"] = 0
    ops.append(op_id)

    #
    # mul operations for computing loss
    #

    op_id = "mul_op_9"
    mul_op_def[op_id] = dict()
    mul_op_def[op_id]["X_shape"] = (num_overlap_samples, 1, hidden_dim)
    mul_op_def[op_id]["Y_shape"] = (num_overlap_samples, hidden_dim, 1)
    mul_op_def[op_id]["batch_size"] = 0
    mul_op_def[op_id]["is_constant"] = True
    mul_op_def[op_id]["mul_type"] = "matmul"
    mul_op_def[op_id]["batch_axis"] = 0
    ops.append(op_id)

    op_id = "mul_op_10"
    mul_op_def[op_id] = dict()
    mul_op_def[op_id]["X_shape"] = (1, hidden_dim)
    mul_op_def[op_id]["Y_shape"] = (hidden_dim, hidden_dim)
    mul_op_def[op_id]["batch_size"] = 0
    mul_op_def[op_id]["is_constant"] = True
    mul_op_def[op_id]["mul_type"] = "matmul"
    mul_op_def[op_id]["batch_axis"] = 0
    ops.append(op_id)

    op_id = "mul_op_11"
    mul_op_def[op_id] = dict()
    mul_op_def[op_id]["X_shape"] = (1, hidden_dim)
    mul_op_def[op_id]["Y_shape"] = (hidden_dim, 1)
    mul_op_def[op_id]["batch_size"] = 0
    mul_op_def[op_id]["is_constant"] = True
    mul_op_def[op_id]["mul_type"] = "matmul"
    mul_op_def[op_id]["batch_axis"] = 0
    ops.append(op_id)

    op_id = "mul_op_12"
    mul_op_def[op_id] = dict()
    mul_op_def[op_id]["X_shape"] = (num_overlap_samples, hidden_dim)
    mul_op_def[op_id]["Y_shape"] = (num_overlap_samples, hidden_dim)
    mul_op_def[op_id]["batch_size"] = 0
    mul_op_def[op_id]["is_constant"] = True
    mul_op_def[op_id]["mul_type"] = "multiply"
    mul_op_def[op_id]["batch_axis"] = 0
    ops.append(op_id)

    return mul_op_def, ops
