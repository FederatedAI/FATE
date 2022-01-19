import numpy as np
from fate_arch.session import computing_session


# TODO
def random_sampling():
    """
    Normal random row subsample
    """
    pass


def goss_sampling(grad_and_hess, top_rate, other_rate):
    """
    sampling method introduced in LightGBM
    """
    sample_num = grad_and_hess.count()
    g_h_generator = grad_and_hess.collect()
    id_list, g_list, h_list = [], [], []
    for id_, g_h in g_h_generator:
        id_list.append(id_)
        g_list.append(g_h[0])
        h_list.append(g_h[1])

    id_type = type(id_list[0])
    id_list = np.array(id_list)

    g_arr = np.array(g_list).astype(np.float64)
    h_arr = np.array(h_list).astype(np.float64)

    g_sum_arr = np.abs(g_arr).sum(axis=1)  # if it is multi-classification case, we need to sum g
    abs_g_list_arr = g_sum_arr
    sorted_idx = np.argsort(-abs_g_list_arr, kind='stable')  # stable sample result

    a_part_num = int(sample_num * top_rate)
    b_part_num = int(sample_num * other_rate)

    if a_part_num == 0 or b_part_num == 0:
        raise ValueError('subsampled result is 0: top sample {}, other sample {}'.format(a_part_num, b_part_num))

    # index of a part
    a_sample_idx = sorted_idx[:a_part_num]

    # index of b part
    rest_sample_idx = sorted_idx[a_part_num:]
    b_sample_idx = np.random.choice(rest_sample_idx, size=b_part_num, replace=False)

    # small gradient sample weights
    amplify_weights = (1 - top_rate) / other_rate
    g_arr[b_sample_idx] *= amplify_weights
    h_arr[b_sample_idx] *= amplify_weights

    # get selected sample
    a_idx_set, b_idx_set = set(list(a_sample_idx)), set(list(b_sample_idx))
    idx_set = a_idx_set.union(b_idx_set)
    selected_idx = np.array(list(idx_set))
    selected_g, selected_h = g_arr[selected_idx], h_arr[selected_idx]
    selected_id = id_list[selected_idx]

    data = [(id_type(id_), (g, h)) for id_, g, h in zip(selected_id, selected_g, selected_h)]
    new_g_h_table = computing_session.parallelize(data, include_key=True, partition=grad_and_hess.partitions)

    return new_g_h_table
