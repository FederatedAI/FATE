from federatedml.param.boosting_param import DecisionTreeParam
from federatedml.ensemble.basic_algorithms import HeteroFastDecisionTreeGuest, HeteroFastDecisionTreeHost, \
    HeteroDecisionTreeGuest, HeteroDecisionTreeHost
from federatedml.util import consts


def produce_hetero_tree_learner(role, tree_param: DecisionTreeParam, flow_id, data_bin, bin_split_points,
                                bin_sparse_points, task_type, valid_features, host_party_list,
                                runtime_idx,
                                cipher_compress=True,
                                mo_tree=False,
                                class_num=1,
                                g_h=None, encrypter=None,  # guest only
                                goss_subsample=False, complete_secure=False,
                                max_sample_weights=1.0,
                                bin_num=None,  # host only
                                fast_sbt=False,
                                tree_type=None, target_host_id=None,  # fast sbt only
                                guest_depth=2, host_depth=3  # fast sbt only
                                ):
    if role == consts.GUEST:
        if not fast_sbt:
            tree = HeteroDecisionTreeGuest(tree_param)
        else:
            tree = HeteroFastDecisionTreeGuest(tree_param)
            tree.set_tree_work_mode(tree_type, target_host_id)
            tree.set_layered_depth(guest_depth, host_depth)

        tree.init(flowid=flow_id,
                  data_bin=data_bin,
                  bin_split_points=bin_split_points,
                  bin_sparse_points=bin_sparse_points,
                  grad_and_hess=g_h,
                  encrypter=encrypter,
                  task_type=task_type,
                  valid_features=valid_features,
                  host_party_list=host_party_list,
                  runtime_idx=runtime_idx,
                  goss_subsample=goss_subsample,
                  complete_secure=complete_secure,
                  cipher_compressing=cipher_compress,
                  max_sample_weight=max_sample_weights,
                  mo_tree=mo_tree,
                  class_num=class_num
                  )

    elif role == consts.HOST:
        if not fast_sbt:
            tree = HeteroDecisionTreeHost(tree_param)
        else:
            tree = HeteroFastDecisionTreeHost(tree_param)
            tree.set_tree_work_mode(tree_type, target_host_id)
            tree.set_layered_depth(guest_depth, host_depth)
            tree.set_self_host_id(runtime_idx)
            tree.set_host_party_idlist(host_party_list)

        tree.init(flowid=flow_id,
                  valid_features=valid_features,
                  data_bin=data_bin,
                  bin_split_points=bin_split_points,
                  bin_sparse_points=bin_sparse_points,
                  runtime_idx=runtime_idx,
                  goss_subsample=goss_subsample,
                  complete_secure=complete_secure,
                  cipher_compressing=cipher_compress,
                  bin_num=bin_num,
                  mo_tree=mo_tree
                  )

    else:
        raise ValueError('unknown role: {}'.format(role))

    return tree


def load_hetero_tree_learner(role, tree_param, model_meta, model_param, flow_id, runtime_idx, host_party_list=None,
                             fast_sbt=False, tree_type=None, target_host_id=None):
    if role == consts.HOST:

        if fast_sbt:
            tree = HeteroFastDecisionTreeHost(tree_param)
        else:
            tree = HeteroDecisionTreeHost(tree_param)

        tree.load_model(model_meta, model_param)
        tree.set_flowid(flow_id)
        tree.set_runtime_idx(runtime_idx)

        if fast_sbt:
            tree.set_tree_work_mode(tree_type, target_host_id)
            tree.set_self_host_id(runtime_idx)

    elif role == consts.GUEST:

        if fast_sbt:
            tree = HeteroFastDecisionTreeGuest(tree_param)
        else:
            tree = HeteroDecisionTreeGuest(tree_param)

        tree.load_model(model_meta, model_param)
        tree.set_flowid(flow_id)
        tree.set_runtime_idx(runtime_idx)
        tree.set_host_party_idlist(host_party_list)

        if fast_sbt:
            tree.set_tree_work_mode(tree_type, target_host_id)

    else:
        raise ValueError('unknown role: {}'.format(role))

    return tree
