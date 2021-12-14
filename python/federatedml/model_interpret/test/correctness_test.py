from federatedml.util import LOGGER
from federatedml.protobuf.generated.boosting_tree_model_param_pb2 import BoostingTreeModelParam
from federatedml.ensemble.boosting.boosting import Boosting
from federatedml.ensemble import HeteroDecisionTreeHost


def send_real_data(transfer_inst, arr, suffix='debug'):
    transfer_inst.remote(arr, suffix=suffix)


def get_real_data(transfer_inst, suffix='debug'):
    LOGGER.debug('get data from host')
    arr = transfer_inst.get(-1, suffix=suffix)
    LOGGER.debug('arr is {}'.format(arr))
    return arr


def extract_host_route(decision_tree_list):

    real_bid = {}
    host_fid = {}
    missing_dir = {}
    for tidx, tree in enumerate(decision_tree_list):

        bids = {}
        fids = {}
        missing = {}
        for node in tree.tree_node:
            if node.sitename == tree.sitename:
                bids[node.id] = tree.split_maskdict[node.id]
                fids[node.id] = node.fid
                missing[node.id] = tree.missing_dir_maskdict[node.id]

        # tidx == tree index
        real_bid[tidx] = bids
        host_fid[tidx] = fids
        missing_dir[tidx] = missing

    return real_bid, host_fid, missing_dir


def merge_hetero_model(tree_param: BoostingTreeModelParam, host_bid, host_fid, missing_dir, fid_offset, host_feat_num):

    """
    Mock host node, replaced by fed host feature
    """

    for t_idx, tree in enumerate(tree_param.trees_):
        node_bid = host_bid[t_idx]
        for nid in node_bid:
            tree.split_maskdict[nid] = node_bid[nid]
            tree.missing_dir_maskdict[nid] = missing_dir[t_idx][nid]
            tree.tree_[nid].fid = host_fid[t_idx][nid] + fid_offset

    for i in range(0, host_feat_num):
        tree_param.feature_name_fid_mapping[i+fid_offset] = str(i + fid_offset)

    return tree_param


def get_host_feat_num(example):
    inst = example[1]
    return len(inst.features)

# debug codes
# host_sample = get_real_data(self.transfer_variable.host_feat_num)
# bid_fid = get_real_data(self.transfer_variable.host_feat_num, suffix='hostmodel')[0]
# host_feat_num = get_host_feat_num(host_sample[0][0])
# guest_feat_num = len(data_inst.schema['header'])
# test_tree_param = copy.deepcopy(self.tree_model_param)
# tree_param = merge_hetero_model(test_tree_param, bid_fid[0], bid_fid[1], bid_fid[2],
#                                 guest_feat_num, host_feat_num)
# test_lgb_model = self.convert_sbt_to_lgb(tree_param, self.tree_model_meta)
# host_arr = []
# for sample in host_sample[0]:
#     host_arr.append(sample[1].features)
# host_arr = np.array(host_arr)
# LOGGER.debug('host arr is {}'.format(host_arr))
# complete_arr = np.concatenate([arr, host_arr], axis=1)
# contrib_test = test_lgb_model.predict(complete_arr, pred_contrib=True)

# from federatedml.model_interpret.test.correctness_test import send_real_data, get_real_data, extract_host_route, \
#     get_host_feat_num, merge_hetero_model

# # debug codes
# send_real_data(self.transfer_variable.host_feat_num, interpret_sample)
# self.load_host_boosting_model()
# host_bid, host_fid, missing_dir = extract_host_route(self.decision_tree_list)
# send_real_data(self.transfer_variable.host_feat_num, (host_bid, host_fid, missing_dir), suffix='hostmodel')
