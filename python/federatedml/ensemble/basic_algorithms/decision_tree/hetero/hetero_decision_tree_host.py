import numpy as np
import functools
from federatedml.ensemble.basic_algorithms.decision_tree.tree_core.node import Node
from federatedml.util import LOGGER
from federatedml.protobuf.generated.boosting_tree_model_meta_pb2 import DecisionTreeModelMeta
from federatedml.protobuf.generated.boosting_tree_model_param_pb2 import DecisionTreeModelParam
from federatedml.ensemble.basic_algorithms.decision_tree.tree_core.decision_tree import DecisionTree
from federatedml.transfer_variable.transfer_class.hetero_decision_tree_transfer_variable import \
    HeteroDecisionTreeTransferVariable
from federatedml.util import consts
from federatedml.ensemble.basic_algorithms.decision_tree.tree_core.g_h_optim import PackedGHCompressor
from federatedml.ensemble.basic_algorithms.decision_tree.tree_core.splitter import SplitInfo


class HeteroDecisionTreeHost(DecisionTree):

    def __init__(self, tree_param):

        super(HeteroDecisionTreeHost, self).__init__(tree_param)

        # add host side feature importance support
        self.feature_importance_type = 'split'
        self.encrypted_grad_and_hess = None
        self.runtime_idx = 0
        self.sitename = consts.HOST  # will be modified in self.set_runtime_idx()
        self.complete_secure_tree = False
        self.host_party_idlist = []

        # feature shuffling / missing_dir masking
        self.feature_num = -1
        self.missing_dir_mask_left = {}  # mask for left direction
        self.missing_dir_mask_right = {}  # mask for right direction
        self.split_maskdict = {}  # save split value
        self.split_feature_dict = {}  # save split feature idx
        self.missing_dir_maskdict = {}  # save missing dir
        self.fid_bid_random_mapping = {}
        self.inverse_fid_bid_random_mapping = {}
        self.bin_num = None

        # goss subsample
        self.run_goss = False

        # transfer variable
        self.transfer_inst = HeteroDecisionTreeTransferVariable()

        # cipher compressing
        self.cipher_compressor = None
        self.run_cipher_compressing = True

        # code version control
        self.new_ver = True

        # multi mode
        self.mo_tree = False

    """
    Setting
    """

    def report_init_status(self):

        LOGGER.info('reporting initialization status')
        LOGGER.info('using new version code {}'.format(self.new_ver))
        if self.complete_secure_tree:
            LOGGER.info('running complete secure')
        if self.run_goss:
            LOGGER.info('running goss')
        if self.run_cipher_compressing:
            LOGGER.info('running cipher compressing')
        LOGGER.debug('bin num and feature num: {}/{}'.format(self.bin_num, self.feature_num))

    def init(self, flowid, runtime_idx, data_bin, bin_split_points, bin_sparse_points, bin_num,
             valid_features,
             complete_secure=False,
             goss_subsample=False,
             cipher_compressing=False,
             new_ver=True,
             mo_tree=False
             ):

        super(HeteroDecisionTreeHost, self).init_data_and_variable(flowid, runtime_idx, data_bin, bin_split_points,
                                                                   bin_sparse_points, valid_features, None)

        self.check_max_split_nodes()
        self.complete_secure_tree = complete_secure
        self.run_goss = goss_subsample
        self.bin_num = bin_num
        self.run_cipher_compressing = cipher_compressing
        self.feature_num = self.bin_split_points.shape[0]
        self.new_ver = new_ver
        self.mo_tree = mo_tree

        self.report_init_status()

    def set_host_party_idlist(self, l):
        self.host_party_idlist = l

    """
    Node encode/decode
    """

    def generate_missing_dir(self, dep, left_num=3, right_num=3):
        """
        randomly generate missing dir mask
        """
        rn = np.random.choice(range(left_num + right_num), left_num + right_num, replace=False)
        left_dir = rn[0:left_num]
        right_dir = rn[left_num:]
        self.missing_dir_mask_left[dep] = left_dir
        self.missing_dir_mask_right[dep] = right_dir

    @staticmethod
    def generate_fid_bid_random_mapping(feature_num, bin_num):

        total_id_num = feature_num * bin_num

        mapping = {}
        idx = 0
        id_list = np.random.choice(range(total_id_num), total_id_num, replace=False)
        for fid in range(feature_num):
            for bid in range(bin_num):
                mapping[(fid, bid)] = int(id_list[idx])
                idx += 1

        return mapping

    def save_split_info(self, etype="feature_idx", val=None, nid=None):

        if etype == "feature_idx":
            self.split_feature_dict[nid] = val

        elif etype == "feature_val":
            self.split_maskdict[nid] = val

        elif etype == "missing_dir":
            self.missing_dir_maskdict[nid] = val

        else:
            raise TypeError("encode type %s is not support!" % (str(etype)))

    @staticmethod
    def decode(dtype="feature_idx", val=None, nid=None, split_maskdict=None, missing_dir_maskdict=None):

        if dtype == "feature_idx":
            return val

        if dtype == "feature_val":
            if nid in split_maskdict:
                return split_maskdict[nid]
            else:
                raise ValueError("decode val %s cause error, can't recognize it!" % (str(val)))

        if dtype == "missing_dir":
            if nid in missing_dir_maskdict:
                return missing_dir_maskdict[nid]
            else:
                raise ValueError("decode val %s cause error, can't recognize it!" % (str(val)))

        return TypeError("decode type %s is not support!" % (str(dtype)))

    def generate_split_point_masking_variable(self, dep):
        # for split point masking
        self.generate_missing_dir(dep, 5, 5)
        self.fid_bid_random_mapping = self.generate_fid_bid_random_mapping(self.feature_num, self.bin_num)
        self.inverse_fid_bid_random_mapping = {v: k for k, v in self.fid_bid_random_mapping.items()}

    def unmask_split_info(self, split_info_list, inverse_mask_id_mapping, left_missing_dir, right_missing_dir):

        for split_info in split_info_list:
            if split_info.mask_id is not None:
                fid, bid = inverse_mask_id_mapping[split_info.mask_id]
                split_info.best_fid, split_info.best_bid = fid, bid
                masked_missing_dir = split_info.missing_dir
                if masked_missing_dir in left_missing_dir:
                    split_info.missing_dir = -1
                elif masked_missing_dir in right_missing_dir:
                    split_info.missing_dir = 1

        return split_info_list

    def record_split_info(self, split_info_list):

        final_split_info = []
        for i, split_info in enumerate(split_info_list):

            if split_info.best_fid != -1:
                LOGGER.debug('sitename is {}, self.sitename is {}'
                             .format(split_info.sitename, self.sitename))
                assert split_info.sitename == self.sitename
                self.save_split_info("feature_idx", split_info.best_fid, self.cur_to_split_nodes[i].id)
                assert split_info.best_fid is not None
                self.save_split_info("feature_val", split_info.best_bid, self.cur_to_split_nodes[i].id)
                self.save_split_info("missing_dir", split_info.missing_dir, self.cur_to_split_nodes[i].id)
                split_info.mask_id = None
            else:
                LOGGER.debug('this node can not be further split by host feature: {}'.format(split_info))

            final_split_info.append(split_info)

        return final_split_info

    """
    Federation Functions
    """

    def init_compressor_and_sync_gh(self):
        LOGGER.info("get encrypted grad and hess")

        if self.run_cipher_compressing:
            self.cipher_compressor = PackedGHCompressor(mo_mode=self.mo_tree)

        self.grad_and_hess = self.transfer_inst.encrypted_grad_and_hess.get(idx=0)

    def sync_node_positions(self, dep=-1):
        LOGGER.info("get tree node queue of depth {}".format(dep))
        node_positions = self.transfer_inst.node_positions.get(idx=0,
                                                               suffix=(dep,))
        return node_positions

    def sync_tree_node_queue(self, dep=-1):
        LOGGER.info("get tree node queue of depth {}".format(dep))
        self.cur_layer_nodes = self.transfer_inst.tree_node_queue.get(idx=0,
                                                                      suffix=(dep,))

    def sync_encrypted_splitinfo_host(self, encrypted_splitinfo_host, dep=-1, batch=-1):
        LOGGER.info("send encrypted splitinfo of depth {}, batch {}".format(dep, batch))
        self.transfer_inst.encrypted_splitinfo_host.remote(encrypted_splitinfo_host,
                                                           role=consts.GUEST,
                                                           idx=-1,
                                                           suffix=(dep, batch,))

    def sync_federated_best_splitinfo_host(self, dep=-1, batch=-1):
        LOGGER.info("get federated best splitinfo of depth {}, batch {}".format(dep, batch))
        federated_best_splitinfo_host = self.transfer_inst.federated_best_splitinfo_host.get(idx=0,
                                                                                             suffix=(dep, batch,))
        return federated_best_splitinfo_host

    def sync_dispatch_node_host(self, dep):
        LOGGER.info("get node from host to dispath, depth is {}".format(dep))
        dispatch_node_host = self.transfer_inst.dispatch_node_host.get(idx=0,
                                                                       suffix=(dep,))
        return dispatch_node_host

    def sync_dispatch_node_host_result(self, dispatch_node_host_result, dep=-1):
        LOGGER.info("send host dispatch result, depth is {}".format(dep))

        self.transfer_inst.dispatch_node_host_result.remote(dispatch_node_host_result,
                                                            role=consts.GUEST,
                                                            idx=-1,
                                                            suffix=(dep,))

    def sync_tree(self, ):
        LOGGER.info("sync tree from guest")
        self.tree_node = self.transfer_inst.tree.get(idx=0)

    def sync_predict_finish_tag(self, recv_times):
        LOGGER.info("get the {}-th predict finish tag from guest".format(recv_times))
        finish_tag = self.transfer_inst.predict_finish_tag.get(idx=0,
                                                               suffix=(recv_times,))
        return finish_tag

    def sync_predict_data(self, recv_times):
        LOGGER.info("srecv predict data to host, recv times is {}".format(recv_times))
        predict_data = self.transfer_inst.predict_data.get(idx=0,
                                                           suffix=(recv_times,))
        return predict_data

    def sync_data_predicted_by_host(self, predict_data, send_times):
        LOGGER.info("send predicted data by host, send times is {}".format(send_times))

        self.transfer_inst.predict_data_by_host.remote(predict_data,
                                                       role=consts.GUEST,
                                                       idx=0,
                                                       suffix=(send_times,))

    """
    Tree Updating
    """

    @staticmethod
    def assign_an_instance(value1, value2, sitename=None, decoder=None,
                           split_feature_dict=None,
                           bin_sparse_points=None,
                           use_missing=False, zero_as_missing=False,
                           split_maskdict=None,
                           missing_dir_maskdict=None):

        unleaf_state, fid, bid, node_sitename, nodeid, left_nodeid, right_nodeid = value1
        if node_sitename != sitename:
            return value1

        fid = split_feature_dict[nodeid]
        bid = decoder("feature_val", bid, nodeid, split_maskdict=split_maskdict)
        missing_dir = decoder("missing_dir", 1, nodeid, missing_dir_maskdict=missing_dir_maskdict)
        direction = HeteroDecisionTreeHost.make_decision(value2, fid, bid, missing_dir, use_missing, zero_as_missing,
                                                         bin_sparse_points[fid])

        return (unleaf_state, left_nodeid) if direction else (unleaf_state, right_nodeid)

    def assign_instances_to_new_node(self, dispatch_node_host, dep=-1):

        LOGGER.info("start to find host dispath of depth {}".format(dep))
        dispatch_node_method = functools.partial(self.assign_an_instance,
                                                 sitename=self.sitename,
                                                 decoder=self.decode,
                                                 split_maskdict=self.split_maskdict,
                                                 split_feature_dict=self.split_feature_dict,
                                                 bin_sparse_points=self.bin_sparse_points,
                                                 use_missing=self.use_missing,
                                                 zero_as_missing=self.zero_as_missing,
                                                 missing_dir_maskdict=self.missing_dir_maskdict)
        dispatch_node_host_result = dispatch_node_host.join(self.data_bin, dispatch_node_method)
        self.sync_dispatch_node_host_result(dispatch_node_host_result, dep)

    def update_instances_node_positions(self):

        # join data and inst2node_idx to update current node positions of samples
        self.data_with_node_assignments = self.data_bin.join(self.inst2node_idx, lambda v1, v2: (v1, v2))

    """
    Pre-Process / Post-Process
    """

    def remove_redundant_splitinfo_in_split_maskdict(self, split_nid_used):
        LOGGER.info("remove duplicated nodes from split mask dict")
        duplicated_nodes = set(self.split_maskdict.keys()) - set(split_nid_used)
        for nid in duplicated_nodes:
            del self.split_maskdict[nid]

    def convert_bin_to_real(self, split_maskdict):
        LOGGER.info("convert tree node bins to real value")
        split_nid_used = []

        for i in range(len(self.tree_node)):
            if self.tree_node[i].is_leaf:
                continue
            if self.tree_node[i].sitename == self.sitename:
                fid = self.split_feature_dict[self.tree_node[i].id]
                bid = self.decode("feature_val", self.tree_node[i].bid, self.tree_node[i].id, split_maskdict)
                # recover real split value
                real_splitval = self.bin_split_points[fid][bid]
                self.split_maskdict[self.tree_node[i].id] = real_splitval
                self.tree_node[i].fid = fid
                split_nid_used.append(self.tree_node[i].id)

        self.remove_redundant_splitinfo_in_split_maskdict(split_nid_used)

    def collect_host_split_feat_importance(self):

        for node in self.tree_node:
            if node.is_leaf:
                continue
            elif node.sitename == self.sitename:
                LOGGER.debug('sitename are {} {}'.format(node.sitename, self.sitename))
                fid = self.split_feature_dict[node.id]
                self.update_feature_importance(SplitInfo(sitename=self.sitename, best_fid=fid), False)

    """
    Split finding
    """

    def get_computing_inst2node_idx(self):
        if self.run_goss:
            inst2node_idx = self.inst2node_idx.join(self.grad_and_hess, lambda x1, x2: x1)
        else:
            inst2node_idx = self.inst2node_idx
        return inst2node_idx

    def compute_best_splits(self, cur_to_split_nodes: list, node_map, dep, batch):

        LOGGER.info('solving node batch {}, node num is {}'.format(batch, len(cur_to_split_nodes)))
        if not self.complete_secure_tree:
            data = self.data_with_node_assignments
            inst2node_idx = self.get_computing_inst2node_idx()
            node_sample_count = self.count_node_sample_num(inst2node_idx, node_map)
            LOGGER.debug('sample count is {}'.format(node_sample_count))
            acc_histograms = self.get_local_histograms(dep, data, self.grad_and_hess, node_sample_count,
                                                       cur_to_split_nodes, node_map, ret='tb',
                                                       hist_sub=True)

            split_info_table = self.splitter.host_prepare_split_points(
                histograms=acc_histograms,
                use_missing=self.use_missing,
                valid_features=self.valid_features,
                sitename=self.sitename,
                left_missing_dir=self.missing_dir_mask_left[dep],
                right_missing_dir=self.missing_dir_mask_right[dep],
                mask_id_mapping=self.fid_bid_random_mapping,
                batch_size=self.bin_num,
                cipher_compressor=self.cipher_compressor,
                shuffle_random_seed=np.abs(hash((dep, batch))))

            # test split info encryption
            self.transfer_inst.encrypted_splitinfo_host.remote(split_info_table,
                                                               role=consts.GUEST,
                                                               idx=-1,
                                                               suffix=(dep, batch))
            best_split_info = self.transfer_inst.federated_best_splitinfo_host.get(suffix=(dep, batch), idx=0)
            unmasked_split_info = self.unmask_split_info(
                best_split_info,
                self.inverse_fid_bid_random_mapping,
                self.missing_dir_mask_left[dep],
                self.missing_dir_mask_right[dep])
            self.record_split_info(unmasked_split_info)
        else:
            LOGGER.debug('skip splits computation')

    """
    Fit & Predict
    """

    def fit(self):

        LOGGER.info("begin to fit host decision tree")

        self.init_compressor_and_sync_gh()
        LOGGER.debug('grad and hess count {}'.format(self.grad_and_hess.count()))

        for dep in range(self.max_depth):

            LOGGER.debug('At dep {}'.format(dep))
            self.sync_tree_node_queue(dep)
            self.generate_split_point_masking_variable(dep)

            if len(self.cur_layer_nodes) == 0:
                break

            self.inst2node_idx = self.sync_node_positions(dep)
            self.update_instances_node_positions()

            batch = 0
            for i in range(0, len(self.cur_layer_nodes), self.max_split_nodes):
                self.cur_to_split_nodes = self.cur_layer_nodes[i: i + self.max_split_nodes]
                self.compute_best_splits(self.cur_to_split_nodes,
                                         node_map=self.get_node_map(self.cur_to_split_nodes),
                                         dep=dep, batch=batch)
                batch += 1
            dispatch_node_host = self.sync_dispatch_node_host(dep)
            self.assign_instances_to_new_node(dispatch_node_host, dep=dep)

        self.sync_tree()
        self.print_leafs()
        # convert bin index to real split-value, and remove redundant nid in split mask dict
        self.convert_bin_to_real(split_maskdict=self.split_maskdict)
        self.collect_host_split_feat_importance()
        LOGGER.info("fitting host decision tree done")

    @staticmethod
    def traverse_tree(predict_state, data_inst, tree_=None,
                      decoder=None, split_maskdict=None, sitename=consts.HOST,
                      use_missing=False, zero_as_missing=False,
                      missing_dir_maskdict=None):

        nid, _ = predict_state
        if tree_[nid].sitename != sitename:
            return predict_state

        while tree_[nid].sitename == sitename:
            nid = HeteroDecisionTreeHost.go_next_layer(tree_[nid], data_inst, use_missing, zero_as_missing,
                                                       None, split_maskdict, missing_dir_maskdict, decoder)

        return nid, 0

    def predict(self, data_inst):
        LOGGER.info("start to predict!")
        site_guest_send_times = 0
        while True:

            finish_tag = self.sync_predict_finish_tag(site_guest_send_times)
            if finish_tag is True:
                break

            predict_data = self.sync_predict_data(site_guest_send_times)

            traverse_tree = functools.partial(self.traverse_tree,
                                              tree_=self.tree_node,
                                              decoder=self.decode,
                                              split_maskdict=self.split_maskdict,
                                              sitename=self.sitename,
                                              use_missing=self.use_missing,
                                              zero_as_missing=self.zero_as_missing,
                                              missing_dir_maskdict=self.missing_dir_maskdict)
            predict_data = predict_data.join(data_inst, traverse_tree)

            self.sync_data_predicted_by_host(predict_data, site_guest_send_times)

            site_guest_send_times += 1

        LOGGER.info("predict finish!")

    """
    Tree Output
    """

    def get_model_meta(self):
        model_meta = DecisionTreeModelMeta()

        model_meta.max_depth = self.max_depth
        model_meta.min_sample_split = self.min_sample_split
        model_meta.min_impurity_split = self.min_impurity_split
        model_meta.min_leaf_node = self.min_leaf_node
        model_meta.use_missing = self.use_missing
        model_meta.zero_as_missing = self.zero_as_missing

        return model_meta

    def set_model_meta(self, model_meta):
        self.max_depth = model_meta.max_depth
        self.min_sample_split = model_meta.min_sample_split
        self.min_impurity_split = model_meta.min_impurity_split
        self.min_leaf_node = model_meta.min_leaf_node
        self.use_missing = model_meta.use_missing
        self.zero_as_missing = model_meta.zero_as_missing

    def get_model_param(self):
        model_param = DecisionTreeModelParam()
        for node in self.tree_node:
            model_param.tree_.add(id=node.id,
                                  sitename=node.sitename,
                                  fid=node.fid,
                                  bid=node.bid,
                                  weight=node.weight,
                                  is_leaf=node.is_leaf,
                                  left_nodeid=node.left_nodeid,
                                  right_nodeid=node.right_nodeid,
                                  missing_dir=node.missing_dir)

        model_param.split_maskdict.update(self.split_maskdict)
        model_param.missing_dir_maskdict.update(self.missing_dir_maskdict)

        return model_param

    def set_model_param(self, model_param):
        self.tree_node = []
        for node_param in model_param.tree_:
            _node = Node(id=node_param.id,
                         sitename=node_param.sitename,
                         fid=node_param.fid,
                         bid=node_param.bid,
                         weight=node_param.weight,
                         is_leaf=node_param.is_leaf,
                         left_nodeid=node_param.left_nodeid,
                         right_nodeid=node_param.right_nodeid,
                         missing_dir=node_param.missing_dir)

            self.tree_node.append(_node)

        self.split_maskdict = dict(model_param.split_maskdict)
        self.missing_dir_maskdict = dict(model_param.missing_dir_maskdict)

    """
    don t have to implements
    """

    def initialize_root_node(self, *args):
        pass

    def update_tree(self, *args):
        pass
