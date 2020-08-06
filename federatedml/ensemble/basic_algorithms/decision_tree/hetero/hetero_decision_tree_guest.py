import copy
import functools

from arch.api import session

from federatedml.ensemble.basic_algorithms.decision_tree.tree_core.decision_tree import DecisionTree
from federatedml.ensemble.basic_algorithms.decision_tree.tree_core.node import Node
from federatedml.transfer_variable.transfer_class.hetero_decision_tree_transfer_variable import \
    HeteroDecisionTreeTransferVariable
from federatedml.util import consts

from federatedml.protobuf.generated.boosting_tree_model_meta_pb2 import CriterionMeta
from federatedml.protobuf.generated.boosting_tree_model_meta_pb2 import DecisionTreeModelMeta
from federatedml.protobuf.generated.boosting_tree_model_param_pb2 import DecisionTreeModelParam

from arch.api.utils import log_utils

from federatedml.feature.fate_element_type import NoneType

LOGGER = log_utils.getLogger()


class HeteroDecisionTreeGuest(DecisionTree):

    def __init__(self, tree_param):
        super(HeteroDecisionTreeGuest, self).__init__(tree_param)
        self.encrypter = None
        self.encrypted_mode_calculator = None
        self.transfer_inst = HeteroDecisionTreeTransferVariable()

        self.sitename = consts.GUEST  # will be modified in self.set_runtime_idx()
        self.complete_secure_tree = False
        self.split_maskdict = {}
        self.missing_dir_maskdict = {}

        self.host_party_idlist = []

    def encode(self, etype="feature_idx", val=None, nid=None):
        if etype == "feature_idx":
            return val

        if etype == "feature_val":
            self.split_maskdict[nid] = val
            return None

        if etype == "missing_dir":
            self.missing_dir_maskdict[nid] = val
            return None

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

    def set_encrypter(self, encrypter):
        LOGGER.info("set encrypter")
        self.encrypter = encrypter

    def set_encrypted_mode_calculator(self, encrypted_mode_calculator):
        self.encrypted_mode_calculator = encrypted_mode_calculator

    def set_as_complete_secure_tree(self):
        self.complete_secure_tree = True

    def set_host_party_idlist(self, host_list):
        self.host_party_idlist = host_list

    def encrypt(self, val):
        return self.encrypter.encrypt(val)

    def decrypt(self, val):
        return self.encrypter.decrypt(val)

    def find_host_split(self, value):

        cur_split_node, encrypted_splitinfo_host = value
        sum_grad = cur_split_node.sum_grad
        sum_hess = cur_split_node.sum_hess
        best_gain = self.min_impurity_split - consts.FLOAT_ZERO
        best_idx = -1

        perform_recorder = {}
        gains = []

        for i in range(len(encrypted_splitinfo_host)):
            sum_grad_l, sum_hess_l = encrypted_splitinfo_host[i]
            sum_grad_l = self.decrypt(sum_grad_l)
            sum_hess_l = self.decrypt(sum_hess_l)
            sum_grad_r = sum_grad - sum_grad_l
            sum_hess_r = sum_hess - sum_hess_l
            gain = self.splitter.split_gain(sum_grad, sum_hess, sum_grad_l,
                                            sum_hess_l, sum_grad_r, sum_hess_r)

            perform_recorder[i] = gain
            gains.append(gain)

            if gain > self.min_impurity_split and gain > best_gain:
                best_gain = gain
                best_idx = i

        encrypted_best_gain = self.encrypt(best_gain)
        return best_idx, encrypted_best_gain, best_gain

    def find_best_split_guest_and_host(self, splitinfo_guest_host):

        best_gain_host = self.decrypt(splitinfo_guest_host[1].gain)
        best_gain_host_idx = 1
        for i in range(1, len(splitinfo_guest_host)):
            gain_host_i = self.decrypt(splitinfo_guest_host[i].gain)
            if best_gain_host < gain_host_i:
                best_gain_host = gain_host_i
                best_gain_host_idx = i

        # if merge_host_split_only is True, guest hists is None
        if splitinfo_guest_host[0] is not None and \
                splitinfo_guest_host[0].gain >= best_gain_host - consts.FLOAT_ZERO:
            best_splitinfo = splitinfo_guest_host[0]
        else:
            best_splitinfo = splitinfo_guest_host[best_gain_host_idx]
            LOGGER.debug('best split info is {}, {}'.format(best_splitinfo.sum_grad, best_splitinfo.sum_hess))

            # when this node can not be further split, host sum_grad and sum_hess is not an encrypted number but 0
            # so need type checking here
            best_splitinfo.sum_grad = self.decrypt(best_splitinfo.sum_grad) \
                if type(best_splitinfo.sum_grad) != int else best_splitinfo.sum_grad
            best_splitinfo.sum_hess = self.decrypt(best_splitinfo.sum_hess) \
                if type(best_splitinfo.sum_hess) != int else best_splitinfo.sum_hess
            best_splitinfo.gain = best_gain_host

        return best_splitinfo

    def sync_encrypted_grad_and_hess(self, idx=-1):
        encrypted_grad_and_hess = self.encrypted_mode_calculator.encrypt(self.grad_and_hess)

        self.transfer_inst.encrypted_grad_and_hess.remote(encrypted_grad_and_hess,
                                                          role=consts.HOST,
                                                          idx=idx)

    def sync_cur_to_split_nodes(self, cur_to_split_node, dep=-1, idx=-1):
        LOGGER.info("send tree node queue of depth {}".format(dep))
        mask_tree_node_queue = copy.deepcopy(cur_to_split_node)
        for i in range(len(mask_tree_node_queue)):
            mask_tree_node_queue[i] = Node(id=mask_tree_node_queue[i].id)

        self.transfer_inst.tree_node_queue.remote(mask_tree_node_queue,
                                                  role=consts.HOST,
                                                  idx=idx,
                                                  suffix=(dep,))

    def sync_node_positions(self, dep, idx=-1):
        LOGGER.info("send node positions of depth {}".format(dep))
        self.transfer_inst.node_positions.remote(self.inst2node_idx,
                                                 role=consts.HOST,
                                                 idx=idx,
                                                 suffix=(dep,))

    def sync_encrypted_splitinfo_host(self, dep=-1, batch=-1, idx=-1):
        LOGGER.info("get encrypted splitinfo of depth {}, batch {}".format(dep, batch))

        LOGGER.debug('host idx is {}'.format(idx))
        encrypted_splitinfo_host = self.transfer_inst.encrypted_splitinfo_host.get(idx=idx,
                                                                                   suffix=(dep, batch,))
        ret = []
        if idx == -1:
            for obj in encrypted_splitinfo_host:
                ret.append(obj.get_data())
        else:
            ret.append(encrypted_splitinfo_host.get_data())

        return ret

    def sync_federated_best_splitinfo_host(self, federated_best_splitinfo_host, dep=-1, batch=-1, idx=-1):
        LOGGER.info("send federated best splitinfo of depth {}, batch {}".format(dep, batch))
        self.transfer_inst.federated_best_splitinfo_host.remote(federated_best_splitinfo_host,
                                                                role=consts.HOST,
                                                                idx=idx,
                                                                suffix=(dep, batch,))

    def sync_final_split_host(self, dep=-1, batch=-1, idx=-1):
        LOGGER.info("get host final splitinfo of depth {}, batch {}".format(dep, batch))
        final_splitinfo_host = self.transfer_inst.final_splitinfo_host.get(idx=idx,
                                                                           suffix=(dep, batch,))
        return final_splitinfo_host if idx == -1 else [final_splitinfo_host]

    def sync_dispatch_node_host(self, dispatch_guest_data, dep=-1, idx=-1):

        LOGGER.info("send node to host to dispath, depth is {}".format(dep))
        self.transfer_inst.dispatch_node_host.remote(dispatch_guest_data,
                                                     role=consts.HOST,
                                                     idx=idx,
                                                     suffix=(dep,))
        LOGGER.info("get host dispatch result, depth is {}".format(dep))
        ret = self.transfer_inst.dispatch_node_host_result.get(idx=idx, suffix=(dep,))
        return ret if idx == -1 else [ret]

    def remove_sensitive_info(self):
        """
        host is not allowed to get weights/g/h
        """
        new_tree_ = copy.deepcopy(self.tree_node)
        for node in new_tree_:
            node.weight = None
            node.sum_grad = None
            node.sum_hess = None

        return new_tree_

    def sync_tree(self, idx=-1):
        LOGGER.info("sync tree to host")
        tree_nodes = self.remove_sensitive_info()
        self.transfer_inst.tree.remote(tree_nodes,
                                       role=consts.HOST,
                                       idx=idx)

    def merge_splitinfo(self, splitinfo_guest, splitinfo_host, merge_host_split_only=False):

        LOGGER.info("merge splitinfo, merge_host_split_only is {}".format(merge_host_split_only))

        if merge_host_split_only:
            splitinfo_guest = [None for i in range(len(splitinfo_host[0]))]

        merge_infos = []
        for i in range(len(splitinfo_guest)):
            splitinfo = [splitinfo_guest[i]]
            for j in range(len(splitinfo_host)):
                splitinfo.append(splitinfo_host[j][i])

            merge_infos.append(splitinfo)

        splitinfo_guest_host_table = session.parallelize(merge_infos,
                                                         include_key=False,
                                                         partition=self.data_bin.partitions)

        best_splitinfo_table = splitinfo_guest_host_table.mapValues(self.find_best_split_guest_and_host)
        best_splitinfos = [best_splitinfo[1] for best_splitinfo in best_splitinfo_table.collect()]

        return best_splitinfos

    def federated_find_split(self, dep=-1, batch=-1, idx=-1):

        LOGGER.info("federated find split of depth {}, batch {}".format(dep, batch))
        encrypted_splitinfo_host = self.sync_encrypted_splitinfo_host(dep, batch, idx=idx)

        for i in range(len(encrypted_splitinfo_host)):
            init_gain = self.min_impurity_split - consts.FLOAT_ZERO
            encrypted_init_gain = self.encrypter.encrypt(init_gain)
            best_splitinfo_host = [[-1, encrypted_init_gain] for j in range(len(self.cur_to_split_nodes))]
            best_gains = [init_gain for j in range(len(self.cur_to_split_nodes))]
            max_nodes = max(len(encrypted_splitinfo_host[i][j]) for j in range(len(self.cur_to_split_nodes)))
            for k in range(0, max_nodes, consts.MAX_FEDERATED_NODES):
                batch_splitinfo_host = [encrypted_splitinfo[k: k + consts.MAX_FEDERATED_NODES] for encrypted_splitinfo
                                        in encrypted_splitinfo_host[i]]

                encrypted_splitinfo_host_table = session.parallelize(zip(self.cur_to_split_nodes, batch_splitinfo_host),
                                                                     include_key=False,
                                                                     partition=self.data_bin.partitions)

                splitinfos = encrypted_splitinfo_host_table.mapValues(self.find_host_split).collect()

                for _, splitinfo in splitinfos:
                    LOGGER.debug('_, splitinfo are {}, {}'.format(_, splitinfo))
                    if best_splitinfo_host[_][0] == -1:
                        best_splitinfo_host[_] = list(splitinfo[:2])
                        best_gains[_] = splitinfo[2]
                    elif splitinfo[0] != -1 and splitinfo[2] > best_gains[_]:
                        best_splitinfo_host[_][0] = k + splitinfo[0]
                        best_splitinfo_host[_][1] = splitinfo[1]
                        best_gains[_] = splitinfo[2]

            if idx != -1:
                self.sync_federated_best_splitinfo_host(best_splitinfo_host, dep, batch, idx)
                break

            self.sync_federated_best_splitinfo_host(best_splitinfo_host, dep, batch, i)

    def initialize_root_node(self,):
        root_sum_grad, root_sum_hess = self.get_grad_hess_sum(self.grad_and_hess)
        root_node = Node(id=0, sitename=self.sitename, sum_grad=root_sum_grad, sum_hess=root_sum_hess,
                         weight=self.splitter.node_weight(root_sum_grad, root_sum_hess))
        return root_node

    def compute_best_splits(self, node_map, dep, batch_idx):

        acc_histograms = self.get_local_histograms(node_map, ret='tensor')
        best_split_info_guest = self.splitter.find_split(acc_histograms, self.valid_features,
                                                         self.data_bin.partitions, self.sitename,
                                                         self.use_missing, self.zero_as_missing)
        LOGGER.debug('computing local splits done')

        if self.complete_secure_tree:
            return best_split_info_guest

        self.federated_find_split(dep, batch_idx)
        host_split_info = self.sync_final_split_host(dep, batch_idx)

        cur_best_split = self.merge_splitinfo(splitinfo_guest=best_split_info_guest,
                                              splitinfo_host=host_split_info,
                                              merge_host_split_only=False)

        return cur_best_split

    def update_tree(self, split_info, reach_max_depth):

        LOGGER.info("update tree node, splitlist length is {}, tree node queue size is".format(
            len(split_info), len(self.cur_layer_nodes)))
        new_tree_node_queue = []
        for i in range(len(self.cur_layer_nodes)):
            sum_grad = self.cur_layer_nodes[i].sum_grad
            sum_hess = self.cur_layer_nodes[i].sum_hess
            if reach_max_depth or split_info[i].gain <= \
                    self.min_impurity_split + consts.FLOAT_ZERO:
                self.cur_layer_nodes[i].is_leaf = True
            else:
                self.cur_layer_nodes[i].left_nodeid = self.tree_node_num + 1
                self.cur_layer_nodes[i].right_nodeid = self.tree_node_num + 2
                self.tree_node_num += 2

                left_node = Node(id=self.cur_layer_nodes[i].left_nodeid,
                                 sitename=self.sitename,
                                 sum_grad=split_info[i].sum_grad,
                                 sum_hess=split_info[i].sum_hess,
                                 weight=self.splitter.node_weight(split_info[i].sum_grad, split_info[i].sum_hess))
                right_node = Node(id=self.cur_layer_nodes[i].right_nodeid,
                                  sitename=self.sitename,
                                  sum_grad=sum_grad - split_info[i].sum_grad,
                                  sum_hess=sum_hess - split_info[i].sum_hess,
                                  weight=self.splitter.node_weight(
                                      sum_grad - split_info[i].sum_grad,
                                      sum_hess - split_info[i].sum_hess))

                new_tree_node_queue.append(left_node)
                new_tree_node_queue.append(right_node)

                self.cur_layer_nodes[i].sitename = split_info[i].sitename
                if self.cur_layer_nodes[i].sitename == self.sitename:
                    self.cur_layer_nodes[i].fid = self.encode("feature_idx", split_info[i].best_fid)
                    self.cur_layer_nodes[i].bid = self.encode("feature_val", split_info[i].best_bid,
                                                              self.cur_layer_nodes[i].id)
                    self.cur_layer_nodes[i].missing_dir = self.encode("missing_dir",
                                                                      split_info[i].missing_dir,
                                                                      self.cur_layer_nodes[i].id)
                else:
                    self.cur_layer_nodes[i].fid = split_info[i].best_fid
                    self.cur_layer_nodes[i].bid = split_info[i].best_bid

                self.update_feature_importance(split_info[i])

            self.tree_node.append(self.cur_layer_nodes[i])

        self.cur_layer_nodes = new_tree_node_queue

    @staticmethod
    def assign_a_instance(value, tree_=None, decoder=None, sitename=consts.GUEST,
                      split_maskdict=None, bin_sparse_points=None,
                      use_missing=False, zero_as_missing=False,
                      missing_dir_maskdict=None):

        unleaf_state, nodeid = value[1]

        if tree_[nodeid].is_leaf is True:
            return tree_[nodeid].weight
        else:
            if tree_[nodeid].sitename == sitename:
                fid = decoder("feature_idx", tree_[nodeid].fid, split_maskdict=split_maskdict)
                bid = decoder("feature_val", tree_[nodeid].bid, nodeid, split_maskdict=split_maskdict)
                if not use_missing:
                    if value[0].features.get_data(fid, bin_sparse_points[fid]) <= bid:
                        return 1, tree_[nodeid].left_nodeid
                    else:
                        return 1, tree_[nodeid].right_nodeid
                else:
                    missing_dir = decoder("missing_dir", tree_[nodeid].missing_dir, nodeid,
                                          missing_dir_maskdict=missing_dir_maskdict)

                    missing_val = False
                    if zero_as_missing:
                        if value[0].features.get_data(fid, None) is None or \
                                value[0].features.get_data(fid) == NoneType():
                            missing_val = True
                    elif use_missing and value[0].features.get_data(fid) == NoneType():
                        missing_val = True

                    if missing_val:
                        if missing_dir == 1:
                            return 1, tree_[nodeid].right_nodeid
                        else:
                            return 1, tree_[nodeid].left_nodeid
                    else:
                        LOGGER.debug("fid is {}, bid is {}, sitename is {}".format(fid, bid, sitename))
                        if value[0].features.get_data(fid, bin_sparse_points[fid]) <= bid:
                            return 1, tree_[nodeid].left_nodeid
                        else:
                            return 1, tree_[nodeid].right_nodeid
            else:
                return (1, tree_[nodeid].fid, tree_[nodeid].bid, tree_[nodeid].sitename,
                        nodeid, tree_[nodeid].left_nodeid, tree_[nodeid].right_nodeid)

    def assign_instances_to_new_node(self, dep, reach_max_depth=False):

        LOGGER.info("redispatch node of depth {}".format(dep))
        dispatch_node_method = functools.partial(self.assign_a_instance,
                                                 tree_=self.tree_node,
                                                 decoder=self.decode,
                                                 sitename=self.sitename,
                                                 split_maskdict=self.split_maskdict,
                                                 bin_sparse_points=self.bin_sparse_points,
                                                 use_missing=self.use_missing,
                                                 zero_as_missing=self.zero_as_missing,
                                                 missing_dir_maskdict=self.missing_dir_maskdict)

        dispatch_guest_result = self.data_with_node_assignments.mapValues(dispatch_node_method)
        LOGGER.info("remask dispatch node result of depth {}".format(dep))

        dispatch_to_host_result = dispatch_guest_result.filter(
            lambda key, value: isinstance(value, tuple) and len(value) > 2)

        dispatch_guest_result = dispatch_guest_result.subtractByKey(dispatch_to_host_result)
        leaf = dispatch_guest_result.filter(lambda key, value: isinstance(value, tuple) is False)

        if self.sample_weights is None:
            self.sample_weights = leaf
        else:
            self.sample_weights = self.sample_weights.union(leaf)

        if reach_max_depth:
            return

        dispatch_guest_result = dispatch_guest_result.subtractByKey(leaf)

        dispatch_node_host_result = self.sync_dispatch_node_host(dispatch_to_host_result, dep)

        self.inst2node_idx = None
        for idx in range(len(dispatch_node_host_result)):
            if self.inst2node_idx is None:
                self.inst2node_idx = dispatch_node_host_result[idx]
            else:
                self.inst2node_idx = self.inst2node_idx.join(dispatch_node_host_result[idx],
                                                             lambda unleaf_state_nodeid1, unleaf_state_nodeid2:
                                                             unleaf_state_nodeid1 if len(
                                                             unleaf_state_nodeid1) == 2 else unleaf_state_nodeid2)

        self.inst2node_idx = self.inst2node_idx.union(dispatch_guest_result)

    def convert_bin_to_real(self):
        LOGGER.info("convert tree node bins to real value")
        for i in range(len(self.tree_node)):
            if self.tree_node[i].is_leaf is True:
                continue
            if self.tree_node[i].sitename == self.sitename:
                fid = self.decode("feature_idx", self.tree_node[i].fid, split_maskdict=self.split_maskdict)
                bid = self.decode("feature_val", self.tree_node[i].bid, self.tree_node[i].id, self.split_maskdict)
                real_splitval = self.encode("feature_val", self.bin_split_points[fid][bid], self.tree_node[i].id)
                self.tree_node[i].bid = real_splitval

    def fit(self):

        LOGGER.debug('fitting a hetero decision tree')

        self.sync_encrypted_grad_and_hess()
        root_node = self.initialize_root_node()
        self.cur_layer_nodes = [root_node]
        self.inst2node_idx = self.assign_instance_to_root_node(self.data_bin, root_node_id=root_node.id)

        for dep in range(self.max_depth):

            self.sync_cur_to_split_nodes(self.cur_layer_nodes, dep)
            if len(self.cur_layer_nodes) == 0:
                break

            self.sync_node_positions(dep)
            self.data_with_node_assignments = self.data_bin.join(self.inst2node_idx, lambda data_inst, dispatch_info: (
                                                                      data_inst, dispatch_info))

            split_info = []
            for batch_idx, i in enumerate(range(0, len(self.cur_layer_nodes), self.max_split_nodes)):

                self.cur_to_split_nodes = self.cur_layer_nodes[i: i + self.max_split_nodes]
                cur_splitinfos = self.compute_best_splits(self.get_node_map(self.cur_to_split_nodes), dep, batch_idx, )
                split_info.extend(cur_splitinfos)

            self.update_tree(split_info, False)
            self.assign_instances_to_new_node(dep)

        if self.cur_layer_nodes:
            self.update_tree([], True)
            self.data_with_node_assignments = self.data_bin.join(self.inst2node_idx, lambda data_inst, dispatch_info: (
                data_inst, dispatch_info))
            self.assign_instances_to_new_node(self.max_depth, reach_max_depth=True)

        self.convert_bin_to_real()
        self.sync_tree()
        LOGGER.info("tree node num is %d" % len(self.tree_node))
        LOGGER.info("end to fit guest decision tree")

    def sync_predict_finish_tag(self, finish_tag, send_times):
        LOGGER.info("send the {}-th predict finish tag {} to host".format(finish_tag, send_times))

        self.transfer_inst.predict_finish_tag.remote(finish_tag,
                                                     role=consts.HOST,
                                                     idx=-1,
                                                     suffix=(send_times,))

    def sync_predict_data(self, predict_data, send_times):
        LOGGER.info("send predict data to host, sending times is {}".format(send_times))
        self.transfer_inst.predict_data.remote(predict_data,
                                               role=consts.HOST,
                                               idx=-1,
                                               suffix=(send_times,))

    def sync_data_predicted_by_host(self, send_times):
        LOGGER.info("get predicted data by host, recv times is {}".format(send_times))
        predict_data = self.transfer_inst.predict_data_by_host.get(idx=-1,
                                                                   suffix=(send_times,))
        return predict_data

    @staticmethod
    def traverse_tree(predict_state, data_inst, tree_=None,
                      decoder=None, sitename=consts.GUEST, split_maskdict=None,
                      use_missing=None, zero_as_missing=None, missing_dir_maskdict=None, return_leaf_id=False):

        nid, tag = predict_state

        while tree_[nid].sitename == sitename:
            if tree_[nid].is_leaf is True:
                return tree_[nid].weight if not return_leaf_id else nid

            fid = decoder("feature_idx", tree_[nid].fid, split_maskdict=split_maskdict)
            bid = decoder("feature_val", tree_[nid].bid, nid, split_maskdict=split_maskdict)
            if use_missing:
                missing_dir = decoder("missing_dir", 1, nid, missing_dir_maskdict=missing_dir_maskdict)
            else:
                missing_dir = 1

            if use_missing and zero_as_missing:
                missing_dir = decoder("missing_dir", 1, nid, missing_dir_maskdict=missing_dir_maskdict)
                if data_inst.features.get_data(fid) == NoneType() or data_inst.features.get_data(fid, None) is None:
                    if missing_dir == 1:
                        nid = tree_[nid].right_nodeid
                    else:
                        nid = tree_[nid].left_nodeid
                elif data_inst.features.get_data(fid) <= bid:
                    nid = tree_[nid].left_nodeid
                else:
                    nid = tree_[nid].right_nodeid
            elif data_inst.features.get_data(fid) == NoneType():
                if missing_dir == 1:
                    nid = tree_[nid].right_nodeid
                else:
                    nid = tree_[nid].left_nodeid
            elif data_inst.features.get_data(fid, 0) <= bid:
                nid = tree_[nid].left_nodeid
            else:
                nid = tree_[nid].right_nodeid

        return nid, 1

    def predict(self, data_inst):

        LOGGER.info("start to predict!")
        predict_data = data_inst.mapValues(lambda inst: (0, 1))
        site_host_send_times = 0
        predict_result = None

        while True:
            traverse_tree = functools.partial(self.traverse_tree,
                                              tree_=self.tree_node,
                                              decoder=self.decode,
                                              sitename=self.sitename,
                                              split_maskdict=self.split_maskdict,
                                              use_missing=self.use_missing,
                                              zero_as_missing=self.zero_as_missing,
                                              missing_dir_maskdict=self.missing_dir_maskdict,
                                              return_leaf_id=False)

            predict_data = predict_data.join(data_inst, traverse_tree)
            predict_leaf = predict_data.filter(lambda key, value: isinstance(value, tuple) is False)
            if predict_result is None:
                predict_result = predict_leaf
            else:
                predict_result = predict_result.union(predict_leaf)

            predict_data = predict_data.subtractByKey(predict_leaf)

            unleaf_node_count = predict_data.count()

            if unleaf_node_count == 0:
                self.sync_predict_finish_tag(True, site_host_send_times)
                break

            self.sync_predict_finish_tag(False, site_host_send_times)
            self.sync_predict_data(predict_data, site_host_send_times)

            predict_data_host = self.sync_data_predicted_by_host(site_host_send_times)
            for i in range(len(predict_data_host)):
                predict_data = predict_data.join(predict_data_host[i],
                                                 lambda state1_nodeid1, state2_nodeid2:
                                                 state1_nodeid1 if state1_nodeid1[
                                                                       1] == 0 else state2_nodeid2)

            site_host_send_times += 1

        LOGGER.info("predict finish!")
        return predict_result

    def get_model_meta(self):

        model_meta = DecisionTreeModelMeta()
        model_meta.criterion_meta.CopyFrom(CriterionMeta(criterion_method=self.criterion_method,
                                                         criterion_param=self.criterion_params))

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
        self.criterion_method = model_meta.criterion_meta.criterion_method
        self.criterion_params = list(model_meta.criterion_meta.criterion_param)
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

