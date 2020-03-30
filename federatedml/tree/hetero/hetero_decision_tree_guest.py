#!/usr/bin/env python    
# -*- coding: utf-8 -*- 

#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
################################################################################
#
#
################################################################################

# =============================================================================
# HeteroDecisionTreeGuest
# =============================================================================

import copy
import functools

from arch.api import session
from arch.api.utils import log_utils
from federatedml.feature.fate_element_type import NoneType
from federatedml.protobuf.generated.boosting_tree_model_meta_pb2 import CriterionMeta
from federatedml.protobuf.generated.boosting_tree_model_meta_pb2 import DecisionTreeModelMeta
from federatedml.protobuf.generated.boosting_tree_model_param_pb2 import DecisionTreeModelParam
from federatedml.transfer_variable.transfer_class.hetero_decision_tree_transfer_variable import \
    HeteroDecisionTreeTransferVariable
from federatedml.tree import DecisionTree
from federatedml.tree import FeatureHistogram
from federatedml.tree import Node
from federatedml.tree import Splitter
from federatedml.util import consts

LOGGER = log_utils.getLogger()


class HeteroDecisionTreeGuest(DecisionTree):
    def __init__(self, tree_param):
        LOGGER.info("hetero decision tree guest init!")
        super(HeteroDecisionTreeGuest, self).__init__(tree_param)
        self.splitter = Splitter(self.criterion_method, self.criterion_params, self.min_impurity_split,
                                 self.min_sample_split, self.min_leaf_node)

        self.data_bin = None
        self.grad_and_hess = None
        self.bin_split_points = None
        self.bin_sparse_points = None
        self.data_bin_with_node_dispatch = None
        self.node_dispatch = None
        self.infos = None
        self.valid_features = None
        self.encrypter = None
        self.encrypted_mode_calculator = None
        self.best_splitinfo_guest = None
        self.tree_node_queue = None
        self.cur_split_nodes = None
        self.tree_ = []
        self.tree_node_num = 0
        self.split_maskdict = {}
        self.missing_dir_maskdict = {}
        self.transfer_inst = HeteroDecisionTreeTransferVariable()
        self.predict_weights = None
        self.host_party_idlist = []
        self.runtime_idx = 0
        self.sitename = consts.GUEST
        self.feature_importances_ = {}

    def set_flowid(self, flowid=0):
        LOGGER.info("set flowid, flowid is {}".format(flowid))
        self.transfer_inst.set_flowid(flowid)

    def set_host_party_idlist(self, host_party_idlist):
        self.host_party_idlist = host_party_idlist

    def set_inputinfo(self, data_bin=None, grad_and_hess=None, bin_split_points=None, bin_sparse_points=None):
        LOGGER.info("set input info")
        self.data_bin = data_bin
        self.grad_and_hess = grad_and_hess
        self.bin_split_points = bin_split_points
        self.bin_sparse_points = bin_sparse_points

    def set_encrypter(self, encrypter):
        LOGGER.info("set encrypter")
        self.encrypter = encrypter

    def set_encrypted_mode_calculator(self, encrypted_mode_calculator):
        self.encrypted_mode_calculator = encrypted_mode_calculator

    def encrypt(self, val):
        return self.encrypter.encrypt(val)

    def decrypt(self, val):
        return self.encrypter.decrypt(val)

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
                raise ValueError("decode val %s cause error, can't reconize it!" % (str(val)))

        if dtype == "missing_dir":
            if nid in missing_dir_maskdict:
                return missing_dir_maskdict[nid]
            else:
                raise ValueError("decode val %s cause error, can't reconize it!" % (str(val)))

        return TypeError("decode type %s is not support!" % (str(dtype)))

    def set_valid_features(self, valid_features=None):
        LOGGER.info("set valid features")
        self.valid_features = valid_features

    def sync_encrypted_grad_and_hess(self):
        LOGGER.info("send encrypted grad and hess to host")
        encrypted_grad_and_hess = self.encrypt_grad_and_hess()
        # LOGGER.debug("encrypted_grad_and_hess is {}".format(list(encrypted_grad_and_hess.collect())))

        self.transfer_inst.encrypted_grad_and_hess.remote(encrypted_grad_and_hess,
                                                          role=consts.HOST,
                                                          idx=-1)
        """
        federation.remote(obj=encrypted_grad_and_hess,
                          name=self.transfer_inst.encrypted_grad_and_hess.name,
                          tag=self.transfer_inst.generate_transferid(self.transfer_inst.encrypted_grad_and_hess),
                          role=consts.HOST,
                          idx=-1)
        """

    def encrypt_grad_and_hess(self):
        LOGGER.info("start to encrypt grad and hess")
        encrypted_grad_and_hess = self.encrypted_mode_calculator.encrypt(self.grad_and_hess)
        return encrypted_grad_and_hess

    def get_grad_hess_sum(self, grad_and_hess_table):
        LOGGER.info("calculate the sum of grad and hess")
        grad, hess = grad_and_hess_table.reduce(
            lambda value1, value2: (value1[0] + value2[0], value1[1] + value2[1]))
        return grad, hess

    def dispatch_all_node_to_root(self, root_id=0):
        LOGGER.info("dispatch all node to root")
        self.node_dispatch = self.data_bin.mapValues(lambda data_inst: (1, root_id))

    def get_histograms(self, node_map={}):
        LOGGER.info("start to get node histograms")
        acc_histograms = FeatureHistogram.calculate_histogram(
            self.data_bin_with_node_dispatch, self.grad_and_hess,
            self.bin_split_points, self.bin_sparse_points,
            self.valid_features, node_map,
            self.use_missing, self.zero_as_missing)

        return acc_histograms

    def sync_tree_node_queue(self, tree_node_queue, dep=-1):
        LOGGER.info("send tree node queue of depth {}".format(dep))
        mask_tree_node_queue = copy.deepcopy(tree_node_queue)
        for i in range(len(mask_tree_node_queue)):
            mask_tree_node_queue[i] = Node(id=mask_tree_node_queue[i].id)

        self.transfer_inst.tree_node_queue.remote(mask_tree_node_queue,
                                                  role=consts.HOST,
                                                  idx=-1,
                                                  suffix=(dep,))
        """
        federation.remote(obj=mask_tree_node_queue,
                          name=self.transfer_inst.tree_node_queue.name,
                          tag=self.transfer_inst.generate_transferid(self.transfer_inst.tree_node_queue, dep),
                          role=consts.HOST,
                          idx=-1)
        """

    def sync_node_positions(self, dep):
        LOGGER.info("send node positions of depth {}".format(dep))
        self.transfer_inst.node_positions.remote(self.node_dispatch,
                                                 role=consts.HOST,
                                                 idx=-1,
                                                 suffix=(dep,))
        """
        federation.remote(obj=self.node_dispatch,
                          name=self.transfer_inst.node_positions.name,
                          tag=self.transfer_inst.generate_transferid(self.transfer_inst.node_positions, dep),
                          role=consts.HOST,
                          idx=-1)
        """

    def sync_encrypted_splitinfo_host(self, dep=-1, batch=-1):
        LOGGER.info("get encrypted splitinfo of depth {}, batch {}".format(dep, batch))
        encrypted_splitinfo_host = self.transfer_inst.encrypted_splitinfo_host.get(idx=-1,
                                                                                   suffix=(dep, batch,))

        ret = []
        for obj in encrypted_splitinfo_host:
            ret.append(obj.get_data())
        """
        encrypted_splitinfo_host = federation.get(name=self.transfer_inst.encrypted_splitinfo_host.name,
                                                  tag=self.transfer_inst.generate_transferid(
                                                      self.transfer_inst.encrypted_splitinfo_host, dep, batch),
                                                  idx=-1)
        """
        return ret

    def sync_federated_best_splitinfo_host(self, federated_best_splitinfo_host, dep=-1, batch=-1, idx=-1):
        LOGGER.info("send federated best splitinfo of depth {}, batch {}".format(dep, batch))
        self.transfer_inst.federated_best_splitinfo_host.remote(federated_best_splitinfo_host,
                                                                role=consts.HOST,
                                                                idx=idx,
                                                                suffix=(dep, batch,))
        """
        federation.remote(obj=federated_best_splitinfo_host,
                          name=self.transfer_inst.federated_best_splitinfo_host.name,
                          tag=self.transfer_inst.generate_transferid(self.transfer_inst.federated_best_splitinfo_host,
                                                                     dep,
                                                                     batch),
                          role=consts.HOST,
                          idx=idx)
        """

    def find_host_split(self, value):
        cur_split_node, encrypted_splitinfo_host = value
        sum_grad = cur_split_node.sum_grad
        sum_hess = cur_split_node.sum_hess
        best_gain = self.min_impurity_split - consts.FLOAT_ZERO
        best_idx = -1

        for i in range(len(encrypted_splitinfo_host)):
            sum_grad_l, sum_hess_l = encrypted_splitinfo_host[i]
            sum_grad_l = self.decrypt(sum_grad_l)
            sum_hess_l = self.decrypt(sum_hess_l)
            sum_grad_r = sum_grad - sum_grad_l
            sum_hess_r = sum_hess - sum_hess_l
            gain = self.splitter.split_gain(sum_grad, sum_hess, sum_grad_l,
                                            sum_hess_l, sum_grad_r, sum_hess_r)

            if gain > self.min_impurity_split and gain > best_gain:
                best_gain = gain
                best_idx = i

        encrypted_best_gain = self.encrypt(best_gain)
        return best_idx, encrypted_best_gain, best_gain

    def federated_find_split(self, dep=-1, batch=-1):
        LOGGER.info("federated find split of depth {}, batch {}".format(dep, batch))
        encrypted_splitinfo_host = self.sync_encrypted_splitinfo_host(dep, batch)

        for i in range(len(encrypted_splitinfo_host)):
            best_splitinfo_host = [None for j in range(len(self.cur_split_nodes))]
            best_gains = [None for j in range(len(self.cur_split_nodes))]
            max_nodes = max(len(encrypted_splitinfo_host[i][j]) for j in range(len(self.cur_split_nodes)))
            for k in range(0, max_nodes, consts.MAX_FEDERATED_NODES):
                batch_splitinfo_host = [encrypted_splitinfo[k: k + consts.MAX_FEDERATED_NODES] for encrypted_splitinfo
                                        in encrypted_splitinfo_host[i]]
                encrypted_splitinfo_host_table = session.parallelize(zip(self.cur_split_nodes, batch_splitinfo_host),
                                                                     include_key=False,
                                                                     partition=self.data_bin._partitions)
                splitinfos = encrypted_splitinfo_host_table.mapValues(self.find_host_split).collect()
                for _, splitinfo in splitinfos:
                    if not best_splitinfo_host[_]:
                        best_splitinfo_host[_] = list(splitinfo[:2])
                        best_gains[_] = splitinfo[2]
                    elif splitinfo[0] != -1 and splitinfo[2] > best_gains[_]:
                        best_splitinfo_host[_][0] = k + splitinfo[0]
                        best_splitinfo_host[_][1] = splitinfo[1]
                        best_gains[_] = splitinfo[2]

            self.sync_federated_best_splitinfo_host(best_splitinfo_host, dep, batch, i)

    def sync_final_split_host(self, dep=-1, batch=-1):
        LOGGER.info("get host final splitinfo of depth {}, batch {}".format(dep, batch))
        final_splitinfo_host = self.transfer_inst.final_splitinfo_host.get(idx=-1,
                                                                           suffix=(dep, batch,))
        """
        final_splitinfo_host = federation.get(name=self.transfer_inst.final_splitinfo_host.name,
                                              tag=self.transfer_inst.generate_transferid(
                                                  self.transfer_inst.final_splitinfo_host, dep, batch),
                                              idx=-1)
        """
        return final_splitinfo_host

    def find_best_split_guest_and_host(self, splitinfo_guest_host):
        best_gain_host = self.decrypt(splitinfo_guest_host[1].gain)
        best_gain_host_idx = 1
        for i in range(1, len(splitinfo_guest_host)):
            gain_host_i = self.decrypt(splitinfo_guest_host[i].gain)
            if best_gain_host < gain_host_i:
                best_gain_host = gain_host_i
                best_gain_host_idx = i

        if splitinfo_guest_host[0].gain >= best_gain_host - consts.FLOAT_ZERO:
            best_splitinfo = splitinfo_guest_host[0]
        else:
            best_splitinfo = splitinfo_guest_host[best_gain_host_idx]
            best_splitinfo.sum_grad = self.decrypt(best_splitinfo.sum_grad)
            best_splitinfo.sum_hess = self.decrypt(best_splitinfo.sum_hess)
            best_splitinfo.gain = best_gain_host

        return best_splitinfo

    def merge_splitinfo(self, splitinfo_guest, splitinfo_host):
        LOGGER.info("merge splitinfo")
        merge_infos = []
        for i in range(len(splitinfo_guest)):
            splitinfo = [splitinfo_guest[i]]
            for j in range(len(splitinfo_host)):
                splitinfo.append(splitinfo_host[j][i])

            merge_infos.append(splitinfo)

        splitinfo_guest_host_table = session.parallelize(merge_infos,
                                                         include_key=False,
                                                         partition=self.data_bin._partitions)
        best_splitinfo_table = splitinfo_guest_host_table.mapValues(self.find_best_split_guest_and_host)

        best_splitinfos = [None for i in range(len(merge_infos))]
        for _, best_splitinfo in best_splitinfo_table.collect():
            best_splitinfos[_] = best_splitinfo
        # best_splitinfos = [best_splitinfo[1] for best_splitinfo in best_splitinfo_table.collect()]

        return best_splitinfos

    def update_feature_importance(self, splitinfo):
        if self.feature_importance_type == "split":
            inc = 1
        elif self.feature_importance_type == "gain":
            inc = splitinfo.gain
        else:
            raise ValueError("feature importance type {} not support yet".format(self.feature_importance_type))

        sitename = splitinfo.sitename
        fid = splitinfo.best_fid

        if (sitename, fid) not in self.feature_importances_:
            self.feature_importances_[(sitename, fid)] = 0

        self.feature_importances_[(sitename, fid)] += inc

    def update_tree_node_queue(self, splitinfos, max_depth_reach):
        LOGGER.info("update tree node, splitlist length is {}, tree node queue size is".format(
            len(splitinfos), len(self.tree_node_queue)))
        new_tree_node_queue = []
        for i in range(len(self.tree_node_queue)):
            sum_grad = self.tree_node_queue[i].sum_grad
            sum_hess = self.tree_node_queue[i].sum_hess
            if max_depth_reach or splitinfos[i].gain <= \
                    self.min_impurity_split + consts.FLOAT_ZERO:
                self.tree_node_queue[i].is_leaf = True
            else:
                self.tree_node_queue[i].left_nodeid = self.tree_node_num + 1
                self.tree_node_queue[i].right_nodeid = self.tree_node_num + 2
                self.tree_node_num += 2

                left_node = Node(id=self.tree_node_queue[i].left_nodeid,
                                 sitename=self.sitename,
                                 sum_grad=splitinfos[i].sum_grad,
                                 sum_hess=splitinfos[i].sum_hess,
                                 weight=self.splitter.node_weight(splitinfos[i].sum_grad, splitinfos[i].sum_hess))
                right_node = Node(id=self.tree_node_queue[i].right_nodeid,
                                  sitename=self.sitename,
                                  sum_grad=sum_grad - splitinfos[i].sum_grad,
                                  sum_hess=sum_hess - splitinfos[i].sum_hess,
                                  weight=self.splitter.node_weight( \
                                      sum_grad - splitinfos[i].sum_grad,
                                      sum_hess - splitinfos[i].sum_hess))

                new_tree_node_queue.append(left_node)
                new_tree_node_queue.append(right_node)

                self.tree_node_queue[i].sitename = splitinfos[i].sitename
                if self.tree_node_queue[i].sitename == self.sitename:
                    self.tree_node_queue[i].fid = self.encode("feature_idx", splitinfos[i].best_fid)
                    self.tree_node_queue[i].bid = self.encode("feature_val", splitinfos[i].best_bid,
                                                              self.tree_node_queue[i].id)
                    self.tree_node_queue[i].missing_dir = self.encode("missing_dir",
                                                                      splitinfos[i].missing_dir,
                                                                      self.tree_node_queue[i].id)
                else:
                    self.tree_node_queue[i].fid = splitinfos[i].best_fid
                    self.tree_node_queue[i].bid = splitinfos[i].best_bid

                self.update_feature_importance(splitinfos[i])
            self.tree_.append(self.tree_node_queue[i])

        self.tree_node_queue = new_tree_node_queue

    @staticmethod
    def dispatch_node(value, tree_=None, decoder=None, sitename=consts.GUEST,
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

    def sync_dispatch_node_host(self, dispatch_guest_data, dep=-1):
        LOGGER.info("send node to host to dispath, depth is {}".format(dep))
        self.transfer_inst.dispatch_node_host.remote(dispatch_guest_data,
                                                     role=consts.HOST,
                                                     idx=-1,
                                                     suffix=(dep,))
        """
        federation.remote(obj=dispatch_guest_data,
                          name=self.transfer_inst.dispatch_node_host.name,
                          tag=self.transfer_inst.generate_transferid(self.transfer_inst.dispatch_node_host, dep),
                          role=consts.HOST,
                          idx=-1)
        """

    def sync_dispatch_node_host_result(self, dep=-1):
        LOGGER.info("get host dispatch result, depth is {}".format(dep))
        dispatch_node_host_result = self.transfer_inst.dispatch_node_host_result.get(idx=-1,
                                                                                     suffix=(dep,))
        """
        dispatch_node_host_result = federation.get(name=self.transfer_inst.dispatch_node_host_result.name,
                                                   tag=self.transfer_inst.generate_transferid(
                                                       self.transfer_inst.dispatch_node_host_result, dep),
                                                   idx=-1)
        """
        return dispatch_node_host_result

    def redispatch_node(self, dep=-1):
        LOGGER.info("redispatch node of depth {}".format(dep))
        dispatch_node_method = functools.partial(self.dispatch_node,
                                                 tree_=self.tree_,
                                                 decoder=self.decode,
                                                 sitename=self.sitename,
                                                 split_maskdict=self.split_maskdict,
                                                 bin_sparse_points=self.bin_sparse_points,
                                                 use_missing=self.use_missing,
                                                 zero_as_missing=self.zero_as_missing,
                                                 missing_dir_maskdict=self.missing_dir_maskdict)
        dispatch_guest_result = self.data_bin_with_node_dispatch.mapValues(dispatch_node_method)
        tree_node_num = self.tree_node_num
        LOGGER.info("remask dispatch node result of depth {}".format(dep))

        dispatch_to_host_result = dispatch_guest_result.filter(
            lambda key, value: isinstance(value, tuple) and len(value) > 2)

        dispatch_guest_result = dispatch_guest_result.subtractByKey(dispatch_to_host_result)
        leaf = dispatch_guest_result.filter(lambda key, value: isinstance(value, tuple) is False)
        if self.predict_weights is None:
            self.predict_weights = leaf
        else:
            self.predict_weights = self.predict_weights.union(leaf)

        dispatch_guest_result = dispatch_guest_result.subtractByKey(leaf)

        self.sync_dispatch_node_host(dispatch_to_host_result, dep)
        dispatch_node_host_result = self.sync_dispatch_node_host_result(dep)

        self.node_dispatch = None
        for idx in range(len(dispatch_node_host_result)):
            if self.node_dispatch is None:
                self.node_dispatch = dispatch_node_host_result[idx]
            else:
                self.node_dispatch = self.node_dispatch.join(dispatch_node_host_result[idx], \
                                                             lambda unleaf_state_nodeid1, unleaf_state_nodeid2: \
                                                                 unleaf_state_nodeid1 if len(
                                                                     unleaf_state_nodeid1) == 2 else unleaf_state_nodeid2)
        self.node_dispatch = self.node_dispatch.union(dispatch_guest_result)

    def sync_tree(self):
        LOGGER.info("sync tree to host")

        self.transfer_inst.tree.remote(self.tree_,
                                       role=consts.HOST,
                                       idx=-1)
        """
        federation.remote(obj=self.tree_,
                          name=self.transfer_inst.tree.name,
                          tag=self.transfer_inst.generate_transferid(self.transfer_inst.tree),
                          role=consts.HOST,
                          idx=-1)
        """

    def convert_bin_to_real(self):
        LOGGER.info("convert tree node bins to real value")
        for i in range(len(self.tree_)):
            if self.tree_[i].is_leaf is True:
                continue
            if self.tree_[i].sitename == self.sitename:
                fid = self.decode("feature_idx", self.tree_[i].fid, split_maskdict=self.split_maskdict)
                bid = self.decode("feature_val", self.tree_[i].bid, self.tree_[i].id, self.split_maskdict)
                real_splitval = self.encode("feature_val", self.bin_split_points[fid][bid], self.tree_[i].id)
                self.tree_[i].bid = real_splitval

    def fit(self):
        LOGGER.info("begin to fit guest decision tree")
        self.sync_encrypted_grad_and_hess()

        # LOGGER.debug("self.grad and hess is {}".format(list(self.grad_and_hess.collect())))
        root_sum_grad, root_sum_hess = self.get_grad_hess_sum(self.grad_and_hess)
        root_node = Node(id=0, sitename=self.sitename, sum_grad=root_sum_grad, sum_hess=root_sum_hess,
                         weight=self.splitter.node_weight(root_sum_grad, root_sum_hess))
        self.tree_node_queue = [root_node]

        self.dispatch_all_node_to_root()

        for dep in range(self.max_depth):
            LOGGER.info("start to fit depth {}, tree node queue size is {}".format(dep, len(self.tree_node_queue)))

            self.sync_tree_node_queue(self.tree_node_queue, dep)
            if len(self.tree_node_queue) == 0:
                break

            self.sync_node_positions(dep)

            self.data_bin_with_node_dispatch = self.data_bin.join(self.node_dispatch,
                                                                  lambda data_inst, dispatch_info: (
                                                                      data_inst, dispatch_info))

            batch = 0
            splitinfos = []
            for i in range(0, len(self.tree_node_queue), self.max_split_nodes):
                self.cur_split_nodes = self.tree_node_queue[i: i + self.max_split_nodes]

                node_map = {}
                node_num = 0
                for tree_node in self.cur_split_nodes:
                    node_map[tree_node.id] = node_num
                    node_num += 1

                acc_histograms = self.get_histograms(node_map=node_map)

                self.best_splitinfo_guest = self.splitter.find_split(acc_histograms, self.valid_features,
                                                                     self.data_bin._partitions,
                                                                     self.sitename,
                                                                     self.use_missing, self.zero_as_missing)
                self.federated_find_split(dep, batch)
                final_splitinfo_host = self.sync_final_split_host(dep, batch)

                cur_splitinfos = self.merge_splitinfo(self.best_splitinfo_guest, final_splitinfo_host)
                splitinfos.extend(cur_splitinfos)

                batch += 1

            max_depth_reach = True if dep + 1 == self.max_depth else False
            self.update_tree_node_queue(splitinfos, max_depth_reach)

            self.redispatch_node(dep)

        self.sync_tree()
        self.convert_bin_to_real()
        tree_ = self.tree_
        LOGGER.info("tree node num is %d" % len(tree_))
        LOGGER.info("end to fit guest decision tree")

    @staticmethod
    def traverse_tree(predict_state, data_inst, tree_=None,
                      decoder=None, sitename=consts.GUEST, split_maskdict=None,
                      use_missing=None, zero_as_missing=None, missing_dir_maskdict=None):
        nid, tag = predict_state

        while tree_[nid].sitename == sitename:
            if tree_[nid].is_leaf is True:
                return tree_[nid].weight

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

    def sync_predict_finish_tag(self, finish_tag, send_times):
        LOGGER.info("send the {}-th predict finish tag {} to host".format(finish_tag, send_times))

        self.transfer_inst.predict_finish_tag.remote(finish_tag,
                                                     role=consts.HOST,
                                                     idx=-1,
                                                     suffix=(send_times,))
        """
        federation.remote(obj=finish_tag,
                          name=self.transfer_inst.predict_finish_tag.name,
                          tag=self.transfer_inst.generate_transferid(self.transfer_inst.predict_finish_tag, send_times),
                          role=consts.HOST,
                          idx=-1)
        """

    def sync_predict_data(self, predict_data, send_times):
        LOGGER.info("send predict data to host, sending times is {}".format(send_times))
        self.transfer_inst.predict_data.remote(predict_data,
                                               role=consts.HOST,
                                               idx=-1,
                                               suffix=(send_times,))

        """
        federation.remote(obj=predict_data,
                          name=self.transfer_inst.predict_data.name,
                          tag=self.transfer_inst.generate_transferid(self.transfer_inst.predict_data, send_times),
                          role=consts.HOST,
                          idx=-1)
        """

    def sync_data_predicted_by_host(self, send_times):
        LOGGER.info("get predicted data by host, recv times is {}".format(send_times))
        predict_data = self.transfer_inst.predict_data_by_host.get(idx=-1,
                                                                   suffix=(send_times,))
        """
        predict_data = federation.get(name=self.transfer_inst.predict_data_by_host.name,
                                      tag=self.transfer_inst.generate_transferid(
                                          self.transfer_inst.predict_data_by_host, send_times),
                                      idx=-1)
        """
        return predict_data

    def predict(self, data_inst):
        LOGGER.info("start to predict!")
        predict_data = data_inst.mapValues(lambda data_inst: (0, 1))
        site_host_send_times = 0
        predict_result = None

        while True:
            traverse_tree = functools.partial(self.traverse_tree,
                                              tree_=self.tree_,
                                              decoder=self.decode,
                                              sitename=self.sitename,
                                              split_maskdict=self.split_maskdict,
                                              use_missing=self.use_missing,
                                              zero_as_missing=self.zero_as_missing,
                                              missing_dir_maskdict=self.missing_dir_maskdict)
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
        for node in self.tree_:
            model_param.tree_.add(id=node.id,
                                  sitename=node.sitename,
                                  fid=node.fid,
                                  bid=node.bid,
                                  weight=node.weight,
                                  is_leaf=node.is_leaf,
                                  left_nodeid=node.left_nodeid,
                                  right_nodeid=node.right_nodeid,
                                  missing_dir=node.missing_dir)
            LOGGER.debug("missing_dir is {}, sitename is {}, is_leaf is {}".format(node.missing_dir, node.sitename,
                                                                                   node.is_leaf))

        model_param.split_maskdict.update(self.split_maskdict)
        model_param.missing_dir_maskdict.update(self.missing_dir_maskdict)

        return model_param

    def set_model_param(self, model_param):
        self.tree_ = []
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

            self.tree_.append(_node)

        self.split_maskdict = dict(model_param.split_maskdict)
        self.missing_dir_maskdict = dict(model_param.missing_dir_maskdict)

    def get_model(self):
        model_meta = self.get_model_meta()
        model_param = self.get_model_param()

        return model_meta, model_param

    def load_model(self, model_meta=None, model_param=None):
        LOGGER.info("load tree model")
        self.set_model_meta(model_meta)
        self.set_model_param(model_param)

    def get_feature_importance(self):
        return self.feature_importances_
