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

from arch.api.utils import log_utils

import functools
from arch.api import eggroll
from arch.api import federation
from arch.api.proto.boosting_tree_model_meta_pb2 import CriterionMeta
from arch.api.proto.boosting_tree_model_meta_pb2 import DecisionTreeModelMeta
from arch.api.proto.boosting_tree_model_param_pb2 import DecisionTreeModelParam
from arch.api.proto.boosting_tree_model_param_pb2 import NodeParam
import random
from federatedml.util import HeteroDecisionTreeTransferVariable
from federatedml.util import consts
from federatedml.tree import FeatureHistogram
from federatedml.tree import DecisionTree
from federatedml.tree import Splitter
from federatedml.tree import Node

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
        self.best_splitinfo_guest = None
        self.tree_node_queue = None
        self.cur_split_nodes = None
        self.tree_ = []
        self.tree_node_num = 0
        self.split_maskdict = {}
        self.transfer_inst = HeteroDecisionTreeTransferVariable()
        self.predict_weights = None

    def set_flowid(self, flowid=0):
        LOGGER.info("set flowid, flowid is {}".format(flowid))
        self.transfer_inst.set_flowid(flowid)

    def set_inputinfo(self, data_bin=None, grad_and_hess=None, bin_split_points=None, bin_sparse_points=None):
        LOGGER.info("set input info")
        self.data_bin = data_bin
        self.grad_and_hess = grad_and_hess
        self.bin_split_points = bin_split_points
        self.bin_sparse_points = bin_sparse_points

    def set_encrypter(self, encrypter):
        LOGGER.info("set encrypter")
        self.encrypter = encrypter

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

        raise TypeError("encode type %s is not support!" % (str(etype)))

    @staticmethod
    def decode(dtype="feature_idx", val=None, nid=None, split_maskdict=None):
        if dtype == "feature_idx":
            return val

        if dtype == "feature_val":
            if nid in split_maskdict:
                return split_maskdict[nid]
            else:
                raise ValueError("decode val %s cause error, can't reconize it!" % (str(val)))

        return TypeError("decode type %s is not support!" % (str(dtype)))

    def set_valid_features(self, valid_features=None):
        LOGGER.info("set valid features")
        self.valid_features = valid_features

    def sync_encrypted_grad_and_hess(self):
        LOGGER.info("send encrypted grad and hess to host")
        encrypted_grad_and_hess = self.encrypt_grad_and_hess()
        federation.remote(obj=encrypted_grad_and_hess,
                          name=self.transfer_inst.encrypted_grad_and_hess.name,
                          tag=self.transfer_inst.generate_transferid(self.transfer_inst.encrypted_grad_and_hess),
                          role=consts.HOST,
                          idx=0)

    def encrypt_grad_and_hess(self):
        LOGGER.info("start to encrypt grad and hess")
        encrypter = self.encrypter
        encrypted_grad_and_hess = self.grad_and_hess.mapValues(
            lambda grad_hess: (encrypter.encrypt(grad_hess[0]), encrypter.encrypt(grad_hess[1])))
        LOGGER.info("finish to encrypt grad and hess")
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
        histograms = FeatureHistogram.calculate_histogram(
            self.data_bin_with_node_dispatch, self.grad_and_hess,
            self.bin_split_points, self.bin_sparse_points,
            self.valid_features, node_map)
        acc_histograms = FeatureHistogram.accumulate_histogram(histograms)
        LOGGER.info("acc histogram shape is {}".format(len(acc_histograms)))
        return acc_histograms

    def sync_tree_node_queue(self, tree_node_queue, dep=-1):
        LOGGER.info("send tree node queue of depth {}".format(dep))
        mask_tree_node_queue = tree_node_queue.copy()
        for i in range(len(mask_tree_node_queue)):
            mask_tree_node_queue[i] = Node(id=mask_tree_node_queue[i].id)

        federation.remote(obj=mask_tree_node_queue,
                          name=self.transfer_inst.tree_node_queue.name,
                          tag=self.transfer_inst.generate_transferid(self.transfer_inst.tree_node_queue, dep),
                          role=consts.HOST,
                          idx=0)

    def sync_node_positions(self, dep):
        LOGGER.info("send node positions of depth {}".format(dep))
        federation.remote(obj=self.node_dispatch,
                          name=self.transfer_inst.node_positions.name,
                          tag=self.transfer_inst.generate_transferid(self.transfer_inst.node_positions, dep),
                          role=consts.HOST,
                          idx=0)

    def sync_encrypted_splitinfo_host(self, dep=-1, batch=-1):
        LOGGER.info("get encrypted splitinfo of depth {}, batch {}".format(dep, batch))
        encrypted_splitinfo_host = federation.get(name=self.transfer_inst.encrypted_splitinfo_host.name,
                                                  tag=self.transfer_inst.generate_transferid(
                                                      self.transfer_inst.encrypted_splitinfo_host, dep, batch),
                                                  idx=0)
        return encrypted_splitinfo_host

    def sync_federated_best_splitinfo_host(self, federated_best_splitinfo_host, dep=-1, batch=-1):
        LOGGER.info("send federated best splitinfo of depth {}, batch {}".format(dep, batch))
        federation.remote(obj=federated_best_splitinfo_host,
                          name=self.transfer_inst.federated_best_splitinfo_host.name,
                          tag=self.transfer_inst.generate_transferid(self.transfer_inst.federated_best_splitinfo_host,
                                                                     dep,
                                                                     batch),
                          role=consts.HOST,
                          idx=0)

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

        best_gain = self.encrypt(best_gain)
        return best_idx, best_gain

    def federated_find_split(self, dep=-1, batch=-1):
        LOGGER.info("federated find split of depth {}, batch {}".format(dep, batch))
        encrypted_splitinfo_host = self.sync_encrypted_splitinfo_host(dep, batch)

        LOGGER.info("begin to find split")
        encrypted_splitinfo_host_table = eggroll.parallelize(
            zip(self.cur_split_nodes, encrypted_splitinfo_host), include_key=False, partition=self.data_bin._partitions)

        splitinfos = encrypted_splitinfo_host_table.mapValues(self.find_host_split).collect()
        best_splitinfo_host = [splitinfo[1] for splitinfo in splitinfos]

        LOGGER.info("end to find split")
        self.sync_federated_best_splitinfo_host(best_splitinfo_host, dep, batch)

    def sync_final_split_host(self, dep=-1, batch=-1):
        LOGGER.info("get host final splitinfo of depth {}, batch {}".format(dep, batch))
        final_splitinfo_host = federation.get(name=self.transfer_inst.final_splitinfo_host.name,
                                              tag=self.transfer_inst.generate_transferid(
                                                  self.transfer_inst.final_splitinfo_host, dep, batch),
                                              idx=0)

        return final_splitinfo_host

    def find_best_split_guest_and_host(self, splitinfo_guest_host):
        splitinfo_guest, splitinfo_host = splitinfo_guest_host
        best_splitinfo = None
        gain_host = self.decrypt(splitinfo_host.gain)
        if splitinfo_guest.gain >= gain_host - consts.FLOAT_ZERO:
            best_splitinfo = splitinfo_guest
        else:
            best_splitinfo = splitinfo_host
            best_splitinfo.sum_grad = self.decrypt(best_splitinfo.sum_grad)
            best_splitinfo.sum_hess = self.decrypt(best_splitinfo.sum_hess)
            best_splitinfo.gain = gain_host

        return best_splitinfo

    def merge_splitinfo(self, splitinfo_guest, splitinfo_host):
        LOGGER.info("merge splitinfo")
        splitinfo_guest_host_table = eggroll.parallelize(zip(splitinfo_guest, splitinfo_host),
                                                         include_key=False,
                                                         partition=self.data_bin._partitions)
        best_splitinfo_table = splitinfo_guest_host_table.mapValues(self.find_best_split_guest_and_host)
        best_splitinfos = [best_splitinfo[1] for best_splitinfo in best_splitinfo_table.collect()]

        return best_splitinfos

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
                                 sitename=consts.GUEST,
                                 sum_grad=splitinfos[i].sum_grad,
                                 sum_hess=splitinfos[i].sum_hess,
                                 weight=self.splitter.node_weight(splitinfos[i].sum_grad, splitinfos[i].sum_hess))
                right_node = Node(id=self.tree_node_queue[i].right_nodeid,
                                  sitename=consts.GUEST,
                                  sum_grad=sum_grad - splitinfos[i].sum_grad,
                                  sum_hess=sum_hess - splitinfos[i].sum_hess,
                                  weight=self.splitter.node_weight( \
                                      sum_grad - splitinfos[i].sum_grad,
                                      sum_hess - splitinfos[i].sum_hess))

                new_tree_node_queue.append(left_node)
                new_tree_node_queue.append(right_node)
                LOGGER.info("tree_node_queue {} split!!!".format(self.tree_node_queue[i].id))

                self.tree_node_queue[i].sitename = splitinfos[i].sitename
                if self.tree_node_queue[i].sitename == consts.GUEST:
                    self.tree_node_queue[i].fid = self.encode("feature_idx", splitinfos[i].best_fid)
                    self.tree_node_queue[i].bid = self.encode("feature_val", splitinfos[i].best_bid,
                                                              self.tree_node_queue[i].id)
                else:
                    self.tree_node_queue[i].fid = splitinfos[i].best_fid
                    self.tree_node_queue[i].bid = splitinfos[i].best_bid

            self.tree_.append(self.tree_node_queue[i])

        self.tree_node_queue = new_tree_node_queue

    @staticmethod
    def dispatch_node(value, tree_=None, decoder=None,
                      split_maskdict=None, bin_sparse_points=None):
        unleaf_state, nodeid = value[1]
        if unleaf_state == 0:
            return value[1]

        if tree_[nodeid].is_leaf is True:
            return (0, nodeid)
        else:
            if tree_[nodeid].sitename == consts.GUEST:
                fid = decoder("feature_idx", tree_[nodeid].fid, split_maskdict=split_maskdict)
                bid = decoder("feature_val", tree_[nodeid].bid, nodeid, split_maskdict)
                if value[0].features.get_data(fid, bin_sparse_points[fid]) <= bid:
                    return (1, tree_[nodeid].left_nodeid)
                else:
                    return (1, tree_[nodeid].right_nodeid)
            else:
                return (1, tree_[nodeid].fid, tree_[nodeid].bid, \
                        nodeid, tree_[nodeid].left_nodeid, tree_[nodeid].right_nodeid)

    def sync_dispatch_node_host(self, dispatch_guest_data, dep=-1):
        LOGGER.info("send node to host to dispath, depth is {}".format(dep))
        federation.remote(obj=dispatch_guest_data,
                          name=self.transfer_inst.dispatch_node_host.name,
                          tag=self.transfer_inst.generate_transferid(self.transfer_inst.dispatch_node_host, dep),
                          role=consts.HOST,
                          idx=0)

    def sync_dispatch_node_host_result(self, dep=-1):
        LOGGER.info("get host dispatch result, depth is {}".format(dep))
        dispatch_node_host_result = federation.get(name=self.transfer_inst.dispatch_node_host_result.name,
                                                   tag=self.transfer_inst.generate_transferid(
                                                       self.transfer_inst.dispatch_node_host_result, dep),
                                                   idx=0)

        return dispatch_node_host_result

    def redispatch_node(self, dep=-1):
        LOGGER.info("redispatch node of depth {}".format(dep))
        dispatch_node_method = functools.partial(self.dispatch_node,
                                                 tree_=self.tree_,
                                                 decoder=self.decode,
                                                 split_maskdict=self.split_maskdict,
                                                 bin_sparse_points=self.bin_sparse_points)
        dispatch_guest_result = self.data_bin_with_node_dispatch.mapValues(dispatch_node_method)
        tree_node_num = self.tree_node_num
        LOGGER.info("rmask edispatch node result of depth {}".format(dep))
        dispatch_node_mask = dispatch_guest_result.mapValues(
            lambda state_nodeid: (state_nodeid[0], random.randint(0, tree_node_num)) if len(
                state_nodeid) == 2 else state_nodeid)
        self.sync_dispatch_node_host(dispatch_node_mask, dep)
        dispatch_node_host_result = self.sync_dispatch_node_host_result(dep)

        self.node_dispatch = dispatch_guest_result.join(dispatch_node_host_result, \
                                                        lambda unleaf_state_nodeid1, unleaf_state_nodeid2: \
                                                            unleaf_state_nodeid1 if len(
                                                                unleaf_state_nodeid1) == 2 else unleaf_state_nodeid2)

    def sync_tree(self):
        LOGGER.info("sync tree to host")

        federation.remote(obj=self.tree_,
                          name=self.transfer_inst.tree.name,
                          tag=self.transfer_inst.generate_transferid(self.transfer_inst.tree),
                          role=consts.HOST,
                          idx=0)

    def convert_bin_to_real(self):
        LOGGER.info("convert tree node bins to real value")
        for i in range(len(self.tree_)):
            if self.tree_[i].is_leaf is True:
                continue
            if self.tree_[i].sitename == consts.GUEST:
                fid = self.decode("feature_idx", self.tree_[i].fid, split_maskdict=self.split_maskdict)
                bid = self.decode("feature_val", self.tree_[i].bid, self.tree_[i].id, self.split_maskdict)
                real_splitval = self.encode("feature_val", self.bin_split_points[fid][bid], self.tree_[i].id)
                self.tree_[i].bid = real_splitval

    def fit(self):
        LOGGER.info("begin to fit guest decision tree")
        self.sync_encrypted_grad_and_hess()

        root_sum_grad, root_sum_hess = self.get_grad_hess_sum(self.grad_and_hess)
        root_node = Node(id=0, sitename=consts.GUEST, sum_grad=root_sum_grad, sum_hess=root_sum_hess,
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
                                                                     self.data_bin._partitions)
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
        self.predict_weights = self.node_dispatch.mapValues(
            lambda unleaf_state_nodeid: tree_[unleaf_state_nodeid[1]].weight)

        LOGGER.info("end to fit guest decision tree")

    @staticmethod
    def traverse_tree(predict_state, data_inst, tree_=None,
                      decoder=None, split_maskdict=None):
        tag, nid = predict_state
        if tag == 0:
            return (tag, nid)

        while tree_[nid].sitename != consts.HOST:
            if tree_[nid].is_leaf is True:
                return (0, nid)

            fid = decoder("feature_idx", tree_[nid].fid, split_maskdict=split_maskdict)
            bid = decoder("feature_val", tree_[nid].bid, nid, split_maskdict)

            if data_inst.features.get_data(fid, 0) <= bid:
                nid = tree_[nid].left_nodeid
            else:
                nid = tree_[nid].right_nodeid

        return (1, nid)

    def sync_predict_finish_tag(self, finish_tag, send_times):
        LOGGER.info("send the {}-th predict finish tag {} to host".format(finish_tag, send_times))
        federation.remote(obj=finish_tag,
                          name=self.transfer_inst.predict_finish_tag.name,
                          tag=self.transfer_inst.generate_transferid(self.transfer_inst.predict_finish_tag, send_times),
                          role=consts.HOST,
                          idx=0)

    def sync_predict_data(self, predict_data, send_times):
        LOGGER.info("send predict data to host, sending times is {}".format(send_times))
        federation.remote(obj=predict_data,
                          name=self.transfer_inst.predict_data.name,
                          tag=self.transfer_inst.generate_transferid(self.transfer_inst.predict_data, send_times),
                          role=consts.HOST,
                          idx=0)

    def sync_data_predicted_by_host(self, send_times):
        LOGGER.info("get predicted data by host, recv times is {}".format(send_times))
        predict_data = federation.get(name=self.transfer_inst.predict_data_by_host.name,
                                      tag=self.transfer_inst.generate_transferid(
                                          self.transfer_inst.predict_data_by_host, send_times),
                                      idx=0)
        return predict_data

    def predict(self, data_inst):
        LOGGER.info("start to predict!")
        predict_data = data_inst.mapValues(lambda data_inst: (1, 0))
        site_host_send_times = 0
        while True:
            traverse_tree = functools.partial(self.traverse_tree,
                                              tree_=self.tree_,
                                              decoder=self.decode,
                                              split_maskdict=self.split_maskdict)
            predict_data = predict_data.join(data_inst, traverse_tree)

            unleaf_node_count = predict_data.reduce(lambda value1, value2: (value1[0] + value2[0], 0))[0]

            if unleaf_node_count == 0:
                self.sync_predict_finish_tag(True, site_host_send_times)
                break

            predict_data_mask = predict_data.mapValues(
                lambda state_nodeid:
                (state_nodeid[0], random.randint(0, len(self.tree_) - 1))
                if state_nodeid[0] == 0 else state_nodeid)

            self.sync_predict_finish_tag(False, site_host_send_times)
            self.sync_predict_data(predict_data_mask, site_host_send_times)

            predict_data_host = self.sync_data_predicted_by_host(site_host_send_times)
            predict_data = predict_data.join(predict_data_host,
                                             lambda unleaf_state1_nodeid1, unleaf_state2_nodeid2:
                                             unleaf_state1_nodeid1 if unleaf_state1_nodeid1[
                                                                          0] == 0 else unleaf_state2_nodeid2)

            site_host_send_times += 1

        predict_data = predict_data.mapValues(lambda tag_nid: self.tree_[tag_nid[1]].weight)

        LOGGER.info("predict finish!")
        return predict_data

    def get_model_meta(self):
        model_meta = DecisionTreeModelMeta()
        model_meta.criterion_meta.CopyFrom(CriterionMeta(criterion_method=self.criterion_method,
                                                         criterion_param=self.criterion_params))

        model_meta.max_depth = self.max_depth
        model_meta.min_sample_split = self.min_sample_split
        model_meta.min_impurity_split = self.min_impurity_split
        model_meta.min_leaf_node = self.min_leaf_node

        return model_meta

    def set_model_meta(self, model_meta):
        self.max_depth = model_meta.max_depth
        self.min_sample_split = model_meta.min_sample_split
        self.min_impurity_split = model_meta.min_impurity_split
        self.min_leaf_node = model_meta.min_leaf_node
        self.criterion_method = model_meta.criterion_meta.criterion_method
        self.criterion_params = list(model_meta.criterion_meta.criterion_param)

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
                                  right_nodeid=node.right_nodeid)

        model_param.split_maskdict.update(self.split_maskdict)

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
                         right_nodeid=node_param.right_nodeid)

            self.tree_.append(_node)

        self.split_maskdict = dict(model_param.split_maskdict)

    def get_model(self):
        model_meta = self.get_model_meta()
        model_param = self.get_model_param()

        return model_meta, model_param

    def load_model(self, model_meta=None, model_param=None):
        LOGGER.info("load tree model")
        self.set_model_meta(model_meta)
        self.set_model_param(model_param)
