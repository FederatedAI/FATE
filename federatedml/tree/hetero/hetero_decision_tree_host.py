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
# HeteroDecisionTreeHost
# =============================================================================

from arch.api.utils import log_utils
from federatedml.protobuf.generated.boosting_tree_model_meta_pb2 import DecisionTreeModelMeta
from federatedml.protobuf.generated.boosting_tree_model_param_pb2 import DecisionTreeModelParam
from federatedml.tree import DecisionTree
from federatedml.tree import Splitter
from federatedml.tree import SplitInfo
from federatedml.tree import FeatureHistogram
from federatedml.transfer_variable.transfer_class.hetero_decision_tree_transfer_variable import \
    HeteroDecisionTreeTransferVariable
from federatedml.util import consts
from federatedml.tree import Node
from federatedml.feature.fate_element_type import NoneType
from arch.api.utils.splitable import segment_transfer_enabled
import functools

LOGGER = log_utils.getLogger()


class HeteroDecisionTreeHost(DecisionTree):
    def __init__(self, tree_param):
        LOGGER.info("hetero decision tree guest init!")
        super(HeteroDecisionTreeHost, self).__init__(tree_param)

        self.splitter = Splitter(self.criterion_method, self.criterion_params, self.min_impurity_split,
                                 self.min_sample_split, self.min_leaf_node)

        self.data_bin = None
        self.data_bin_with_position = None
        self.grad_and_hess = None
        self.bin_split_points = None
        self.bin_sparse_points = None
        self.infos = None
        self.valid_features = None
        self.pubkey = None
        self.privakey = None
        self.tree_id = None
        self.encrypted_grad_and_hess = None
        self.transfer_inst = HeteroDecisionTreeTransferVariable()
        self.tree_node_queue = None
        self.cur_split_nodes = None
        self.split_maskdict = {}
        self.missing_dir_maskdict = {}
        self.tree_ = None
        self.runtime_idx = 0
        self.sitename = consts.HOST

    def set_flowid(self, flowid=0):
        LOGGER.info("set flowid, flowid is {}".format(flowid))
        self.transfer_inst.set_flowid(flowid)

    # def set_runtime_idx(self, runtime_idx):
    #     self.runtime_idx = runtime_idx
    #     self.sitename = ":".join([consts.HOST, str(self.runtime_idx)])

    def set_inputinfo(self, data_bin=None, grad_and_hess=None, bin_split_points=None, bin_sparse_points=None):
        LOGGER.info("set input info")
        self.data_bin = data_bin
        self.grad_and_hess = grad_and_hess
        self.bin_split_points = bin_split_points
        self.bin_sparse_points = bin_sparse_points

    def set_valid_features(self, valid_features=None):
        LOGGER.info("set valid features")
        self.valid_features = valid_features

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

    def sync_encrypted_grad_and_hess(self):
        LOGGER.info("get encrypted grad and hess")
        self.grad_and_hess = self.transfer_inst.encrypted_grad_and_hess.get(idx=0)
        """
        self.grad_and_hess = federation.get(name=self.transfer_inst.encrypted_grad_and_hess.name,
                                            tag=self.transfer_inst.generate_transferid(
                                                self.transfer_inst.encrypted_grad_and_hess),
                                            idx=0)
        """

    def sync_node_positions(self, dep=-1):
        LOGGER.info("get tree node queue of depth {}".format(dep))
        node_positions = self.transfer_inst.node_positions.get(idx=0,
                                                               suffix=(dep,))
        """
        node_positions = federation.get(name=self.transfer_inst.node_positions.name,
                                        tag=self.transfer_inst.generate_transferid(self.transfer_inst.node_positions,
                                                                                   dep),
                                        idx=0)
        """
        return node_positions

    def sync_tree_node_queue(self, dep=-1):
        LOGGER.info("get tree node queue of depth {}".format(dep))
        self.tree_node_queue = self.transfer_inst.tree_node_queue.get(idx=0,
                                                                      suffix=(dep,))
        """
        self.tree_node_queue = federation.get(name=self.transfer_inst.tree_node_queue.name,
                                              tag=self.transfer_inst.generate_transferid(
                                                  self.transfer_inst.tree_node_queue, dep),
                                              idx=0)
        """

    def get_histograms(self, node_map={}):
        LOGGER.info("start to get node histograms")
        acc_histograms = FeatureHistogram.calculate_histogram(
            self.data_bin_with_position, self.grad_and_hess,
            self.bin_split_points, self.bin_sparse_points,
            self.valid_features, node_map,
            self.use_missing, self.zero_as_missing, ret="tb")

        return acc_histograms

    def sync_encrypted_splitinfo_host(self, encrypted_splitinfo_host, dep=-1, batch=-1):
        LOGGER.info("send encrypted splitinfo of depth {}, batch {}".format(dep, batch))

        self.transfer_inst.encrypted_splitinfo_host.remote(encrypted_splitinfo_host,
                                                           role=consts.GUEST,
                                                           idx=-1,
                                                           suffix=(dep, batch,))
        """
        self.transfer_inst.encrypted_splitinfo_host.remote(encrypted_splitinfo_host,
                                                           role=consts.GUEST,
                                                           idx=-1,
                                                           suffix=(dep, batch,))
        """

    def sync_federated_best_splitinfo_host(self, dep=-1, batch=-1):
        LOGGER.info("get federated best splitinfo of depth {}, batch {}".format(dep, batch))
        federated_best_splitinfo_host = self.transfer_inst.federated_best_splitinfo_host.get(idx=0,
                                                                                             suffix=(dep, batch,))
        """
        federated_best_splitinfo_host = federation.get(name=self.transfer_inst.federated_best_splitinfo_host.name,
                                                       tag=self.transfer_inst.generate_transferid(
                                                           self.transfer_inst.federated_best_splitinfo_host, dep,
                                                           batch),
                                                       idx=0)
        """

        return federated_best_splitinfo_host

    def sync_final_splitinfo_host(self, splitinfo_host, federated_best_splitinfo_host, dep=-1, batch=-1):
        LOGGER.info("send host final splitinfo of depth {}, batch {}".format(dep, batch))
        final_splitinfos = []
        for i in range(len(splitinfo_host)):
            best_idx, best_gain = federated_best_splitinfo_host[i]
            if best_idx != -1:
                assert splitinfo_host[i][best_idx].sitename == self.sitename
                splitinfo = splitinfo_host[i][best_idx]
                splitinfo.best_fid = self.encode("feature_idx", splitinfo.best_fid)
                assert splitinfo.best_fid is not None
                splitinfo.best_bid = self.encode("feature_val", splitinfo.best_bid, self.cur_split_nodes[i].id)
                splitinfo.missing_dir = self.encode("missing_dir", splitinfo.missing_dir, self.cur_split_nodes[i].id)
                splitinfo.gain = best_gain
            else:
                splitinfo = SplitInfo(sitename=self.sitename, best_fid=-1, best_bid=-1, gain=best_gain)

            final_splitinfos.append(splitinfo)

        self.transfer_inst.final_splitinfo_host.remote(final_splitinfos,
                                                       role=consts.GUEST,
                                                       idx=-1,
                                                       suffix=(dep, batch,))

        """
        federation.remote(obj=final_splitinfos,
                          name=self.transfer_inst.final_splitinfo_host.name,
                          tag=self.transfer_inst.generate_transferid(self.transfer_inst.final_splitinfo_host, dep,
                                                                     batch),
                          role=consts.GUEST,
                          idx=-1)
        """

    def sync_dispatch_node_host(self, dep):
        LOGGER.info("get node from host to dispath, depth is {}".format(dep))
        dispatch_node_host = self.transfer_inst.dispatch_node_host.get(idx=0,
                                                                       suffix=(dep,))
        """
        dispatch_node_host = federation.get(name=self.transfer_inst.dispatch_node_host.name,
                                            tag=self.transfer_inst.generate_transferid(
                                                self.transfer_inst.dispatch_node_host, dep),
                                            idx=0)
        """
        return dispatch_node_host

    @staticmethod
    def dispatch_node(value1, value2, sitename=None, decoder=None,
                      split_maskdict=None, bin_sparse_points=None,
                      use_missing=False, zero_as_missing=False,
                      missing_dir_maskdict=None):

        unleaf_state, fid, bid, node_sitename, nodeid, left_nodeid, right_nodeid = value1
        if node_sitename != sitename:
            return value1

        fid = decoder("feature_idx", fid, split_maskdict=split_maskdict)
        bid = decoder("feature_val", bid, nodeid, split_maskdict=split_maskdict)
        if not use_missing:
            if value2.features.get_data(fid, bin_sparse_points[fid]) <= bid:
                return unleaf_state, left_nodeid
            else:
                return unleaf_state, right_nodeid
        else:
            missing_dir = decoder("missing_dir", 1, nodeid,
                                  missing_dir_maskdict=missing_dir_maskdict)
            missing_val = False
            if zero_as_missing:
                if value2.features.get_data(fid, None) is None or \
                        value2.features.get_data(fid) == NoneType():
                    missing_val = True
            elif use_missing and value2.features.get_data(fid) == NoneType():
                missing_val = True

            if missing_val:
                if missing_dir == 1:
                    return unleaf_state, right_nodeid
                else:
                    return unleaf_state, left_nodeid
            else:
                if value2.features.get_data(fid, bin_sparse_points[fid]) <= bid:
                    return unleaf_state, left_nodeid
                else:
                    return unleaf_state, right_nodeid

    def sync_dispatch_node_host_result(self, dispatch_node_host_result, dep=-1):
        LOGGER.info("send host dispatch result, depth is {}".format(dep))

        self.transfer_inst.dispatch_node_host_result.remote(dispatch_node_host_result,
                                                            role=consts.GUEST,
                                                            idx=-1,
                                                            suffix=(dep,))

        """
        federation.remote(obj=dispatch_node_host_result,
                          name=self.transfer_inst.dispatch_node_host_result.name,
                          tag=self.transfer_inst.generate_transferid(self.transfer_inst.dispatch_node_host_result, dep),
                          role=consts.GUEST,
                          idx=-1)
        """

    def find_dispatch(self, dispatch_node_host, dep=-1):
        LOGGER.info("start to find host dispath of depth {}".format(dep))
        dispatch_node_method = functools.partial(self.dispatch_node,
                                                 sitename=self.sitename,
                                                 decoder=self.decode,
                                                 split_maskdict=self.split_maskdict,
                                                 bin_sparse_points=self.bin_sparse_points,
                                                 use_missing=self.use_missing,
                                                 zero_as_missing=self.zero_as_missing,
                                                 missing_dir_maskdict=self.missing_dir_maskdict)
        dispatch_node_host_result = dispatch_node_host.join(self.data_bin, dispatch_node_method)
        self.sync_dispatch_node_host_result(dispatch_node_host_result, dep)

    def sync_tree(self):
        LOGGER.info("sync tree from guest")
        self.tree_ = self.transfer_inst.tree.get(idx=0)
        """
        self.tree_ = federation.get(name=self.transfer_inst.tree.name,
                                    tag=self.transfer_inst.generate_transferid(self.transfer_inst.tree),
                                    idx=0)
        """

    def remove_duplicated_split_nodes(self, split_nid_used):
        LOGGER.info("remove duplicated nodes from split mask dict")
        duplicated_nodes = set(self.split_maskdict.keys()) - set(split_nid_used)
        for nid in duplicated_nodes:
            del self.split_maskdict[nid]

    def convert_bin_to_real(self):
        LOGGER.info("convert tree node bins to real value")
        split_nid_used = []
        for i in range(len(self.tree_)):
            if self.tree_[i].is_leaf is True:
                continue

            if self.tree_[i].sitename == self.sitename:
                fid = self.decode("feature_idx", self.tree_[i].fid, split_maskdict=self.split_maskdict)
                bid = self.decode("feature_val", self.tree_[i].bid, self.tree_[i].id, self.split_maskdict)
                LOGGER.debug ("shape of bin_split_points is {}".format(len(self.bin_split_points[fid])))
                real_splitval = self.encode("feature_val", self.bin_split_points[fid][bid], self.tree_[i].id)
                self.tree_[i].bid = real_splitval

                split_nid_used.append(self.tree_[i].id)

        self.remove_duplicated_split_nodes(split_nid_used)

    @staticmethod
    def traverse_tree(predict_state, data_inst, tree_=None,
                      decoder=None, split_maskdict=None, sitename=consts.HOST,
                      use_missing=False, zero_as_missing=False,
                      missing_dir_maskdict=None):

        nid, _ = predict_state
        if tree_[nid].sitename != sitename:
            return predict_state

        while tree_[nid].sitename == sitename:
            fid = decoder("feature_idx", tree_[nid].fid, split_maskdict=split_maskdict)
            bid = decoder("feature_val", tree_[nid].bid, nid, split_maskdict)

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

        return nid, 0

    def sync_predict_finish_tag(self, recv_times):
        LOGGER.info("get the {}-th predict finish tag from guest".format(recv_times))
        finish_tag = self.transfer_inst.predict_finish_tag.get(idx=0,
                                                               suffix=(recv_times,))
        """
        finish_tag = federation.get(name=self.transfer_inst.predict_finish_tag.name,
                                    tag=self.transfer_inst.generate_transferid(self.transfer_inst.predict_finish_tag,
                                                                               recv_times),
                                    idx=0)
        """

        return finish_tag

    def sync_predict_data(self, recv_times):
        LOGGER.info("srecv predict data to host, recv times is {}".format(recv_times))
        predict_data = self.transfer_inst.predict_data.get(idx=0,
                                                           suffix=(recv_times,))
        """
        predict_data = federation.get(name=self.transfer_inst.predict_data.name,
                                      tag=self.transfer_inst.generate_transferid(self.transfer_inst.predict_data,
                                                                                 recv_times),
                                      idx=0)
        """

        return predict_data

    def sync_data_predicted_by_host(self, predict_data, send_times):
        LOGGER.info("send predicted data by host, send times is {}".format(send_times))

        self.transfer_inst.predict_data_by_host.remote(predict_data,
                                                       role=consts.GUEST,
                                                       idx=0,
                                                       suffix=(send_times,))
        """
        federation.remote(obj=predict_data,
                          name=self.transfer_inst.predict_data_by_host.name,
                          tag=self.transfer_inst.generate_transferid(self.transfer_inst.predict_data_by_host,
                                                                     send_times),
                          role=consts.GUEST,
                          idx=0)
        """

    def fit(self):
        LOGGER.info("begin to fit host decision tree")
        self.sync_encrypted_grad_and_hess()

        for dep in range(self.max_depth):
            self.sync_tree_node_queue(dep)
            if len(self.tree_node_queue) == 0:
                break

            node_positions = self.sync_node_positions(dep)
            self.data_bin_with_position = self.data_bin.join(node_positions, lambda v1, v2: (v1, v2))

            batch = 0
            for i in range(0, len(self.tree_node_queue), self.max_split_nodes):
                self.cur_split_nodes = self.tree_node_queue[i: i + self.max_split_nodes]
                node_map = {}
                node_num = 0
                for tree_node in self.cur_split_nodes:
                    node_map[tree_node.id] = node_num
                    node_num += 1

                acc_histograms = self.get_histograms(node_map=node_map)

                splitinfo_host, encrypted_splitinfo_host = self.splitter.find_split_host(acc_histograms,
                                                                                         self.valid_features,
                                                                                         node_map,
                                                                                         self.sitename,
                                                                                         self.use_missing,
                                                                                         self.zero_as_missing)

                self.sync_encrypted_splitinfo_host(encrypted_splitinfo_host, dep, batch)
                federated_best_splitinfo_host = self.sync_federated_best_splitinfo_host(dep, batch)
                self.sync_final_splitinfo_host(splitinfo_host, federated_best_splitinfo_host, dep, batch)

                batch += 1

            dispatch_node_host = self.sync_dispatch_node_host(dep)
            self.find_dispatch(dispatch_node_host, dep)

        self.sync_tree()
        self.convert_bin_to_real()

        LOGGER.info("end to fit guest decision tree")

    def predict(self, data_inst):
        LOGGER.info("start to predict!")
        site_guest_send_times = 0
        while True:
            finish_tag = self.sync_predict_finish_tag(site_guest_send_times)
            if finish_tag is True:
                break

            predict_data = self.sync_predict_data(site_guest_send_times)

            traverse_tree = functools.partial(self.traverse_tree,
                                              tree_=self.tree_,
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
