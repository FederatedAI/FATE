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
from fate.ml.ensemble.learner.decision_tree.tree_core.decision_tree import DecisionTree, Node, _get_sample_on_local_nodes, _update_sample_pos
from fate.ml.ensemble.learner.decision_tree.tree_core.hist import SBTHistogramBuilder
from fate.ml.ensemble.learner.decision_tree.tree_core.splitter import FedSBTSplitter
from fate.ml.ensemble.learner.decision_tree.tree_core.loss import get_task_info
from fate.ml.utils.predict_tools import BINARY, MULTI, REGRESSION
from fate.arch import Context
from fate.arch.dataframe import DataFrame
from typing import List
import functools
import logging 
import pandas as pd
import torch as t
import numpy as np


logger = logging.getLogger(__name__)

<<<<<<< HEAD
FIX_POINT_PRECISION = 52
=======
FIX_POINT_PRECISION = 2**52

>>>>>>> dev-2.0.0-beta

class HeteroDecisionTreeGuest(DecisionTree):

    def __init__(self, max_depth=3, valid_features=None, use_missing=False, zero_as_missing=False, goss=False, l1=0.1, l2=0, 
<<<<<<< HEAD
                 min_impurity_split=1e-2, min_sample_split=2, min_leaf_node=1, min_child_weight=1, gh_pack=True, objective=None):
=======
                 min_impurity_split=1e-2, min_sample_split=2, min_leaf_node=1, min_child_weight=1, gh_pack=False, objective=None):
>>>>>>> dev-2.0.0-beta

        super().__init__(max_depth, use_missing=use_missing, zero_as_missing=zero_as_missing, valid_features=valid_features)
        self.host_sitenames = None
        self._tree_node_num = 0
        self.hist_builder = None
        self.splitter = None

        # regularization
        self.l1 = l1
        self.l2 = l2
        self.min_impurity_split = min_impurity_split
        self.min_sample_split = min_sample_split
        self.min_leaf_node = min_leaf_node
        self.min_child_weight = min_child_weight

        # goss
        self.goss = goss

        # other
        self._valid_features = valid_features

        # homographic encryption
        self._encrypt_kit = None
        self._sk = None
        self._pk = None
        self._coder = None
        self._evaluator = None
        self._encryptor = None
        self._decryptor = None

        # for g, h packing
        self._gh_pack = gh_pack
        self._g_offset = 0
        self._g_abs_max = 0
        self._h_abs_max = 0
        self._objective = objective
        if gh_pack:
            if objective is None:
                raise ValueError('objective must be specified when gh_pack is True')
<<<<<<< HEAD
        self._pack_info = {}
=======
>>>>>>> dev-2.0.0-beta


    def set_encrypt_kit(self, kit):
        self._encrypt_kit = kit
        self._sk, self._pk, self._coder, self._evaluator, self._encryptor = kit.sk, kit.pk, kit.coder, kit.evaluator, kit.get_tensor_encryptor()
        self._decryptor = kit.get_tensor_decryptor()
        logger.info('encrypt kit setup through setter')

    def _init_encrypt_kit(self, ctx):
        kit = ctx.cipher.phe.setup(options={"kind": "paillier", "key_length": 1024})
        self._sk, self._pk, self._coder, self._evaluator, self._encryptor = kit.sk, kit.pk, kit.coder, kit.evaluator, kit.get_tensor_encryptor()
        self._decryptor = kit.get_tensor_decryptor()
        logger.info('encrypt kit is not setup, auto initializing')

    def _get_column_max_bin(self, result_dict):
        bin_len = {}
        
        for column, values in result_dict.items():
            bin_num = len(values)
            bin_len[column] = bin_num 
        
        max_max_value = max(bin_len.values())
        
        return bin_len, max_max_value
    
    def _update_sample_pos(self, ctx, cur_layer_nodes: List[Node], sample_pos: DataFrame, data: DataFrame, node_map: dict):

        sitename = ctx.local.party[0] + '_' + ctx.local.party[1]
        data_with_pos = DataFrame.hstack([data, sample_pos])
        map_func = functools.partial(_get_sample_on_local_nodes, cur_layer_node=cur_layer_nodes, node_map=node_map, sitename=sitename)
        local_sample_idx = data_with_pos.apply_row(map_func)
        local_samples = data_with_pos.loc(local_sample_idx.get_indexer(target="sample_id"), preserve_order=True)[local_sample_idx.values.as_tensor()]
        logger.info('{}/{} samples on local nodes'.format(len(local_samples), len(data)))
        if len(local_samples) == 0:
            updated_sample_pos = None
        else:
            updated_sample_pos = sample_pos.loc(local_samples.get_indexer(target="sample_id"), preserve_order=True).create_frame()
            update_func = functools.partial(_update_sample_pos, cur_layer_node=cur_layer_nodes, node_map=node_map)
            map_rs = local_samples.apply_row(update_func)
            updated_sample_pos["node_idx"] = map_rs # local_samples.apply_row(update_func)

        # synchronize sample pos
        host_update_sample_pos = ctx.hosts.get('updated_data')
        new_sample_pos = sample_pos.empty_frame()

        for host_data in host_update_sample_pos:
            if host_data[0]:  # True
                pos_data, pos_index = host_data[1]
                tmp_frame = sample_pos.create_frame()
                tmp_frame = tmp_frame.loc(pos_index, preserve_order=True)
                tmp_frame['node_idx'] = pos_data
                new_sample_pos = DataFrame.vstack([new_sample_pos, tmp_frame])

        if updated_sample_pos is not None:
            if len(updated_sample_pos) == len(data):  # all samples are on local
                new_sample_pos = updated_sample_pos
            else:
                logger.info('stack new sample pos, guest len {}, host len {}'.format(len(updated_sample_pos), len(new_sample_pos)))
                new_sample_pos = DataFrame.vstack([updated_sample_pos, new_sample_pos])
        else:
            new_sample_pos = new_sample_pos  # all samples are on host

       # share new sample position with all hosts
        ctx.hosts.put('new_sample_pos', (new_sample_pos.as_tensor(), new_sample_pos.get_indexer(target='sample_id')))
        self.sample_pos = new_sample_pos

        return new_sample_pos
    
    def _g_h_process(self, grad_and_hess: DataFrame):
        
        en_grad_hess = grad_and_hess.create_frame()

<<<<<<< HEAD
        def make_long_tensor(s: pd.Series, coder, pk, offset, shift_bit, precision, encryptor, pack_num=2):
            pack_tensor = t.Tensor(s.values)
            pack_tensor[0] = pack_tensor[0] + offset
            pack_vec = coder.pack_floats(pack_tensor, shift_bit, pack_num, precision)
            en = pk.encrypt_encoded(pack_vec, obfuscate=True)
            ret = encryptor.lift(en, (len(en), 1), pack_tensor.dtype, pack_tensor.device)
            return ret

        def compute_offset_bit(sample_num, g_max, h_max):
            g_bit = int(np.log2(2**FIX_POINT_PRECISION * sample_num * g_max) + 1) # add 1 more bit for safety
            h_bit = int(np.log2(2**FIX_POINT_PRECISION * sample_num * h_max) + 1) 
=======
        def make_long_tensor(s: pd.Series, coder, pk, encryptor, offset=0, pack_num=2, shift_bit=52):
            gh = t.LongTensor([int((s['g']+offset)*FIX_POINT_PRECISION), int(s['h']*FIX_POINT_PRECISION)])
            pack_vec = coder.pack_vec(gh, num_shift_bit=shift_bit, num_elem_each_pack=pack_num)
            en = pk.encrypt_encoded(pack_vec, obfuscate=True)
            return encryptor.lift(en, (len(en), 1), t.long, gh.device)

        def compute_offset_bit(sample_num, g_max, h_max):
            g_bit = int(np.log2(FIX_POINT_PRECISION * sample_num * g_max) + 1) # add 1 more bit for safety
            h_bit = int(np.log2(FIX_POINT_PRECISION * sample_num * h_max) + 1) 
>>>>>>> dev-2.0.0-beta
            return max(g_bit, h_bit)

        if self._gh_pack:
            
            task_type = get_task_info(self._objective)

            if task_type == BINARY or task_type == MULTI:
                self._g_offset = 1
                self._g_abs_max = 2
                self._h_abs_max = 1
                
            elif task_type == REGRESSION:
                self._g_offset = abs(float(grad_and_hess['g'].min()['g']))
                self._g_abs_max = abs(float(grad_and_hess['g'].max()['g'])) + self._g_offset
                self._h_abs_max = 2

<<<<<<< HEAD
            pack_num, total_num = 2, 2
            shift_bit = compute_offset_bit(len(grad_and_hess), self._g_abs_max, self._h_abs_max)
            partial_func = functools.partial(make_long_tensor, coder=self._coder, offset=self._g_offset, pk=self._pk,
                                             shift_bit=shift_bit, pack_num=2, precision=FIX_POINT_PRECISION, encryptor=self._encryptor)
            en_grad_hess['gh'] = grad_and_hess.apply_row(partial_func)

            # record pack info
            self._pack_info['shift_bit'] = shift_bit
            self._pack_info['precision'] = FIX_POINT_PRECISION
            self._pack_info['pack_num'] = pack_num
            self._pack_info['total_num'] = total_num
=======
            shift_bit = compute_offset_bit(len(grad_and_hess), self._g_abs_max, self._h_abs_max)
            
            partial_func = functools.partial(make_long_tensor, coder=self._coder, offset=self._g_offset, pk=self._pk,
                                             shift_bit=shift_bit, pack_num=2, encryptor=self._encryptor)
            
            en_grad_hess['gh'] = grad_and_hess.apply_row(partial_func)
>>>>>>> dev-2.0.0-beta
        else:
            en_grad_hess['g'] = self._encryptor.encrypt_tensor(grad_and_hess['g'].as_tensor())
            en_grad_hess['h'] = self._encryptor.encrypt_tensor(grad_and_hess['h'].as_tensor())

        return en_grad_hess

    def _send_gh(self, ctx: Context, grad_and_hess: DataFrame):
        
        # encrypt g & h
        en_grad_hess = self._g_h_process(grad_and_hess)
        ctx.hosts.put('en_gh', en_grad_hess)
        ctx.hosts.put('en_kit', [self._pk, self._evaluator])

    def _mask_node(self, ctx: Context, nodes: List[Node]):
        new_nodes = []
        for n in nodes:
            new_nodes.append(Node(nid=n.nid, is_leaf=n.is_leaf,  l=n.l, r=n.r, is_left_node=n.is_left_node, split_id=n.split_id, sitename=n.sitename, sample_num=n.sample_num))
        return new_nodes

    def _check_assign_result(self, sample_pos: DataFrame, cur_layer_node: List[Node]):
        # debugging function
        sample_pos_df = sample_pos.as_pd_df()
        sample_pos_count = sample_pos_df.groupby('node_idx').count().to_dict()['sample_id']
        for node in cur_layer_node:
            nid = node.nid
            sample_count_0 = node.sample_num
            sample_count_1 = sample_pos_count[nid]
            if sample_count_0 != sample_count_1:
                parent_nid = node.parent_nodeid
                for i in self._nodes:
                    if i.nid == parent_nid:
                        logger.info('parent node {}'.format(i))
                raise ValueError('node {} sample count not match, {} vs {}, node details {}'.format(nid, sample_count_0, sample_count_1, node))

    def _sync_nodes(self, ctx: Context, cur_layer_nodes: List[Node], next_layer_nodes: List[Node]):
        
        mask_cur_layer = self._mask_node(ctx, cur_layer_nodes)
        mask_next_layer = self._mask_node(ctx, next_layer_nodes)
        ctx.hosts.put('sync_nodes', [mask_cur_layer, mask_next_layer])

    def booster_fit(self, ctx: Context, bin_train_data: DataFrame, grad_and_hess: DataFrame, binning_dict: dict):
        
        # Initialization
        train_df = bin_train_data
        sample_pos = self._init_sample_pos(train_df)
        self._sample_on_leaves = sample_pos.empty_frame()
        root_node = self._initialize_root_node(ctx, train_df, grad_and_hess)

        # initialize homographic encryption
        if self._encrypt_kit is None:
            self._init_encrypt_kit(ctx)
        # Send Encrypted Grad and Hess
        self._send_gh(ctx, grad_and_hess)

        # init histogram builder
        self.hist_builder = SBTHistogramBuilder(bin_train_data, binning_dict, None)

        # init splitter
        self.splitter = FedSBTSplitter(bin_train_data, binning_dict, l2=self.l2, l1=self.l1, 
                                       min_sample_split=self.min_sample_split, min_impurity_split=self.min_impurity_split,
                                       min_child_weight=self.min_child_weight, min_leaf_node=self.min_leaf_node)

        # Prepare for training
        node_map = {}
        cur_layer_node = [root_node]

        for cur_depth, sub_ctx in ctx.on_iterations.ctxs_range(self.max_depth):
            
            if len(cur_layer_node) == 0:
                logger.info('no nodes to split, stop training')
                break
            
            assert len(sample_pos) == len(train_df), 'sample pos len not match train data len, {} vs {}'.format(len(sample_pos), len(train_df))

            # debug checking code
            # self._check_assign_result(sample_pos, cur_layer_node)
            # initialize node map
            node_map = {n.nid: idx for idx, n in enumerate(cur_layer_node)}
            # compute histogram
            hist_inst, statistic_result = self.hist_builder.compute_hist(sub_ctx, cur_layer_node, train_df, grad_and_hess, sample_pos, node_map)
            # compute best splits
<<<<<<< HEAD
            split_info = self.splitter.split(sub_ctx, statistic_result, cur_layer_node, node_map, self._sk, self._coder, self._gh_pack, self._pack_info)
=======
            split_info = self.splitter.split(sub_ctx, statistic_result, cur_layer_node, node_map, self._sk, self._coder, self._gh_pack)
>>>>>>> dev-2.0.0-beta
            # update tree with best splits
            next_layer_nodes = self._update_tree(sub_ctx, cur_layer_node, split_info, train_df)
            # update feature importance
            self._update_feature_importance(sub_ctx, split_info, train_df)
            # sync nodes
            self._sync_nodes(sub_ctx, cur_layer_node, next_layer_nodes)
            # update sample positions
            sample_pos = self._update_sample_pos(sub_ctx, cur_layer_node, sample_pos, train_df, node_map)
            # if sample reaches leaf nodes, drop them
            sample_on_leaves = self._get_samples_on_leaves(sample_pos)
            train_df, sample_pos = self._drop_samples_on_leaves(sample_pos, train_df)
            self._sample_on_leaves = DataFrame.vstack([self._sample_on_leaves, sample_on_leaves])
            # next layer nodes
            cur_layer_node = next_layer_nodes
            logger.info('layer {} done: next layer will split {} nodes, active samples num {}'.format(cur_depth, len(cur_layer_node), len(sample_pos)))
            self.next_layer_node = next_layer_nodes

        # handle final leaves
        if len(cur_layer_node) != 0:
            for node in cur_layer_node:
                node.is_leaf = True
                node.sitename = ctx.guest.party[0] + '_' + ctx.guest.party[1] # leaf always on guest
                self._nodes.append(node)
            self._sample_on_leaves = DataFrame.vstack([self._sample_on_leaves, sample_pos])

        # when training is done, all samples must be on leaves
        assert len(self._sample_on_leaves) == len(bin_train_data), 'sample on leaves num not match, {} vs {}'.format(len(self._sample_on_leaves), len(bin_train_data))
        # convert sample pos to weights
        self._sample_weights = self._convert_sample_pos_to_weight(self._sample_on_leaves, self._nodes)
        # convert bid to split value
        self._nodes = self._convert_bin_idx_to_split_val(ctx, self._nodes, binning_dict, bin_train_data.schema)

    def get_hyper_param(self):
        param = {
            'max_depth': self.max_depth,
            'valid_features': self._valid_features,
            'l1': self.l1,
            'l2': self.l2,
            'use_missing': self.use_missing,
            'zero_as_missing': self.zero_as_missing
        }
        return param
    
    @staticmethod
    def from_model(model_dict):
        return HeteroDecisionTreeGuest._from_model(model_dict, HeteroDecisionTreeGuest)
    