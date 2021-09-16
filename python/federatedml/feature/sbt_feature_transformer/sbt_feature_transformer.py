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

import numpy as np
import copy
import functools
from federatedml.model_base import ModelBase
from federatedml.util import LOGGER
from federatedml.util import consts
from federatedml.ensemble import HeteroSecureBoostingTreeGuest, HeteroSecureBoostingTreeHost
from federatedml.ensemble import HeteroFastSecureBoostingTreeGuest, HeteroFastSecureBoostingTreeHost
from federatedml.model_base import MetricMeta
from federatedml.util import abnormal_detection
from federatedml.param.sbt_feature_transformer_param import SBTTransformerParam
from federatedml.feature.sparse_vector import SparseVector
from federatedml.feature.instance import Instance
from federatedml.protobuf.generated.boosting_tree_model_param_pb2 import TransformerParam
from federatedml.protobuf.generated.boosting_tree_model_meta_pb2 import TransformerMeta


class HeteroSBTFeatureTransformerBase(ModelBase):

    def __init__(self):
        super(HeteroSBTFeatureTransformerBase, self).__init__()
        self.tree_model = None
        self.role = None
        self.dense_format = True
        self.model_param = SBTTransformerParam()

    def _init_model(self, param: SBTTransformerParam):
        self.dense_format = param.dense_format

    def _abnormal_detection(self, data_instances):
        """
        Make sure input data_instances is valid.
        """
        abnormal_detection.empty_table_detection(data_instances)
        abnormal_detection.empty_feature_detection(data_instances)
        self.check_schema_content(data_instances.schema)

    @staticmethod
    def _get_model_type(model_dict):
        """
        fast-sbt or sbt ?
        """
        sbt_key_prefix = consts.HETERO_SBT_GUEST_MODEL.replace('Guest', '')
        fast_sbt_key_prefix = consts.HETERO_FAST_SBT_GUEST_MODEL.replace('Guest', '')
        for key in model_dict:
            for model_key in model_dict[key]:
                if sbt_key_prefix in model_key:
                    return consts.HETERO_SBT
                elif fast_sbt_key_prefix in model_key:
                    return consts.HETERO_FAST_SBT

        return None

    def _init_tree(self, tree_model_type):

        if tree_model_type == consts.HETERO_SBT:
            self.tree_model = HeteroSecureBoostingTreeGuest() if self.role == consts.GUEST else HeteroSecureBoostingTreeHost()
        elif tree_model_type == consts.HETERO_FAST_SBT:
            self.tree_model = HeteroFastSecureBoostingTreeGuest() if self.role == consts.GUEST \
                else HeteroFastSecureBoostingTreeHost()

    def _load_tree_model(self, model_dict, key_name='isometric_model'):
        """
        load model
        """
        # judge input model type by key in model_dict
        LOGGER.info('loading model')
        tree_model_type = self._get_model_type(model_dict[key_name])
        LOGGER.info('model type is {}'.format(tree_model_type))
        if tree_model_type is None:
            raise ValueError('key related to tree models is not detected in model dict,'
                             'please check the input model')
        self._init_tree(tree_model_type)
        # initialize tree model
        self.tree_model.load_model(model_dict, model_key=key_name)
        self.tree_model.set_flowid(self.flowid)
        self.tree_model.component_properties = copy.deepcopy(self.component_properties)

        LOGGER.info('loading {} model done'.format(tree_model_type))

    def _make_mock_isometric(self, tran_param, tran_meta):

        tree_param = tran_param.tree_param
        tree_meta = tran_meta.tree_meta
        param_name = tran_param.model_name
        meta_name = tran_meta.model_name
        mock_dict = {'isometric_model': {'sbt': {param_name: tree_param, meta_name: tree_meta}}}
        return mock_dict

    def load_model(self, model_dict):

        LOGGER.debug(f"In load_model, model_dict: {model_dict}")

        if 'isometric_model' in model_dict:
            self._load_tree_model(model_dict, key_name='isometric_model')
        elif 'model' in model_dict:
            tran_param, tran_meta = None, None
            model = model_dict['model']
            for key in model:
                content = model[key]
                for model_key in content:
                    if 'Param' in model_key:
                        tran_param = content[model_key]
                    if 'Meta' in model_key:
                        tran_meta = content[model_key]
            mock_dict = self._make_mock_isometric(tran_param, tran_meta)
            self._load_tree_model(mock_dict)
        else:
            raise ValueError('illegal model input')

    def export_model(self):

        tree_meta_name, model_meta = self.tree_model.get_model_meta()
        tree_param_name, model_param = self.tree_model.get_model_param()
        param, meta = TransformerParam(), TransformerMeta()
        param.tree_param.CopyFrom(model_param)
        param.model_name = tree_param_name
        meta.tree_meta.CopyFrom(model_meta)
        meta.model_name = tree_meta_name
        param_name, meta_name = 'SBTTransformerParam', 'SBTTransformerMeta'

        return {param_name: param, meta_name: meta}


class HeteroSBTFeatureTransformerGuest(HeteroSBTFeatureTransformerBase):

    def __init__(self):
        super(HeteroSBTFeatureTransformerGuest, self).__init__()
        self.role = consts.GUEST
        self.leaf_mapping_list = []
        self.vec_len = -1
        self.feature_title = consts.SECUREBOOST

    @staticmethod
    def join_feature_with_label(inst, leaf_indices, leaf_mapping_list, vec_len, dense):

        label = inst.label
        if dense:
            vec = np.zeros(vec_len)
            offset = 0
            for tree_idx, leaf_idx in enumerate(leaf_indices):
                vec[leaf_mapping_list[tree_idx][leaf_idx] + offset] = 1
                offset += len(leaf_mapping_list[tree_idx])
            return Instance(features=vec, label=label)

        else:
            indices, value = [], []
            offset = 0
            for tree_idx, leaf_idx in enumerate(leaf_indices):
                indices.append(leaf_mapping_list[tree_idx][leaf_idx] + offset)
                value.append(1)
                offset += len(leaf_mapping_list[tree_idx])
            return Instance(features=SparseVector(indices=indices, data=value, shape=vec_len), label=label)

    def _generate_header(self, leaf_mapping):

        header = []
        for tree_idx, mapping in enumerate(leaf_mapping):
            feat_name_prefix = self.feature_title + '_' + str(tree_idx) + '_'
            sorted_leaf_ids = sorted(list(mapping.keys()))
            for leaf_id in sorted_leaf_ids:
                header.append(feat_name_prefix+str(leaf_id))

        return header

    def _generate_callback_result(self, header):

        index = []
        for i in header:
            split_list = i.split('_')
            index.append(split_list[-1])
        return {'feat_name': header, 'index': index}

    def _transform_pred_result(self, data_inst, pred_result):

        self.leaf_mapping_list, self.vec_len = self._extract_leaf_mapping()
        join_func = functools.partial(self.join_feature_with_label, vec_len=self.vec_len, leaf_mapping_list=self.leaf_mapping_list,
                                      dense=self.dense_format)
        rs = data_inst.join(pred_result, join_func)
        # add schema for new data table
        rs.schema['header'] = self._generate_header(self.leaf_mapping_list)
        if 'label_name' in data_inst.schema:
            rs.schema['label_name'] = data_inst.schema['label_name']

        return rs

    def _extract_leaf_mapping(self):

        # one hot encoding
        leaf_mapping_list = []
        for tree_param in self.tree_model.boosting_model_list:
            leaf_mapping = {}
            idx = 0
            for node_param in tree_param.tree_:
                if node_param.is_leaf:
                    leaf_mapping[node_param.id] = idx
                    idx += 1
            leaf_mapping_list.append(leaf_mapping)

        vec_len = 0
        for map_ in leaf_mapping_list:
            vec_len += len(map_)

        return leaf_mapping_list, vec_len

    def _callback_leaf_id_mapping(self, mapping):

        metric_namespace = 'sbt_transformer'
        metric_name = 'leaf_mapping'
        self.tracker.set_metric_meta(metric_namespace, metric_name,
                                     MetricMeta(name=metric_name, metric_type=metric_name, extra_metas=mapping))

    def fit(self, data_inst):

        self._abnormal_detection(data_inst)
        # predict instances to get leaf indexes
        LOGGER.info('tree model running prediction')
        predict_rs = self.tree_model.predict(data_inst, ret_format='leaf')
        LOGGER.info('tree model prediction done')

        # transform pred result to new data table
        LOGGER.debug('use dense is {}'.format(self.dense_format))
        rs = self._transform_pred_result(data_inst, predict_rs)

        # display result callback
        LOGGER.debug('header is {}'.format(rs.schema))
        LOGGER.debug('extra meta is {}'.format(self._generate_callback_result(rs.schema['header'])))
        self._callback_leaf_id_mapping(self._generate_callback_result(rs.schema['header']))

        return rs

    def transform(self, data_inst):
        return self.fit(data_inst)


class HeteroSBTFeatureTransformerHost(HeteroSBTFeatureTransformerBase):

    def __init__(self):
        super(HeteroSBTFeatureTransformerHost, self).__init__()
        self.role = consts.HOST

    def fit(self, data_inst):

        self._abnormal_detection(data_inst)
        self.tree_model.predict(data_inst)

    def transform(self, data_inst):
        self.fit(data_inst)

