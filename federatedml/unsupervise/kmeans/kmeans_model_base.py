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

from arch.api.utils import log_utils
from federatedml.model_base import ModelBase
from federatedml.param.hetero_kmeans_param import KmeansParam
from federatedml.protobuf.generated import hetero_kmeans_meta_pb2, hetero_kmeans_param_pb2
from federatedml.transfer_variable.transfer_class.hetero_kmeans_transfer_variable import HeteroKmeansTransferVariable
from federatedml.util import abnormal_detection
from federatedml.util import consts

LOGGER = log_utils.getLogger()


class BaseKmeansModel(ModelBase):
    def __init__(self):
        super(BaseKmeansModel, self).__init__()
        self.model_param = KmeansParam()
        self.n_iter_ = 0
        self.k = 0
        self.max_iter = 0
        self.tol = 0
        self.iter = iter
        self.centroid_list = None
        self.cluster_result = None
        self.transfer_variable = HeteroKmeansTransferVariable()
        self.model_name = 'toSet'
        self.model_param_name = 'toSet'
        self.model_meta_name = 'toSet'
        self.header = None
        self.reset_union()
        self.is_converged = False
        self.cluster_detail = None
        self.cluster_count = None

    def _init_model(self, params):
        self.model_param = params
        self.k = params.k
        self.max_iter = params.max_iter
        self.tol = params.tol

    def _get_meta(self):
        meta_protobuf_obj = hetero_kmeans_meta_pb2.KmeansModelMeta(k=self.model_param.k,
                                                                   tol=self.model_param.tol,
                                                                   max_iter=self.max_iter)
        return meta_protobuf_obj

    def _get_param(self):
        header = self.header
        LOGGER.debug("In get_param, header: {}".format(header))
        if header is None:
            param_protobuf_obj = hetero_kmeans_param_pb2.KmeansModelParam()
            return param_protobuf_obj
        param_protobuf_obj = hetero_kmeans_param_pb2.KmeansModelParam(count_of_clusters=self.k,
                                                                      max_interation=self.n_iter_,
                                                                      converged=self.is_converged,
                                                                      cluster_detail=self.cluster_detail,
                                                                      centroid_detail=self.centroid_list)
        return param_protobuf_obj

    def export_model(self):
        meta_obj = self._get_meta()
        param_obj = self._get_param()
        result = {
            self.model_meta_name: meta_obj,
            self.model_param_name: param_obj
        }
        return result

    def count(self, iterator):
        count_result = dict()
        for k, v in iterator:
            if v not in count_result:
                count_result[v] = 1
            else:
                count_result[v] += 1
        return count_result

    @staticmethod
    def sum_dict(d1, d2):
        temp = dict()
        for key in d1.keys() | d2.keys():
            temp[key] = sum([d.get(key, 0) for d in (d1, d2)])
        return temp

    def _abnormal_detection(self, data_instances):
        """
        Make sure input data_instances is valid.
        """
        abnormal_detection.empty_table_detection(data_instances)
        abnormal_detection.empty_feature_detection(data_instances)

    def load_model(self, model_dict):
        param_obj = list(model_dict.get('model').values())[0].get(self.model_param_name)
        meta_obj = list(model_dict.get('model').values())[0].get(self.model_meta_name)
        self.k = meta_obj.k
        self.centroid_list = param_obj.centroid_detail
        # self.header = list(result_obj.header)
        # if self.header is None:
        #    return

    def reset_union(self):
        def my_union(previews_data, name_list):
            return previews_data

        self.component_properties.set_union_func(my_union)

    def set_predict_data_schema(self, predict_datas, schemas):
        if predict_datas is None:
            return None, None
            return predict_datas

        predict_data = predict_datas[0][0]
        schema = schemas[0]
        if self.role == consts.ARBITER:
            data_output1 = predict_data[0]
            data_output2 = predict_data[1]
            if data_output1 is not None:
                data_output1.schema = {
                    "header": ["cluster_sample_count", "cluster_inner_dist", "inter_cluster_dist"],
                    "sid_name": "cluster_index"}
            if data_output2 is not None:
                data_output2.schema = {"header": ["predicted_cluster_index", "distance"],
                                       "sid_name": "id"}
            predict_datas[0][0] = tuple([data_output1, data_output2])
        else:
            data_output = predict_data
            if data_output is not None:
                data_output.schema = {"header": ["label", "predict_result"], "sid_name": schema.get('sid_name')}
            predict_datas[0][0] = data_output
        return predict_datas
