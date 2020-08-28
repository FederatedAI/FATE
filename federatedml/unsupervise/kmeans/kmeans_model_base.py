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
from arch.api.utils import log_utils
from federatedml.model_base import ModelBase
from federatedml.transfer_variable.transfer_class.hetero_kmeans_transfer_variable import HeteroKmeansTransferVariable
from federatedml.util import abnormal_detection
from federatedml.param.hetero_kmeans_param import KmeansParam
from federatedml.protobuf.generated import hetero_kmeans_meta_pb2, hetero_kmeans_param_pb2

LOGGER = log_utils.getLogger()

class BaseKmeansModel(ModelBase):
    def __init__(self):
        super(BaseKmeansModel, self).__init__()
        self.model_param=KmeansParam()
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

    def get_header(self, data_instances):
        if self.header is not None:
            return self.header
        return data_instances.schema.get("header")

    def _get_param(self):
        header = self.header
        LOGGER.debug("In get_param, header: {}".format(header))
        if header is None:
            param_protobuf_obj = hetero_kmeans_param_pb2.KmeansModelParam()
            return param_protobuf_obj
        param_protobuf_obj = hetero_kmeans_param_pb2.KmeansModelParam(iters=self.n_iter_)
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
    def sum_dict(d1,d2):
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

