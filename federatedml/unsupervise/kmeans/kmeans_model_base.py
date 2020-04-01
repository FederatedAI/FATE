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


from federatedml.model_base import ModelBase
from federatedml.transfer_variable.transfer_class.hetero_kmeans_transfer_variable import HeteroKmeansTransferVariable
from federatedml.util import abnormal_detection

class BaseKmeansModel(ModelBase):
    def __init__(self):
        super(BaseKmeansModel, self).__init__()
        self.model_param=None
        self.n_iter_ = 0
        self.k = 0
        self.max_iter = 0
        self.tol = 0
        self.iter = iter
        self.centroid_list = None
        self.cluster_result = None

    def _init_model(self, params):
        self.model_param = params
        self.k = params.k
        self.max_iter = params.max_iter
        self.tol = params.tol
        self.transfer_variable = HeteroKmeansTransferVariable()


    def _abnormal_detection(self, data_instances):
        """
        Make sure input data_instances is valid.
        """
        abnormal_detection.empty_table_detection(data_instances)
        abnormal_detection.empty_feature_detection(data_instances)