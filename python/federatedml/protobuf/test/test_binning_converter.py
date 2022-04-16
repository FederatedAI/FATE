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

from federatedml.protobuf.model_migrate.converter.binning_model_converter import FeatureBinningConverter


from federatedml.protobuf.generated.feature_binning_meta_pb2 import FeatureBinningMeta
from federatedml.protobuf.generated.feature_binning_param_pb2 import FeatureBinningParam

from federatedml.protobuf.model_migrate.converter.tree_model_converter import HeteroSBTConverter
from federatedml.protobuf.model_migrate.model_migrate import model_migration
import copy

host_old = [10000, 9999]
host_new = [114, 514, ]

guest_old = [10000]
guest_new = [1919]

param = FeatureBinningParam()

old_header = ['host_10000_0', 'host_10000_1', 'host_10000_2', 'host_10000_3']
param.header_anonymous = old_header

rs = model_migration({'HelloParam': param, 'HelloMeta': {}}, 'HeteroSecureBoost', old_guest_list=guest_old,
                     new_guest_list=guest_new, old_host_list=host_old, new_host_list=host_new, )
print(rs)
