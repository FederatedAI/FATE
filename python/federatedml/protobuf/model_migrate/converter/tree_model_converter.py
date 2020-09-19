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


from typing import Dict

from federatedml.protobuf.generated.boosting_tree_model_meta_pb2 import BoostingTreeModelMeta
from federatedml.protobuf.generated.boosting_tree_model_param_pb2 import BoostingTreeModelParam
from federatedml.protobuf.model_migrate.converter.converter_base import AutoReplace
from federatedml.protobuf.model_migrate.converter.converter_base import ProtoConverterBase


class HeteroSBTConverter(ProtoConverterBase):

    def convert(self, param: BoostingTreeModelParam, meta: BoostingTreeModelMeta,
                guest_id_mapping: Dict,
                host_id_mapping: Dict,
                arbiter_id_mapping: Dict
                ):

        feat_importance_list = list(param.feature_importances)
        tree_list = list(param.trees_)
        replacer = AutoReplace(guest_id_mapping, host_id_mapping, arbiter_id_mapping)

        # fp == feature importance
        for fp in feat_importance_list:
            fp.sitename = replacer.replace(fp.sitename)
            fp.fullname = replacer.replace(fp.fullname)

        for tree in tree_list:
            tree_nodes = list(tree.tree_)
            for node in tree_nodes:
                node.sitename = replacer.replace(node.sitename)

        return param, meta
