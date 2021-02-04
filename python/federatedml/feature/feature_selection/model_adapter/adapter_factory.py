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

from federatedml.feature.feature_selection.model_adapter.statistic_adapter import StatisticAdapter
from federatedml.feature.feature_selection.model_adapter.binning_adapter import BinningAdapter
from federatedml.feature.feature_selection.model_adapter.psi_adapter import PSIAdapter
from federatedml.feature.feature_selection.model_adapter import tree_adapter
from federatedml.feature.feature_selection.model_adapter import pearson_adapter
from federatedml.util import consts


def adapter_factory(model_name):
    if model_name == consts.STATISTIC_MODEL:
        return StatisticAdapter()
    elif model_name == consts.BINNING_MODEL:
        return BinningAdapter()
    elif model_name == consts.PSI:
        return PSIAdapter()
    elif model_name == consts.HETERO_SBT:
        return tree_adapter.HeteroSBTAdapter()
    elif model_name == consts.HOMO_SBT:
        return tree_adapter.HomoSBTAdapter()
    elif model_name in [consts.HETERO_FAST_SBT_MIX, consts.HETERO_FAST_SBT_LAYERED]:
        return tree_adapter.HeteroFastSBTAdapter()
    elif model_name == "HeteroPearson":
        return pearson_adapter.PearsonAdapter()
    else:
        raise ValueError(f"Cannot recognize model_name: {model_name}")
