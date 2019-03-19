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
# DecisionTree Base Class
# =============================================================================
from federatedml.util import consts


class BoostingTreeModelMeta(object):
    def __init__(self):
        self.objective_param = None
        self.task_type = consts.CLASSIFICATION
        self.trees_ = []
        self.init_score = []
        self.tree_dim = 0
        self.num_classes = 0
        self.classes_ = None
        self.loss = []
