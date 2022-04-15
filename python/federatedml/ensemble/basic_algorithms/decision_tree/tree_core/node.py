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
# Decision Tree Node Struture
# =============================================================================

from federatedml.util import consts


class Node(object):
    def __init__(self, id=None, sitename=None, fid=-1,
                 bid=-1, weight=0, is_leaf=False, sum_grad=None,
                 sum_hess=None, left_nodeid=-1, right_nodeid=-1,
                 missing_dir=1, sample_num=0, parent_nodeid=None, is_left_node=False, sibling_nodeid=None,
                 inst_indices=None):
        self.id = id
        self.sitename = sitename
        self.fid = fid
        self.bid = bid
        self.weight = weight
        self.is_leaf = is_leaf
        self.sum_grad = sum_grad
        self.sum_hess = sum_hess
        self.left_nodeid = left_nodeid
        self.right_nodeid = right_nodeid
        self.missing_dir = missing_dir
        self.parent_nodeid = parent_nodeid
        self.sample_num = sample_num
        self.is_left_node = is_left_node
        self.sibling_nodeid = sibling_nodeid
        self.inst_indices = inst_indices

    def __str__(self):
        return "id{}, fid:{},bid:{},weight:{},sum_grad:{},sum_hess:{},left_node:{},right_node:{}, sitename:{}, " \
               "is leaf {}".format(self.id,
                                   self.fid, self.bid, self.weight, self.sum_grad, self.sum_hess, self.left_nodeid,
                                   self.right_nodeid,
                                   self.sitename, self.is_leaf
                                   )
