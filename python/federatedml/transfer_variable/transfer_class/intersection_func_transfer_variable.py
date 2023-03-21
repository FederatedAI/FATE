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
# AUTO GENERATED TRANSFER VARIABLE CLASS. DO NOT MODIFY
#
################################################################################

from federatedml.transfer_variable.base_transfer_variable import BaseTransferVariables


# noinspection PyAttributeOutsideInit
class IntersectionFuncTransferVariable(BaseTransferVariables):
    def __init__(self, flowid=0):
        super().__init__(flowid)
        self.id_map_from_guest = self._create_variable(name='id_map_from_guest', src=['guest'], dst=['host'])
        self.id_map_from_host = self._create_variable(name='id_map_from_host', src=['host'], dst=['guest'])
        self.info_share_from_host = self._create_variable(name='info_share_from_host', src=['host'], dst=['guest'])
        self.info_share_from_guest = self._create_variable(name='info_share_from_guest', src=['guest'], dst=['host'])
        self.join_id_from_guest = self._create_variable(name='join_id_from_guest', src=['guest'], dst=['host'])
        self.join_id_from_host = self._create_variable(name='join_id_from_host', src=['host'], dst=['guest'])
        self.intersect_filter_from_host = self._create_variable(name='intersect_filter_from_host', src=['host'],
                                                                dst=['guest'])
        self.intersect_filter_from_guest = self._create_variable(name='intersect_filter_from_guest', src=['guest'],
                                                                 dst=['host'])
        self.cache_id = self._create_variable(name='cache_id', src=['guest', 'host'], dst=['host', 'guest'])
        self.cache_id_from_host = self._create_variable(name='cache_id_from_host', src=['host'], dst=['guest'])
        self.use_match_id = self._create_variable(name='use_match_id', src=['host'], dst=['guest'])
