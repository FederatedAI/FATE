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
class RawIntersectTransferVariable(BaseTransferVariables):
    def __init__(self, flowid=0):
        super().__init__(flowid)
        self.intersect_ids_guest = self._create_variable(name='intersect_ids_guest', src=['guest'], dst=['host'])
        self.intersect_ids_host = self._create_variable(name='intersect_ids_host', src=['host'], dst=['guest'])
        self.send_ids_guest = self._create_variable(name='send_ids_guest', src=['guest'], dst=['host'])
        self.send_ids_host = self._create_variable(name='send_ids_host', src=['host'], dst=['guest'])
        self.sync_intersect_ids_multi_hosts = self._create_variable(name='sync_intersect_ids_multi_hosts', src=['guest'], dst=['host'])
