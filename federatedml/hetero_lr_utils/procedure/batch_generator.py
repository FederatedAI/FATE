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

from federatedml.hetero_lr_utils.sync import batch_info_sync
from federatedml.model_selection import MiniBatch


class Guest(batch_info_sync.Guest):

    def __init__(self):
        self.mini_batch_obj = None
        self.batch_data_list = []


    def register_batch_generator(self, transfer_variables):
        self._register_batch_data_index_transfer(transfer_variables.batch_info, transfer_variables.batch_data_index)

    def initialize_batch_generator(self, data_instances, batch_size, suffix=tuple()):
        self.mini_batch_obj = MiniBatch(data_instances, batch_size=batch_size)
        batch_info = {"batch_size": batch_size, "batch_num": self.mini_batch_obj.batch_nums}
        self.sync_batch_info(batch_info, suffix)

    def generate_batch_data(self):



