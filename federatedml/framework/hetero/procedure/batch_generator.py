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

from arch.api.utils import log_utils
from federatedml.framework.hetero.sync import batch_info_sync
from federatedml.model_selection import MiniBatch

LOGGER = log_utils.getLogger()


class Guest(batch_info_sync.Guest):
    def __init__(self):
        self.mini_batch_obj = None
        self.finish_sycn = False
        self.batch_nums = None

    def register_batch_generator(self, transfer_variables):
        self._register_batch_data_index_transfer(transfer_variables.batch_info, transfer_variables.batch_data_index)

    def initialize_batch_generator(self, data_instances, batch_size, suffix=tuple()):
        self.mini_batch_obj = MiniBatch(data_instances, batch_size=batch_size)
        self.batch_nums = self.mini_batch_obj.batch_nums
        batch_info = {"batch_size": batch_size, "batch_num": self.batch_nums}
        self.sync_batch_info(batch_info, suffix)
        index_generator = self.mini_batch_obj.mini_batch_data_generator(result='index')
        batch_index = 0
        for batch_data_index in index_generator:
            batch_suffix = suffix + (batch_index,)
            self.sync_batch_index(batch_data_index, batch_suffix)
            batch_index += 1

    def generate_batch_data(self):
        data_generator = self.mini_batch_obj.mini_batch_data_generator(result='data')
        for batch_data in data_generator:
            yield batch_data


class Host(batch_info_sync.Host):
    def __init__(self):
        self.finish_sycn = False
        self.batch_data_insts = []
        self.batch_nums = None

    def register_batch_generator(self, transfer_variables):
        self._register_batch_data_index_transfer(transfer_variables.batch_info, transfer_variables.batch_data_index)

    def initialize_batch_generator(self, data_instances, suffix=tuple()):
        batch_info = self.sync_batch_info(suffix)
        self.batch_nums = batch_info.get('batch_num')
        for batch_index in range(self.batch_nums):
            batch_suffix = suffix + (batch_index,)
            batch_data_index = self.sync_batch_index(suffix=batch_suffix)
            batch_data_inst = batch_data_index.join(data_instances, lambda g, d: d)
            self.batch_data_insts.append(batch_data_inst)

    def generate_batch_data(self):
        batch_index = 0
        for batch_data_inst in self.batch_data_insts:
            LOGGER.info("batch_num: {}, batch_data_inst size:{}".format(
                batch_index, batch_data_inst.count()))
            yield batch_data_inst
            batch_index += 1


class Arbiter(batch_info_sync.Arbiter):
    def __init__(self):
        self.batch_num = None

    def register_batch_generator(self, transfer_variables):
        self._register_batch_data_index_transfer(transfer_variables.batch_info, transfer_variables.batch_data_index)

    def initialize_batch_generator(self, suffix=tuple()):
        batch_info = self.sync_batch_info(suffix)
        self.batch_num = batch_info.get('batch_num')

    def generate_batch_data(self):
        for i in range(self.batch_num):
            yield i
