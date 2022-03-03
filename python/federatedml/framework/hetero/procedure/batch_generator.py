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

from federatedml.framework.hetero.sync import batch_info_sync
from federatedml.model_selection import MiniBatch
from federatedml.util import LOGGER


class Guest(batch_info_sync.Guest):
    def __init__(self):
        self.mini_batch_obj = None
        self.finish_sycn = False
        self.batch_nums = None
        self.batch_masked = False

    def register_batch_generator(self, transfer_variables, has_arbiter=True):
        self._register_batch_data_index_transfer(transfer_variables.batch_info,
                                                 transfer_variables.batch_data_index,
                                                 getattr(transfer_variables, "batch_validate_info", None),
                                                 has_arbiter)

    def initialize_batch_generator(self, data_instances, batch_size, suffix=tuple(),
                                   shuffle=False, batch_strategy="full", masked_rate=0):
        self.mini_batch_obj = MiniBatch(data_instances, batch_size=batch_size, shuffle=shuffle,
                                        batch_strategy=batch_strategy, masked_rate=masked_rate)
        self.batch_nums = self.mini_batch_obj.batch_nums
        self.batch_masked = self.mini_batch_obj.batch_size != self.mini_batch_obj.masked_batch_size
        batch_info = {"batch_size": self.mini_batch_obj.batch_size, "batch_num": self.batch_nums,
                      "batch_mutable": self.mini_batch_obj.batch_mutable,
                      "masked_batch_size": self.mini_batch_obj.masked_batch_size}
        self.sync_batch_info(batch_info, suffix)

        if not self.mini_batch_obj.batch_mutable:
            self.prepare_batch_data(suffix)

    def prepare_batch_data(self, suffix=tuple()):
        self.mini_batch_obj.generate_batch_data()
        index_generator = self.mini_batch_obj.mini_batch_data_generator(result='index')
        batch_index = 0
        for batch_data_index in index_generator:
            batch_suffix = suffix + (batch_index,)
            self.sync_batch_index(batch_data_index, batch_suffix)
            batch_index += 1

    def generate_batch_data(self, with_index=False, suffix=tuple()):
        if self.mini_batch_obj.batch_mutable:
            self.prepare_batch_data(suffix)

        if with_index:
            data_generator = self.mini_batch_obj.mini_batch_data_generator(result='both')
            for batch_data, index_data in data_generator:
                yield batch_data, index_data
        else:
            data_generator = self.mini_batch_obj.mini_batch_data_generator(result='data')
            for batch_data in data_generator:
                yield batch_data

    def verify_batch_legality(self, suffix=tuple()):
        validate_infos = self.sync_batch_validate_info(suffix)
        least_batch_size = 0
        is_legal = True
        for validate_info in validate_infos:
            legality = validate_info.get("legality")
            if not legality:
                is_legal = False
                least_batch_size = max(least_batch_size, validate_info.get("least_batch_size"))

        if not is_legal:
            raise ValueError(f"To use batch masked strategy, "
                             f"(masked_rate + 1) * batch_size should > {least_batch_size}")


class Host(batch_info_sync.Host):
    def __init__(self):
        self.finish_sycn = False
        self.batch_data_insts = []
        self.batch_nums = None
        self.data_inst = None
        self.batch_mutable = False
        self.batch_masked = False
        self.masked_batch_size = None

    def register_batch_generator(self, transfer_variables, has_arbiter=None):
        self._register_batch_data_index_transfer(transfer_variables.batch_info,
                                                 transfer_variables.batch_data_index,
                                                 getattr(transfer_variables, "batch_validate_info", None))

    def initialize_batch_generator(self, data_instances, suffix=tuple(), **kwargs):
        batch_info = self.sync_batch_info(suffix)
        batch_size = batch_info.get("batch_size")
        self.batch_nums = batch_info.get('batch_num')
        self.batch_mutable = batch_info.get("batch_mutable")
        self.masked_batch_size = batch_info.get("masked_batch_size")
        self.batch_masked = self.masked_batch_size != batch_size

        if not self.batch_mutable:
            self.prepare_batch_data(data_instances, suffix)
        else:
            self.data_inst = data_instances

    def prepare_batch_data(self, data_inst, suffix=tuple()):
        self.batch_data_insts = []
        for batch_index in range(self.batch_nums):
            batch_suffix = suffix + (batch_index,)
            batch_data_index = self.sync_batch_index(suffix=batch_suffix)
            # batch_data_inst = batch_data_index.join(data_instances, lambda g, d: d)
            batch_data_inst = data_inst.join(batch_data_index, lambda d, g: d)
            self.batch_data_insts.append(batch_data_inst)

    def generate_batch_data(self, suffix=tuple()):
        if self.batch_mutable:
            self.prepare_batch_data(data_inst=self.data_inst, suffix=suffix)

        batch_index = 0
        for batch_data_inst in self.batch_data_insts:
            LOGGER.info("batch_num: {}, batch_data_inst size:{}".format(
                batch_index, batch_data_inst.count()))
            yield batch_data_inst
            batch_index += 1

    def verify_batch_legality(self, least_batch_size, suffix=tuple()):
        if self.masked_batch_size <= least_batch_size:
            batch_validate_info = {"legality": False,
                                   "least_batch_size": least_batch_size}
            LOGGER.warning(f"masked_batch_size {self.masked_batch_size} is illegal, should > {least_batch_size}")
        else:
            batch_validate_info = {"legality": True}

        self.sync_batch_validate_info(batch_validate_info, suffix)


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
