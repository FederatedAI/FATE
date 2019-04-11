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

import time

from arch.api import eggroll
from arch.api.utils import log_utils
from federatedml.model_selection import indices

LOGGER = log_utils.getLogger()


class MiniBatch:
    def __init__(self, data_inst=None, batch_size=320):
        self.batch_data_sids = None
        self.batch_nums = 0
        self.data_inst = data_inst
        self.batch_size = batch_size
        self.all_batch_data = None
        if self.data_inst is not None and batch_size is not None:
            self.batch_data_sids = self.__mini_batch_data_seperator(data_inst, batch_size)
            # LOGGER.debug("In mini batch init, batch_num:{}".format(self.batch_nums))

    def mini_batch_data_generator(self, data_inst=None, batch_size=None):

        if data_inst is not None or (batch_size is not None and batch_size != self.batch_size):
            self.batch_data_sids = self.__mini_batch_data_seperator(data_inst, batch_size)
            self.batch_size = batch_size

        for batch_data in self.all_batch_data:
            yield batch_data

    def mini_batch_index_generator(self, data_inst=None, batch_size=320):
        if data_inst is not None or batch_size != self.batch_size:
            self.batch_data_sids = self.__mini_batch_data_seperator(data_inst, batch_size)
            self.batch_size = batch_size
        batch_data_sids = self.batch_data_sids

        for bid in range(len(batch_data_sids)):
            index_data = batch_data_sids[bid]
            index_table = eggroll.parallelize(index_data, include_key=True, partition=data_inst._partitions)
            yield index_table

    def __mini_batch_data_seperator(self, data_insts, batch_size):
        data_sids_iter, data_size = indices.collect_index(data_insts)
        batch_nums = (data_size + batch_size - 1) // batch_size

        batch_data_sids = []
        curt_data_num = 0
        curt_batch = 0
        curt_batch_ids = []
        for sid, values in data_sids_iter:
            # print('sid is {}, values is {}'.format(sid, values))
            curt_batch_ids.append((sid, None))
            curt_data_num += 1
            if curt_data_num % batch_size == 0:
                curt_batch += 1
                if curt_batch < batch_nums:
                    batch_data_sids.append(curt_batch_ids)
                    curt_batch_ids = []
            if curt_data_num == data_size and len(curt_batch_ids) != 0:
                batch_data_sids.append(curt_batch_ids)

        self.batch_nums = len(batch_data_sids)

        all_batch_data = []
        for index_data in batch_data_sids:
            # LOGGER.debug('in generator, index_data is {}'.format(index_data))
            t0 = time.time()
            index_table = eggroll.parallelize(index_data, include_key=True, partition=data_insts._partitions)
            t1 = time.time()
            batch_data = index_table.join(data_insts, lambda x, y: y)
            t2 = time.time()
            LOGGER.debug('[compute] parallelize index table time: {}'.format(t1 - t0))
            LOGGER.debug('[compute] join time: {}'.format(t2 - t1))
            # yield batch_data
            all_batch_data.append(batch_data)
        self.all_batch_data = all_batch_data
        return batch_data_sids
