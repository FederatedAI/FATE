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

from arch.api import session
from arch.api.utils import log_utils
from federatedml.model_selection import indices

LOGGER = log_utils.getLogger()


class MiniBatch:
    def __init__(self, data_inst, batch_size=320):
        self.batch_data_sids = None
        self.batch_nums = 0
        self.data_inst = data_inst
        self.all_batch_data = None
        self.all_index_data = None

        if batch_size == -1:
            self.batch_size = data_inst.count()
        else:
            self.batch_size = batch_size

        self.__mini_batch_data_seperator(data_inst, batch_size)
        # LOGGER.debug("In mini batch init, batch_num:{}".format(self.batch_nums))

    def mini_batch_data_generator(self, result='data'):
        """
        Generate mini-batch data or index

        Parameters
        ----------
        result : str, 'data' or 'index', default: 'data'
            Specify you want batch data or batch index.

        Returns
        -------
        A generator that might generate data or index.
        """
        LOGGER.debug("Currently, len of all_batch_data: {}".format(len(self.all_batch_data)))
        if result == 'index':
            for index_table in self.all_index_data:
                yield index_table
        else:
            for batch_data in self.all_batch_data:
                yield batch_data

    def __mini_batch_data_seperator(self, data_insts, batch_size):
        data_sids_iter, data_size = indices.collect_index(data_insts)

        if batch_size > data_size:
            batch_size = data_size
            self.batch_size = batch_size

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
        all_index_data = []
        for index_data in batch_data_sids:
            # LOGGER.debug('in generator, index_data is {}'.format(index_data))
            index_table = session.parallelize(index_data, include_key=True, partition=data_insts._partitions)
            batch_data = index_table.join(data_insts, lambda x, y: y)

            # yield batch_data
            all_batch_data.append(batch_data)
            all_index_data.append(index_table)
        self.all_batch_data = all_batch_data
        self.all_index_data = all_index_data
