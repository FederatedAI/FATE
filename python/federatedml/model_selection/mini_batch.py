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

import random
from fate_arch.session import computing_session as session
from federatedml.model_selection import indices

from federatedml.util import LOGGER


class MiniBatch:
    def __init__(self, data_inst, batch_size=320, shuffle=False, batch_strategy="full", masked_rate=0):
        self.batch_data_sids = None
        self.batch_nums = 0
        self.data_inst = data_inst
        self.all_batch_data = None
        self.all_index_data = None
        self.data_sids_iter = None
        self.batch_data_generator = None
        self.batch_mutable = False
        self.batch_masked = False

        if batch_size == -1:
            self.batch_size = data_inst.count()
        else:
            self.batch_size = batch_size

        self.__init_mini_batch_data_seperator(data_inst, self.batch_size, batch_strategy, masked_rate, shuffle)

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
        LOGGER.debug("Currently, batch_num is: {}".format(self.batch_nums))
        if result == 'index':
            for index_table in self.all_index_data:
                yield index_table
        elif result == "data":
            for batch_data in self.all_batch_data:
                yield batch_data
        else:
            for batch_data, index_table in zip(self.all_batch_data, self.all_index_data):
                yield batch_data, index_table

        # if self.batch_mutable:
        #     self.__generate_batch_data()
    def __init_mini_batch_data_seperator(self, data_insts, batch_size, batch_strategy, masked_rate, shuffle):
        self.data_sids_iter, data_size = indices.collect_index(data_insts)

        self.batch_data_generator = get_batch_generator(
            data_size, batch_size, batch_strategy, masked_rate, shuffle=shuffle)
        self.batch_nums = self.batch_data_generator.batch_nums
        self.batch_mutable = self.batch_data_generator.batch_mutable()
        self.masked_batch_size = self.batch_data_generator.masked_batch_size

        if self.batch_mutable is False:
            self.__generate_batch_data()

    def generate_batch_data(self):
        if self.batch_mutable:
            self.__generate_batch_data()

    def __generate_batch_data(self):
        self.all_index_data, self.all_batch_data = self.batch_data_generator.generate_data(
            self.data_inst, self.data_sids_iter)


def get_batch_generator(data_size, batch_size, batch_strategy, masked_rate, shuffle):
    if batch_size >= data_size:
        LOGGER.warning("As batch_size >= data size, all batch strategy will be disabled")
        return FullBatchDataGenerator(data_size, data_size, shuffle=False)

    # if round((masked_rate + 1) * batch_size) >= data_size:
        # LOGGER.warning("Masked dataset's batch_size >= data size, batch shuffle will be disabled")
        # return FullBatchDataGenerator(data_size, data_size, shuffle=False, masked_rate=masked_rate)
    if batch_strategy == "full":
        if masked_rate > 0:
            LOGGER.warning("If using full batch strategy and masked rate > 0, shuffle will always be true")
            shuffle = True
        return FullBatchDataGenerator(data_size, batch_size, shuffle=shuffle, masked_rate=masked_rate)
    else:
        if shuffle:
            LOGGER.warning("if use random select batch strategy, shuffle will not work")
        return RandomBatchDataGenerator(data_size, batch_size, masked_rate)


class BatchDataGenerator(object):
    def __init__(self, data_size, batch_size, shuffle=False, masked_rate=0):
        self.batch_nums = None
        self.masked_batch_size = min(data_size, round((1 + masked_rate) * batch_size))
        self.batch_size = batch_size
        self.shuffle = shuffle

    def batch_mutable(self):
        return True

    @staticmethod
    def _generate_batch_data_with_batch_ids(data_insts, batch_ids, masked_ids=None):
        batch_index_table = session.parallelize(batch_ids,
                                                include_key=True,
                                                partition=data_insts.partitions)
        batch_data_table = batch_index_table.join(data_insts, lambda x, y: y)

        if masked_ids:
            masked_index_table = session.parallelize(masked_ids,
                                                     include_key=True,
                                                     partition=data_insts.partitions)
            return masked_index_table, batch_data_table
        else:
            return batch_index_table, batch_data_table


class FullBatchDataGenerator(BatchDataGenerator):
    def __init__(self, data_size, batch_size, shuffle=False, masked_rate=0):
        super(FullBatchDataGenerator, self).__init__(data_size, batch_size, shuffle, masked_rate=masked_rate)
        self.batch_nums = (data_size + batch_size - 1) // batch_size

        LOGGER.debug(f"Init Full Batch Data Generator, batch_nums: {self.batch_nums}, batch_size: {self.batch_size}, "
                     f"masked_batch_size: {self.masked_batch_size}, shuffle: {self.shuffle}")

    def generate_data(self, data_insts, data_sids):
        if self.shuffle:
            random.SystemRandom().shuffle(data_sids)

        index_table = []
        batch_data = []
        if self.batch_size != self.masked_batch_size:
            for bid in range(self.batch_nums):
                batch_ids = data_sids[bid * self.batch_size:(bid + 1) * self.batch_size]
                masked_ids_set = set()
                for sid, _ in batch_ids:
                    masked_ids_set.add(sid)
                possible_ids = random.SystemRandom().sample(data_sids, self.masked_batch_size)
                for pid, _ in possible_ids:
                    if pid not in masked_ids_set:
                        masked_ids_set.add(pid)
                        if len(masked_ids_set) == self.masked_batch_size:
                            break

                masked_ids = zip(list(masked_ids_set), [None] * len(masked_ids_set))
                masked_index_table, batch_data_table = self._generate_batch_data_with_batch_ids(data_insts,
                                                                                                batch_ids,
                                                                                                masked_ids)
                index_table.append(masked_index_table)
                batch_data.append(batch_data_table)
        else:
            for bid in range(self.batch_nums):
                batch_ids = data_sids[bid * self.batch_size: (bid + 1) * self.batch_size]
                batch_index_table, batch_data_table = self._generate_batch_data_with_batch_ids(data_insts, batch_ids)
                index_table.append(batch_index_table)
                batch_data.append(batch_data_table)

        return index_table, batch_data

    def batch_mutable(self):
        return self.masked_batch_size > self.batch_size or self.shuffle


class RandomBatchDataGenerator(BatchDataGenerator):
    def __init__(self, data_size, batch_size, masked_rate=0):
        super(RandomBatchDataGenerator, self).__init__(data_size, batch_size, shuffle=False, masked_rate=masked_rate)
        self.batch_nums = 1

        LOGGER.debug(f"Init Random Batch Data Generator, batch_nums: {self.batch_nums}, batch_size: {self.batch_size}, "
                     f"masked_batch_size: {self.masked_batch_size}")

    def generate_data(self, data_insts, data_sids):
        if self.masked_batch_size == self.batch_size:
            batch_ids = random.SystemRandom().sample(data_sids, self.batch_size)
            batch_index_table, batch_data_table = self._generate_batch_data_with_batch_ids(data_insts, batch_ids)
            batch_data_table = batch_index_table.join(data_insts, lambda x, y: y)
            return [batch_index_table], [batch_data_table]
        else:
            masked_ids = random.SystemRandom().sample(data_sids, self.masked_batch_size)
            batch_ids = masked_ids[: self.batch_size]
            masked_index_table, batch_data_table = self._generate_batch_data_with_batch_ids(data_insts,
                                                                                            batch_ids,
                                                                                            masked_ids)
            return [masked_index_table], [batch_data_table]
