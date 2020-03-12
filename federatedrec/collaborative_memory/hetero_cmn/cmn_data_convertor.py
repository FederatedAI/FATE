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

import random
from collections import defaultdict

import numpy as np
import tensorflow as tf

from arch.api.utils import log_utils
LOGGER = log_utils.getLogger()


class DataConverter(object):
    def convert(self, data, *args, **kwargs):
        pass


class CMNDataConverter(DataConverter):
    def convert(self, data, training=True, *args, **kwargs):
        if training:
            return CMNSequenceData(data, *args, **kwargs)
        else:
            return CMNSequencePredictData(data, *args, **kwargs)


class CMNSequenceData(tf.keras.utils.Sequence):
    def __init__(self, data_instances, batch_size, neg_count, flow_id='training'):
        """
        Wraps dataset and produces batches for the model to consume
        :param data_instances: Instance
        :param batch_size: batch size of data
        :param neg_count: num of negative items
        """

        self.batch_size = batch_size
        self.data_instances = data_instances
        self.neg_count = neg_count
        self.max_length = None
        self._keys = []
        self._user_ids = set()
        self._item_ids = set()
        self.flow_id = flow_id

        print(f"initialize class, data type: {type(self.data_instances)}, count:{data_instances.first()}")
        size = self.data_instances.count()
        if size <= 0:
            raise ValueError("empty data")
        self.size = size
        self.y_0 = np.zeros((size * self.neg_count, 1), dtype=np.int32)
        self.y_1 = np.ones((size * self.neg_count, 1), dtype=np.int32)
        self.validate_size = size * (self.neg_count + 1)
        self.validate_y = np.zeros((self.validate_size, 1), dtype=np.int32)
        self.batch_size = batch_size if batch_size > 0 else self.size

        # Neighborhoods
        self.user_items = defaultdict(set)
        self.item_users = defaultdict(set)

        for key, instance in self.data_instances.collect():
            features = np.array(instance.features).squeeze().astype(int).tolist()
            u = features[0]
            i = features[1]
            self.user_items[u].add(i)
            self.item_users[i].add(u)
            self._item_ids.add(i)
            self._user_ids.add(u)
        # Get a list version so we do not need to perform type casting
        self.item_users_list = {k: list(v) for k, v in self.item_users.items()}
        self.max_user_neighbors = max([len(x) for x in self.item_users.values()])
        self.user_items = dict(self.user_items)
        self.item_users = dict(self.item_users)
        self._n_users, self._n_items = max(max(self._user_ids) + 1, self._user_ids.__len__()), self._item_ids.__len__()

        self.users = None
        self.items = None
        self.neg_items = None
        self.neg_length = None
        self.neg_neighbor = None
        self.pos_length = None
        self.pos_neighbor = None

        self.validate_user = None
        self.validate_item = None
        self.validate_length = None
        self.validate_neighbor = None
        self.transfer_data()

    @property
    def data_size(self):
        return self.size

    @property
    def user_ids(self):
        return list(self._user_ids)

    @property
    def item_ids(self):
        return list(self._item_ids)

    @property
    def user_count(self):
        """
        Number of users in dataset
        """
        return self._n_users

    @property
    def item_count(self):
        """
        Number of items in dataset
        """
        return self._n_items

    def _sample_item(self):
        """
        Draw an item uniformly
        """
        return random.choice(self.item_ids)

    def _sample_negative_item(self, user_id):
        """
        Uniformly sample a negative item
        """
        if user_id > self.user_count:
            raise ValueError("Trying to sample user id: {} > user count: {}".format(
                user_id, self.user_count))

        n = self._sample_item()
        positive_items = self.user_items[user_id]

        if len(positive_items) >= self.item_count:
            raise ValueError("The User has rated more items than possible %s / %s" % (
                len(positive_items), self.item_count))
        while n in positive_items:
            n = self._sample_item()
        return n

    def transfer_data(self):
        """
        generate training data and evaluation data in training procedure
        :return:
        """
        size = self.size * self.neg_count if self.neg_count > 0 else self.batch_size
        users = np.zeros((size,), dtype=np.uint32)
        items = np.zeros((size,), dtype=np.uint32)
        neg_items = np.zeros((size,), dtype=np.uint32)
        pos_neighbor = np.zeros((size, self.max_user_neighbors), dtype=np.int32)
        pos_length = np.zeros(size, dtype=np.int32)
        neg_neighbor = np.zeros((size, self.max_user_neighbors), dtype=np.int32)
        neg_length = np.zeros(size, dtype=np.int32)

        validate_user = np.zeros((self.validate_size, ), dtype=np.int32)
        validate_item = np.zeros((self.validate_size, ), dtype=np.int32)
        validate_y = np.zeros((self.validate_size, ), dtype=np.int32)
        validate_neighbor = np.zeros((self.validate_size, self.max_user_neighbors), dtype=np.int32)
        validate_length = np.zeros((self.validate_size,), dtype=np.int32)

        idx = 0
        valid_idx = 0
        for key, instance in self.data_instances.collect():
            self._keys.append(key)
            feature = np.array(instance.features).squeeze().astype(int).tolist()
            user_idx = feature[0]
            item_idx = feature[1]

            validate_user[valid_idx] = user_idx
            validate_item[valid_idx] = item_idx
            validate_y[valid_idx] = 1
            if len(self.item_users.get(item_idx, [])) > 0:
                cur_pos_len = len(self.item_users[item_idx])
                cur_pos_neighbor = self.item_users_list[item_idx]
            else:
                cur_pos_len = 1
                cur_pos_neighbor = [item_idx]

            validate_length[valid_idx] = cur_pos_len
            validate_neighbor[valid_idx, :validate_length[valid_idx]] = cur_pos_neighbor
            valid_idx += 1

            for _ in range(self.neg_count):
                neg_item_idx = self._sample_negative_item(user_idx)
                users[idx] = user_idx
                items[idx] = item_idx
                neg_items[idx] = neg_item_idx

                # Get neighborhood information
                pos_length[idx] = cur_pos_len
                pos_neighbor[idx, :pos_length[idx]] = cur_pos_neighbor

                if len(self.item_users.get(neg_item_idx, [])) > 0:
                    neg_length[idx] = len(self.item_users[neg_item_idx])
                    neg_neighbor[idx, :neg_length[idx]] = self.item_users_list[neg_item_idx]
                    validate_length[valid_idx] = len(self.item_users[neg_item_idx])
                    validate_neighbor[valid_idx, :validate_length[valid_idx]] = self.item_users_list[neg_item_idx]
                else:
                    # Length defaults to 1
                    neg_length[idx] = 1
                    neg_neighbor[idx, 0] = user_idx
                    validate_length[valid_idx] = 1
                    validate_neighbor[valid_idx, 0] = user_idx

                idx += 1
                valid_idx += 1

        if self.flow_id != 'validate':
            self._keys = range(0, idx)
            self.size = idx
        else:
            self._keys = range(0, valid_idx)
            self.size = valid_idx
        self.max_length = max(neg_length.max(), pos_length.max())

        # shuffle data
        shuffle_idx = [i for i in range(idx)]
        random.shuffle(shuffle_idx)
        self.users = users[shuffle_idx]
        self.items = items[shuffle_idx]
        self.neg_items = neg_items[shuffle_idx]
        self.neg_length = neg_length[shuffle_idx]
        self.neg_neighbor = neg_neighbor[shuffle_idx, 0:self.max_length]
        self.pos_length = pos_length[shuffle_idx]
        self.pos_neighbor = pos_neighbor[shuffle_idx, 0:self.max_length]

        # shuffle validate data
        valid_shuffle_idx = [i for i in range(valid_idx)]
        random.shuffle(valid_shuffle_idx)
        self.validate_item = validate_item[valid_shuffle_idx]
        self.validate_user = validate_user[valid_shuffle_idx]
        self.validate_y = validate_y[valid_shuffle_idx]
        self.validate_length = validate_length[valid_shuffle_idx]
        self.validate_neighbor = validate_neighbor[valid_shuffle_idx, :]

    def __getitem__(self, index):
        """Gets batch at position `index`.

        # Arguments
            index: position of the batch in the Sequence.

        # Returns
            A batch
        """
        start = self.batch_size * index
        end = self.batch_size * (index + 1)
        if self.flow_id != 'validate':
            X = [self.users[start: end], self.items[start: end], self.neg_items[start: end],
                 self.pos_length[start:end], self.pos_neighbor[start:end, :],
                 self.neg_length[start:end], self.neg_neighbor[start:end, :]]
            # y = self.y[start:end, :]
            y = [self.y_1[start:end, :], self.y_0[start:end, :], self.y_0[start:end, :]]
        else:
            X = [self.validate_user[start: end], self.validate_item[start: end],
                 self.validate_length[start:end], self.validate_neighbor[start:end, :]]
            y = self.validate_y[start: end]
        return X, y

    def __len__(self):
        """Number of batch in the Sequence.

        # Returns
            The number of batches in the Sequence.
        """
        return int(np.ceil(self.size / float(self.batch_size)))

    def get_keys(self):
        """
        :return: _DTable keys
        """
        return self._keys

    def get_validate_labels(self):
        """
        :return: labels of validation data
        """
        return self.validate_y.astype(int).tolist()


class CMNSequencePredictData(tf.keras.utils.Sequence):
    def __init__(self, data_instances, batch_size, user_items, item_users, max_length, neg_count):
        self.batch_size = batch_size
        self.data_instances = data_instances
        self.max_length = max_length
        self.size = data_instances.count()

        if self.size <= 0:
            raise ValueError("empty data")
        self.batch_size = batch_size if batch_size > 0 else self.size
        self._keys = []

        self.user_items = user_items
        self.item_users = item_users
        self.neg_count = neg_count

        # Get a list version so we do not need to perform type casting
        self.item_users_list = {k: list(v) for k, v in self.item_users.items()}
        self.max_user_neighbors = max([len(x) for x in self.item_users.values()])
        self.user_items = dict(self.user_items)
        self.item_users = dict(self.item_users)

        self.users = None
        self.items = None
        self.pos_length = None
        self.pos_neighbor = None
        self.transfer_data()

    def transfer_data(self):
        size = self.size if self.neg_count > 0 else self.batch_size
        users = np.zeros((size, 1), dtype=np.uint32)
        items = np.zeros((size, 1), dtype=np.uint32)
        pos_neighbor = np.zeros((size, self.max_user_neighbors), dtype=np.int32)
        pos_length = np.zeros(size, dtype=np.int32)

        idx = 0
        for key, instance in self.data_instances.collect():
            self._keys.append(key)
            feature = np.array(instance.features).squeeze().astype(int).tolist()
            user_idx = feature[0]
            item_idx = feature[1]
            users[idx] = user_idx
            items[idx] = item_idx

            # Get neighborhood information
            if len(self.item_users.get(item_idx, [])) > 0:
                pos_length[idx] = len(self.item_users[item_idx])
                pos_neighbor[idx, :pos_length[idx]] = self.item_users_list[item_idx]
            else:
                # Length defaults to 1
                pos_length[idx] = 1
                pos_neighbor[idx, 0] = user_idx

            idx += 1

        # shuffle data
        shuffle_idx = [i for i in range(idx)]
        random.shuffle(shuffle_idx)
        self.users = users[shuffle_idx]
        self.items = items[shuffle_idx]
        self.pos_length = pos_length[shuffle_idx]
        self.pos_neighbor = pos_neighbor[shuffle_idx, :]

    def __getitem__(self, index):
        """Gets batch at position `index`.

        # Arguments
            index: position of the batch in the Sequence.

        # Returns
            A batch
        """
        start = self.batch_size * index
        end = self.batch_size * (index + 1)
        return [self.users[start: end], self.items[start: end], self.pos_length[start:end], self.pos_neighbor[start:end,:]]

    def __len__(self):
        """Number of batch in the Sequence.

        # Returns
            The number of batches in the Sequence.
        """
        return int(np.ceil(self.size / float(self.batch_size)))

    def get_keys(self):
        return self._keys
