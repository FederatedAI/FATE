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


class DataConverter(object):
    def convert(self, data, *args, **kwargs):
        pass


class GMFDataConverter(DataConverter):
    def convert(self, data, training=True, *args, **kwargs):
        if training:
            return GMFSequenceData(data, *args, **kwargs)
        else:
            return GMFSequencePredictData(data, *args, **kwargs)


class GMFSequenceData(tf.keras.utils.Sequence):
    def __init__(self, data_instances, batch_size, neg_count, flow_id='training'):
        """
        Wraps dataset and produces batches for the model to consume
        :param data_instances: data instances: Instance
        :param batch_size: batch size of data
        :param neg_count: num of negative items
        """

        self.batch_size = batch_size
        self.data_instances = data_instances
        self.neg_count = neg_count
        self.max_length = None
        self._keys = []
        self.flow_id = flow_id
        self._user_ids = set()
        self._item_ids = set()

        print(f"initialize class, data type: {type(self.data_instances)}, count:{data_instances.first()}")
        self.size = self.data_instances.count()
        self.y_1 = np.ones((self.size * self.neg_count, 1))
        self.y_0 = np.zeros((self.size * self.neg_count, 1))

        if self.size <= 0:
            raise ValueError("empty data")
        self.batch_size = batch_size if batch_size > 0 else self.size

        # Neighborhoods
        self.user_items = defaultdict(set)
        self.item_users = defaultdict(set)

        for key, instance in self.data_instances.collect():
            features = np.array(instance.features).astype(int).squeeze().tolist()
            u = features[0]
            i = features[1]
            self.user_items[u].add(i)
            self.item_users[i].add(u)
            self._user_ids.add(u)
            self._item_ids.add(i)
        # Get a list version so we do not need to perform type casting
        self.user_items = dict(self.user_items)
        self.item_users = dict(self.item_users)
        self._n_users, self._n_items = max(self._user_ids) + 1, self._item_ids.__len__()

        self.users = None
        self.items = None
        self.neg_items = None
        self.validate_users = None
        self.validate_items = None
        self.validate_y = None
        self.transfer_data()

    @property
    def data_size(self):
        return len(self.size)

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
        return np.random.choice(self.item_ids)

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
        # Allocate inputs
        # print("transfer_data")
        size = self.size * self.neg_count if self.neg_count > 0 else self.batch_size
        users = np.zeros((size,), dtype=np.uint32)
        items = np.zeros((size,), dtype=np.uint32)
        neg_items = np.zeros((size,), dtype=np.uint32)
        validate_size = self.size * (self.neg_count + 1) if self.neg_count > 0 else self.batch_size
        validate_users = np.zeros((validate_size,), dtype=np.uint32)
        validate_items = np.zeros((validate_size,), dtype=np.uint32)
        validate_y = np.zeros((validate_size,), dtype=np.uint32)

        if self.flow_id != 'validate':
            self._keys = range(0, size)
            self.size = size
        else:
            self._keys = range(0, validate_size)
            self.size = validate_size

        idx = 0
        valid_idx = 0
        for key, instance in self.data_instances.collect():
            feature = np.array(instance.features).squeeze().astype(int).tolist()
            user_idx = feature[0]
            item_idx = feature[1]
            validate_items[valid_idx] = item_idx
            validate_users[valid_idx] = user_idx
            validate_y[valid_idx] = 1
            valid_idx += 1
            # TODO: set positive values outside of for loop
            for _ in range(self.neg_count):
                neg_item_idx = self._sample_negative_item(user_idx)
                users[idx] = user_idx
                items[idx] = item_idx
                neg_items[idx] = neg_item_idx
                idx += 1
                validate_items[valid_idx] = neg_item_idx
                validate_users[valid_idx] = user_idx
                valid_idx += 1

        # shuffle data
        self.size = idx
        shuffle_idx = [i for i in range(idx)]
        random.shuffle(shuffle_idx)
        self.users = users[shuffle_idx]
        self.items = items[shuffle_idx]
        self.neg_items = neg_items[shuffle_idx]
        self.y_1 = self.y_1[shuffle_idx]
        self.y_0 = self.y_0[shuffle_idx]

        valid_shuffle_idx = [i for i in range(valid_idx)]
        random.shuffle(valid_shuffle_idx)
        self.validate_users = validate_users[valid_shuffle_idx]
        self.validate_items = validate_items[valid_shuffle_idx]
        self.validate_y = validate_y[valid_shuffle_idx]

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
            X = [self.users[start: end], self.items[start: end], self.neg_items[start: end]]
            # y = self.y_0[start:end, :]
            y = [self.y_1[start:end, :], self.y_0[start:end, :], self.y_0[start:end, :]]
            # y = [self.y_1[start:end, :], self.y_0[start:end, :]]
        else:
            X = [self.validate_users[start: end], self.validate_items[start: end]]
            y = self.validate_y[start:end]
        return X, y

    def __len__(self):
        """Number of batch in the Sequence.

        # Returns
            The number of batches in the Sequence.
        """
        return int(np.ceil(self.size / float(self.batch_size)))

    def get_keys(self):
        return self._keys

    def get_validate_labels(self):
        return self.validate_y.astype(int).tolist()


class GMFSequencePredictData(tf.keras.utils.Sequence):
    def __init__(self, data_instances, batch_size, *args, **kwargs):
        self.batch_size = batch_size
        self.data_instances = data_instances
        self.size = data_instances.count()

        if self.size <= 0:
            raise ValueError("empty data")
        self.batch_size = batch_size if batch_size > 0 else self.size
        self._keys = []

        self.users = None
        self.items = None
        self.labels = None
        self.item_sets = set()
        self.user_sets = set()
        self.transfer_data()

    def transfer_data(self):
        size = self.size if self.size > 0 else self.batch_size
        users = np.zeros((size, 1), dtype=np.uint32)
        items = np.zeros((size, 1), dtype=np.uint32)
        labels = np.zeros((size, 1), dtype=np.uint32)

        idx = 0
        for key, instance in self.data_instances.collect():
            self._keys.append(key)
            feature = np.array(instance.features).squeeze().astype(int).tolist()
            user_idx = feature[0]
            item_idx = feature[1]
            label = feature[2]
            users[idx] = user_idx
            items[idx] = item_idx
            labels[idx] = label
            self.item_sets.add(item_idx)
            self.user_sets.add(user_idx)
            idx += 1
        self.users = users
        self.items = items
        self.labels = labels

    def __getitem__(self, index):
        """Gets batch at position `index`.

        # Arguments
            index: position of the batch in the Sequence.

        # Returns
            A batch
        """
        start = self.batch_size * index
        end = self.batch_size * (index + 1)
        return [self.users[start: end], self.items[start: end]], self.labels[start:end]

    def __len__(self):
        """Number of batch in the Sequence.

        # Returns
            The number of batches in the Sequence.
        """
        return int(np.ceil(self.size / float(self.batch_size)))

    def get_keys(self):
        return self._keys

    @property
    def user_count(self):
        """
        Number of users in dataset
        """
        return self.user_sets.__len__()

    @property
    def item_count(self):
        """
        Number of items in dataset
        """
        return self.item_sets.__len__()
