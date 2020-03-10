import random
from collections import defaultdict

import numpy as np
import tensorflow as tf


class DataConverter(object):
    def convert(self, data, *args, **kwargs):
        pass


class NCFDataConverter(DataConverter):
    def convert(self, data, training=True, *args, **kwargs):
        if training:
            return NCFSequenceData(data, *args, **kwargs)
        else:
            return NCFSequencePredictData(data, *args, **kwargs)


class NCFSequenceData(tf.keras.utils.Sequence):
    def __init__(self, data_instances, batch_size, neg_count):
        """
        Wraps dataset and produces batches for the model to consume. The dataset only has positive clicked data,
        generate negative samples and use neg_count params to control the rate of negative samples.
        :param data_instances: data instances: Instance
        :param batch_size: batch size of data
        :param neg_count: num of negative items
        """

        self.batch_size = batch_size
        self.data_instances = data_instances
        self.neg_count = neg_count
        self._keys = []
        self._user_ids = set()
        self._item_ids = set()

        print(f"initialize class, data type: {type(self.data_instances)}, count:{data_instances.first()}")
        self.size = self.data_instances.count()

        if self.size <= 0:
            raise ValueError("empty data")
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
            self._user_ids.add(u)
            self._item_ids.add(i)

        self._n_users, self._n_items = max(self._user_ids) + 1, self._item_ids.__len__()

        # Get a list version so we do not need to perform type casting
        self.user_items = dict(self.user_items)
        self.item_users = dict(self.item_users)

        self.users = None
        self.items = None
        self.y = None
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
        size = self.size * (self.neg_count + 1) if self.neg_count > 0 else self.batch_size
        users = np.zeros((size,), dtype=np.uint32)
        items = np.zeros((size,), dtype=np.uint32)
        y = np.zeros((size,), dtype=np.uint32)

        idx = 0
        for key, instance in self.data_instances.collect():
            feature = np.array(instance.features).squeeze().astype(int).tolist()
            user_idx = feature[0]
            item_idx = feature[1]
            # add postive sample
            users[idx] = user_idx
            items[idx] = item_idx
            y[idx] = 1
            idx += 1
            # TODO: set positive values outside of for loop
            for _ in range(self.neg_count):
                # if idx % 1000 == 0: print(idx)
                neg_item_idx = self._sample_negative_item(user_idx)
                users[idx] = user_idx
                items[idx] = neg_item_idx
                y[idx] = 0
                idx += 1

        self.size = idx
        shuffle_idx = [i for i in range(idx)]
        random.shuffle(shuffle_idx)
        # TODO: need to transfer into class Instance
        self.users = users[shuffle_idx]
        self.items = items[shuffle_idx]
        self.y = y[shuffle_idx]
        self._keys = range(0, idx)

    def __getitem__(self, index):
        """Gets batch at position `index`.

        # Arguments
            index: position of the batch in the Sequence.

        # Returns
            A batch
        """
        start = self.batch_size * index
        end = self.batch_size * (index + 1)
        X = [self.users[start: end], self.items[start: end]]
        y = self.y[start:end]
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
        return self.y.astype(int).tolist()


class NCFSequencePredictData(tf.keras.utils.Sequence):
    def __init__(self, data_instances, batch_size):
        self.batch_size = batch_size
        self.data_instances = data_instances
        self.size = data_instances.count()

        if self.size <= 0:
            raise ValueError("empty data")
        self.batch_size = batch_size if batch_size > 0 else self.size
        self._keys = []

        self.users = None
        self.items = None
        self.transfer_data()

    def transfer_data(self):
        size = self.size if self.size > 0 else self.batch_size
        users = np.zeros((size, 1), dtype=np.uint32)
        items = np.zeros((size, 1), dtype=np.uint32)

        idx = 0
        for key, instance in self.data_instances.collect():
            self._keys.append(key)
            feature = np.array(instance.features).squeeze().astype(int).tolist()
            user_idx = feature[0]
            item_idx = feature[1]
            users[idx] = user_idx
            items[idx] = item_idx
            idx += 1
        self.users = users
        self.items = items

    def __getitem__(self, index):
        """Gets batch at position `index`.

        # Arguments
            index: position of the batch in the Sequence.

        # Returns
            A batch
        """
        start = self.batch_size * index
        end = self.batch_size * (index + 1)
        return [self.users[start: end], self.items[start: end]]

    def __len__(self):
        """Number of batch in the Sequence.

        # Returns
            The number of batches in the Sequence.
        """
        return int(np.ceil(self.size / float(self.batch_size)))

    def get_keys(self):
        return self._keys
