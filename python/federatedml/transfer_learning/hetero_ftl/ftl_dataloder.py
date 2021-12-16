import numpy as np

import tensorflow as tf
from federatedml.util import LOGGER


class FTLDataLoader(tf.keras.utils.Sequence):

    def __init__(self, non_overlap_samples, overlap_samples, batch_size, guest_side=True):

        self.batch_size = batch_size
        self.guest_side = guest_side
        self._overlap_index = []
        self._non_overlap_index = []

        if guest_side:
            self.size = non_overlap_samples.count() + overlap_samples.count()
        else:
            self.size = overlap_samples.count()

        _, one_data = overlap_samples.first()
        self.y_shape = (1,)
        self.x_shape = one_data.features.shape

        self.x = np.zeros((self.size, *self.x_shape))
        self.y = np.zeros((self.size, *self.y_shape))
        index = 0
        self._overlap_keys = []
        self._non_overlap_keys = []

        for k, inst in overlap_samples.collect():
            self._overlap_keys.append(k)
            self.x[index] = inst.features
            if guest_side:
                self.y[index] = inst.label
            index += 1

        if self.guest_side:
            for k, inst in non_overlap_samples.collect():
                self._non_overlap_keys.append(k)
                self.x[index] = inst.features
                if guest_side:
                    self.y[index] = inst.label
                index += 1

        if guest_side:
            self._overlap_index = np.array(list(range(0, overlap_samples.count())))
            self._non_overlap_index = np.array(list(range(overlap_samples.count(), self.size)))
        else:
            self._overlap_index = list(range(len(self.x)))

    def get_overlap_indexes(self):
        return self._overlap_index

    def get_non_overlap_indexes(self):
        return self._non_overlap_index

    def get_batch_indexes(self, batch_index):
        start = self.batch_size * batch_index
        end = self.batch_size * (batch_index + 1)
        return start, end

    def get_relative_overlap_index(self, batch_index):
        start, end = self.get_batch_indexes(batch_index)
        return self._overlap_index[(self._overlap_index >= start) & (self._overlap_index < end)] % self.batch_size

    def get_overlap_x(self):
        return self.x[self._overlap_index]

    def get_overlap_y(self):
        return self.y[self._overlap_index]

    def get_overlap_keys(self):
        return self._overlap_keys

    def get_non_overlap_keys(self):
        return self._non_overlap_keys

    def __getitem__(self, index):
        start, end = self.get_batch_indexes(index)
        if self.guest_side:
            return self.x[start: end], self.y[start: end]
        else:
            return self.x[start: end]

    def __len__(self):
        return int(np.ceil(self.size / float(self.batch_size)))

    def get_idx(self):
        return self._keys

    def data_basic_info(self):
        return 'total sample num is {}, overlap sample num is {}, non_overlap sample is {},'\
            'x_shape is {}'.format(self.size, len(self._overlap_index), len(self._non_overlap_index),
                                   self.x_shape)
