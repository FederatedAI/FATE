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
#

import tensorflow as tf
import numpy as np


class KerasSequenceData(tf.keras.utils.Sequence):
    def __init__(self, X, y=None):
        if X.shape[0] == 0:
            raise ValueError("Data is empty!")

        self.X = X

        if y is None:
            self.y = np.zeros(X.shape[0])
        else:
            self.y = y

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.X, self.y


class KerasSequenceDataConverter(object):
    @classmethod
    def convert_data(cls, x=None, y = None):
        return KerasSequenceData(x, y)



