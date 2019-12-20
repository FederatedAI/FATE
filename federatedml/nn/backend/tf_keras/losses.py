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

from tensorflow.keras.losses import *
from tensorflow.python.keras import backend as K
from tensorflow.python.framework import ops
from tensorflow.python.util.tf_export import keras_export


@keras_export('keras.losses.keep_predict_loss')
def keep_predict_loss(y_true, y_pred):
    y_pred = ops.convert_to_tensor(y_pred)
    return K.sum(y_true * y_pred)


