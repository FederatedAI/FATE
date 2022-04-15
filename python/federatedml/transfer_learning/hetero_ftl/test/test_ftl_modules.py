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

import unittest

import numpy as np
from federatedml.util import consts
from federatedml.nn.homo_nn.nn_model import get_nn_builder
import json
from federatedml.param.ftl_param import FTLParam
from numpy import array
from fate_arch.session import computing_session as session
import pandas as pd
from federatedml.nn.hetero_nn.backend.tf_keras.data_generator import KerasSequenceDataConverter
from federatedml.transfer_learning.hetero_ftl.ftl_guest import FTLGuest
from federatedml.transfer_learning.hetero_ftl.ftl_host import FTLHost
from federatedml.transfer_learning.hetero_ftl.ftl_base import FTL
from federatedml.param.ftl_param import FTLParam
from federatedml.feature.instance import Instance
import json


class TestFTL(unittest.TestCase):

    def setUp(self):
        session.init('test', 0)

    def test_guest_model_init(self):
        model = FTLGuest()
        param = FTLParam(
            nn_define=json.loads('{"class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "batch_input_shape": [null, 32], "dtype": "float32", "units": 64, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 1.0, "seed": 100, "dtype": "float32"}}, "bias_initializer": {"class_name": "Constant", "config": {"value": 0, "dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "keras_version": "2.2.4-tf", "backend": "tensorflow"}')
        )
        param.check()
        model._init_model(param)

        model.initialize_nn(input_shape=100)
        print(model.nn.get_trainable_weights())

    def test_host_model_init(self):

        model = FTLHost()
        param = FTLParam(
            nn_define=json.loads('{"class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "batch_input_shape": [null, 32], "dtype": "float32", "units": 64, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 1.0, "seed": 100, "dtype": "float32"}}, "bias_initializer": {"class_name": "Constant", "config": {"value": 0, "dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "keras_version": "2.2.4-tf", "backend": "tensorflow"}')
        )
        param.check()
        model._init_model(param)

        model.initialize_nn(input_shape=100)
        print(model.nn.get_trainable_weights())

    def test_label_reset(self):

        l = []
        for i in range(100):
            inst = Instance()
            inst.features = np.random.random(20)
            l.append(inst)
            inst.label = -1
        for i in range(100):
            inst = Instance()
            inst.features = np.random.random(20)
            l.append(inst)
            inst.label = 1

        table = session.parallelize(l, partition=4, include_key=False)
        rs = FTL().check_label(table)
        new_label = [i[1].label for i in list(rs.collect())]
        print(new_label)


if __name__ == '__main__':
    unittest.main()
