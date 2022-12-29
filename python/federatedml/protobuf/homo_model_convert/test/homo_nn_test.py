#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Copyright 2021 The FATE Authors. All Rights Reserved.
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

import unittest

import os
import tempfile
import torch as t
from collections import OrderedDict
from federatedml.nn.backend.utils.common import get_torch_model_bytes
from federatedml.protobuf.homo_model_convert.homo_model_convert import model_convert, save_converted_model
from federatedml.protobuf.generated.homo_nn_model_meta_pb2 import HomoNNMeta
from federatedml.protobuf.generated.homo_nn_model_param_pb2 import HomoNNParam


class FakeModule(t.nn.Module):

    def __init__(self):
        super(FakeModule, self).__init__()
        self.fc = t.nn.Linear(100, 10)
        self.transformer = t.nn.Transformer()

    def forward(self, x):
        print(self.fc)
        return x


class TestHomoNNConverter(unittest.TestCase):

    def _get_param_meta(self, torch_model):
        param = HomoNNParam()
        meta = HomoNNMeta()
        # save param
        param.model_bytes = get_torch_model_bytes({'model': torch_model.state_dict()})
        return param, meta

    def setUp(self):
        self.param_list = []
        self.meta_list = []
        self.model_list = []
        # generate some pytorch model
        model = t.nn.Sequential(
            t.nn.Linear(10, 10),
            t.nn.ReLU(),
            t.nn.LSTM(input_size=10, hidden_size=10),
            t.nn.Sigmoid()
        )
        self.model_list.append(model)
        param, meta = self._get_param_meta(model)
        self.param_list.append(param)
        self.meta_list.append(meta)

        model = t.nn.Sequential(t.nn.ReLU())
        self.model_list.append(model)
        param, meta = self._get_param_meta(model)
        self.param_list.append(param)
        self.meta_list.append(meta)

        fake_model = FakeModule()
        self.model_list.append(fake_model)
        param, meta = self._get_param_meta(fake_model)
        self.param_list.append(param)
        self.meta_list.append(meta)

    def test_pytorch_converter(self):
        for param, meta, origin_model in zip(self.param_list, self.meta_list, self.model_list):
            target_framework, model = self._do_convert(param, meta)
            self.assertTrue(target_framework == "pytorch")
            self.assertTrue(isinstance(model['model'], OrderedDict))  # state dict
            origin_model.load_state_dict(model['model'])  # can load state dict
            with tempfile.TemporaryDirectory() as d:
                dest = save_converted_model(model, target_framework, d)
                self.assertTrue(os.path.isfile(dest))
                self.assertTrue(dest.endswith(".pth"))

    @staticmethod
    def _do_convert(model_param, model_meta):
        return model_convert(model_contents={
            'HomoNNParam': model_param,
            'HomoNNMeta': model_meta
        },
            module_name='HomoNN')


if __name__ == '__main__':
    unittest.main()
