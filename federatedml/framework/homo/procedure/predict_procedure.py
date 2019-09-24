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

from federatedml.framework.homo.sync import predict_sync


class Host(predict_sync.Host):

    def register_predict_sync(self, transfer_variables, lr_model):
        self._register_predict_sync(transfer_variables.predict_wx,
                                    transfer_variables.aggregated_model,
                                    transfer_variables.predict_result)
        self._register_func(lr_model.compute_wx)


class Arbiter(predict_sync.Arbiter):

    def register_predict_sync(self, transfer_variables):
        self._register_predict_sync(transfer_variables.predict_wx,
                                    transfer_variables.aggregated_model,
                                    transfer_variables.predict_result)


class Guest(predict_sync.Guest):

    def register_predict_sync(self, transfer_variables, lr_model):
        self._register_predict_sync(transfer_variables.predict_wx,
                                    transfer_variables.aggregated_model,
                                    transfer_variables.predict_result)
        self._register_func(lr_model.compute_wx)
