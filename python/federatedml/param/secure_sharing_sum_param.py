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
from federatedml.param.base_param import BaseParam


class SecureSharingSumParam(BaseParam):
    def __init__(self, need_verify=False, partition=1):
        self.need_verify = need_verify
        self.partition = partition

    def check(self):
        if type(self.need_verify).__name__ not in ['bool']:
            raise ValueError(
                "need_verify should be 'bool'".format(
                    self.need_verify))

        if type(self.partition).__name__ != "int" or self.partition < 1:
            raise ValueError("partition should be an integer large than 0")

