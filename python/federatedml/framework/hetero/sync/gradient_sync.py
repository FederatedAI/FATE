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

import abc


class Guest(object):
    @abc.abstractclassmethod
    def compute_intermediate(self, *args, **kwargs):
        raise NotImplementedError("This method should be be called here")

    @abc.abstractclassmethod
    def aggregate_host_result(self, *args, **kwargs):
        raise NotImplementedError("This method should be be called here")

    def compute_gradient_procedure(self, *args, **kwargs):
        raise NotImplementedError("This method should be be called here")


class Host(object):
    @abc.abstractclassmethod
    def compute_intermediate(self, *args, **kwargs):
        raise NotImplementedError("This method should be be called here")

    def compute_gradient_procedure(self, *args, **kwargs):
        raise NotImplementedError("This method should be be called here")


class Arbiter(object):
    def compute_gradient_procedure(self, *args, **kwargs):
        raise NotImplementedError("This method should be be called here")


