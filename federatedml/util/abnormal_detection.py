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

from federatedml.statistic import data_overview


def empty_table_detection(data_instances):
    num_data = data_instances.count()
    if num_data == 0:
        table_name = data_instances.get_name()
        namespace = data_instances.get_namespace()
        raise ValueError("Count of data_instance is 0, table_name: {}, namespace: {}".format(
            table_name, namespace
        ))


def empty_feature_detection(data_instances):
    is_empty_feature = data_overview.is_empty_feature(data_instances)
    if is_empty_feature:
        table_name = data_instances.get_name()
        namespace = data_instances.get_namespace()
        raise ValueError("Number of features of DTable is 0., table_name: {}, namespace: {}".format(
            table_name, namespace
        ))
