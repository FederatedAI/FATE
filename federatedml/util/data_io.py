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
################################################################################
#
#
################################################################################

import functools
import numpy as np
from arch.api.utils import log_utils

from federatedml.feature.instance import Instance
from federatedml.feature.sparse_vector import SparseVector
from federatedml.util import consts
from arch.api import eggroll

LOGGER = log_utils.getLogger()


# =============================================================================
# DenseFeatureReader
# =============================================================================


class DenseFeatureReader(object):
    def __init__(self, data_io_param):
        self.delimitor = data_io_param.delimitor
        self.data_type = data_io_param.data_type
        self.missing_fill = data_io_param.missing_fill
        self.default_value = data_io_param.default_value
        self.with_label = data_io_param.with_label
        self.label_idx = data_io_param.label_idx
        self.label_type = data_io_param.label_type
        self.output_format = data_io_param.output_format

    def read_data(self, table_name, namespace):
        input_data = eggroll.table(table_name, namespace)
        LOGGER.info("start to read data and change data to instance")

        params = [self.delimitor, self.data_type, self.missing_fill,
                  self.default_value, self.with_label, self.label_idx,
                  self.label_type, self.output_format]

        to_instance_with_param = functools.partial(self.to_instance, params)
        data_instance = input_data.mapValues(to_instance_with_param)

        return data_instance

    @staticmethod
    def to_instance(param_list, value):
        delimitor = param_list[0]
        data_type = param_list[1]
        missing_fill = param_list[2]
        default_value = param_list[3]
        with_label = param_list[4]
        label_idx = param_list[5]
        label_type = param_list[6]
        output_format = param_list[7]

        label = None
        features = None

        cols = value.split(delimitor, -1)
        col_len = len(cols)

        if missing_fill:
            for idx in range(col_len):
                if cols[idx] in ['', 'NULL', 'null', "NA"]:
                    cols[idx] = default_value

        if with_label:
            label = cols[label_idx]

            if label_type == 'int':
                label = int(label)
            elif label_type == 'float64':
                label = float(label)

            if col_len > 1:
                features = cols[: label_idx]
                features.extend(cols[label_idx + 1:])
            features = DenseFeatureReader.gen_output_format(features, data_type, output_format)
        else:
            features = DenseFeatureReader.gen_output_format(cols, data_type, output_format)
        return Instance(inst_id=None,
                        features=features,
                        label=label)

    @staticmethod
    def gen_output_format(features, data_type='float', output_format='dense'):

        if output_format not in ["dense", "sparse"]:
            raise ValueError("output format %s is not define" % (output_format))

        if output_format == "dense":
            return np.asarray(features, dtype=data_type)

        elif output_format == "sparse":
            indices = []
            data = []
            column_shape = len(features)
            non_zero = 0

        for i in range(column_shape):
            if features[i] in ['', 'NULL', 'null', "NA"]:
                continue

            if data_type in ['float', 'float64']:
                if np.fabs(float(features[i])) < consts.FLOAT_ZERO:
                    continue
                indices.append(i)
                data.append(float(features[i]))
                non_zero += 1

            elif data_type in ['int']:
                if int(features[i]) == 0:
                    continue
                indices.append(i)
                data.append(int(features[i]))

            else:
                raise ValueError("data type %s is not define" % (data_type))

        return SparseVector(indices, data, column_shape)


# =============================================================================
# SparseFeatureReader: mainly for libsvm input format
# =============================================================================
class SparseFeatureReader(object):
    def __init__(self):
        pass

    @staticmethod
    def to_instance(kv_pair,
                    delimitor=",",
                    data_type='float64',
                    with_label=None,
                    label_idx=None,
                    label_type='int',
                    output_format="dense"):
        key, value = kv_pair

        label = None
        features = None
        """
        need to implement
        """
