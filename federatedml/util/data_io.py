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
# from federatedml.feature import ImputerProcess
from federatedml.util import consts
from federatedml.util import DataIOParamChecker
from arch.api import eggroll
from arch.api.proto.data_transform_pb2 import DataTransform
from arch.api.model_manager import core
from arch.api.io import feature

LOGGER = log_utils.getLogger()


# =============================================================================
# DenseFeatureReader
# =============================================================================


class DenseFeatureReader(object):
    def __init__(self, data_io_param):
        DataIOParamChecker.check_param(data_io_param)  
        self.delimitor = data_io_param.delimitor
        self.data_type = data_io_param.data_type
        self.missing_fill = data_io_param.missing_fill
        self.default_value = data_io_param.default_value
        self.missing_fill_method = data_io_param.missing_fill_method
        self.missing_impute = data_io_param.missing_impute
        self.outlier_replace = data_io_param.outlier_replace
        self.outlier_replace_method = data_io_param.outlier_replace_method
        self.outlier_impute = data_io_param.outlier_impute
        self.outlier_replace_value = data_io_param.outlier_replace_value
        self.with_label = data_io_param.with_label
        self.label_idx = data_io_param.label_idx
        self.label_type = data_io_param.label_type
        self.output_format = data_io_param.output_format

    def generate_header(self, input_data_feature, mode="fit"):
        if mode == "fit":
            header = load_data_header("meta")
        else:
            header = load_data_header("header")[0]

        if not header:
            feature = get_one_line(input_data_feature)[1]
            feature_name = ["fid" + str(i) for i in range(len(feature))]
            header = dict(zip(feature_name, range(len(feature))))
            # header = [i for i in range(len(feature))]
        # else:
        #    header = header.split(self.delimiter, -1)
        
        if self.with_label:
            label_idx = self.label_idx
        else:
            label_idx = None

        out_header_dict = {"features": header,
                           "label": label_idx}
        # out_header_dict = {"features": dict(zip(header, range(len(header)))),
        #                    "label": label_idx}

        save_data_header(out_header_dict)
        
        return header

    def read_data(self, table_name, namespace, mode="fit"):
        # input_data = eggroll.table(table_name, namespace)
        input_data = feature.get_feature_data_table(table_name)
        LOGGER.debug("input data init is {}".format(list(input_data.collect())))
        LOGGER.info("start to read dense data and change data to instance")
        input_data_features = None
        input_data_labels = None

        if self.with_label:
            if type(self.label_idx).__name__ != "int":
                raise ValueError("label index should be integer")

            input_data_features = input_data.mapValues(lambda value: value.split(self.delimitor, -1)[:self.label_idx] + value.split(self.delimitor, -1)[self.label_idx + 1 :])
            input_data_labels = input_data.mapValues(lambda value: value.split(self.delimitor, -1)[self.label_idx])

        else:
            input_data_features = input_data.mapValues(lambda value: value.split(self.delimitor, -1))

        # input_data = input_data.mapValues(lambda value: value.split(self.delimitor, -1))
        if mode == "transform":
            self.missing_fill, self.missing_fill_method, \
                self.missing_impute, self.default_value, \
                self.outlier_replace, self.outlier_replace_method, \
                self.outlier_impute, self.outlier_replace_value = \
                        load_data_transform_result()

        missing_fill_value = None
        outlier_replace_value = None

        if self.missing_fill:
            from federatedml.feature.imputer import ImputerProcess
            imputer_processor = ImputerProcess(self.missing_impute)
            LOGGER.info("missing_replace_method is {}".format(self.missing_fill_method))
            if mode == "fit":
                input_data_features, missing_fill_value = imputer_processor.fit(input_data_features, 
                                                                       replace_method=self.missing_fill_method,
                                                                       replace_value=self.default_value)
                if self.missing_impute is None:
                    self.missing_impute = imputer_processor.get_imputer_value_list()
            else:
                LOGGER.debug("type method is {}".format(type(self.missing_fill_method).__name__))
                LOGGER.debug("transform value is {}".format(self.default_value))
                input_data_features = imputer_processor.transform(input_data_features, 
                                                                  replace_method=self.missing_fill_method,
                                                         transform_value=self.default_value)

            if self.missing_impute is None:
                self.missing_impute = imputer_processor.get_imputer_value_list()

        if self.outlier_replace:
            imputer_processor = ImputerProcess(self.outlier_impute)
            if mode == "fit":
                input_data_features, outlier_replace_value = imputer_processor.fit(input_data_features,
                                                                                   replace_method=self.outlier_replace_method,
                                                                                   replace_value=self.outlier_replace_value)
 
                if self.outlier_impute is None:
                    self.outlier_impute = imputer_processor.get_imputer_value_list()
            else:
                LOGGER.info("replace method is {}".format(self.outlier_replace_method))
                input_data_features= imputer_processor.transform(input_data_features,
                                                                  replace_method=self.outlier_replace_method,
                                                                  transform_value=self.outlier_replace_value)

        if self.with_label:
            data_instance = input_data_features.join(input_data_labels,
                                                     lambda features, label:
                                                         self.to_instance(features, label))
        else:
            data_instance = input_data_features.mapValues(lambda features: self.to_instance(features))
        
        header = self.generate_header(input_data_features, mode)
        
        if mode == "fit":
            save_data_transform_result(self.missing_fill,
                                       self.missing_fill_method,
                                       self.missing_impute,
                                       missing_fill_value,
                                       self.outlier_replace,
                                       self.outlier_replace_method,
                                       self.outlier_impute, 
                                       outlier_replace_value,
                                       header=header)

        LOGGER.debug("input data is {}".format(list(input_data_features.collect())))
        
        return data_instance

    def to_instance(self, features, label=None):
        if self.with_label:
            if self.label_type == 'int':
                label = int(label)
            elif self.label_type in ["float", "float64"]:
                label = float(label)

            features = DenseFeatureReader.gen_output_format(features, self.data_type, self.output_format, missing_impute=self.missing_impute)
        
        else:
            features = DenseFeatureReader.gen_output_format(features, self.data_type, self.output_format, missing_impute=self.missing_impute)
        
        return Instance(inst_id=None,
                        features=features,
                        label=label)

    @staticmethod
    def gen_output_format(features, data_type='float', output_format='dense', missing_impute=None):

        if output_format not in ["dense", "sparse"]:
            raise ValueError("output format {} is not define".format(output_format))

        if output_format == "dense":
            return np.asarray(features, dtype=data_type)

        indices = []
        data = []
        column_shape = len(features)
        non_zero = 0

        for i in range(column_shape):
            if (missing_impute is not None and features[i] in missing_impute) or \
               (missing_impute is None and features[i] in ['', 'NULL', 'null', "NA"]):
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
                indices.append(i)
                data.append(features[i])

        return SparseVector(indices, data, column_shape)


# =============================================================================
# SparseFeatureReader: mainly for libsvm input format
# =============================================================================
class SparseFeatureReader(object):
    def __init__(self, data_io_param):
        DataIOParamChecker.check_param(data_io_param)  
        self.delimitor = data_io_param.delimitor
        self.data_type = data_io_param.data_type
        self.label_type = data_io_param.label_type
        self.output_format = data_io_param.output_format

    def get_max_feature_index(self, line, delimitor=' '):
        if line.strip() == '':
            raise ValueError("find an empty line, please check!!!")

        cols = line.split(delimitor, -1)
        if len(cols) <= 1:
            return -1

        return max([int(fid_value.split(":", -1)[0]) for fid_value in cols[1:]])

    def generate_header(self, max_feature):
        label_idx = 0
        features = dict(zip(range(max_feature + 1), range(max_feature + 1)))

        out_header_dict = {"features": features,
                           "label": label_idx}

        save_data_header(out_header_dict)

    def read_data(self, table_name, namespace, mode="fit"):
        # input_data = eggroll.table(table_name, namespace)
        input_data = feature.get_feature_data_table(table_name)
        LOGGER.info("start to read sparse data and change data to instance")

        if mode == "transform":
            max_feature = max(load_data_header("header")[0].keys())
        else:
            get_max_fid = functools.partial(self.get_max_feature_index, delimitor=self.delimitor)
            max_feature = input_data.mapValues(get_max_fid).reduce(lambda max_fid1, max_fid2: max(max_fid1, max_fid2))

            if max_feature == -1:
                raise ValueError("no feature value in input data, please check!")

            self.generate_header(max_feature)
            save_data_transform_result()

        params = [self.delimitor, self.data_type,
                  self.label_type,
                  self.output_format, max_feature]

        to_instance_with_param = functools.partial(self.to_instance, params)
        data_instance = input_data.mapValues(to_instance_with_param)

        return data_instance

    @staticmethod
    def to_instance(param_list, value):
        delimitor = param_list[0]
        data_type = param_list[1]
        label_type = param_list[2]
        output_format = param_list[3]
        max_fid = param_list[4]

        if output_format not in ["dense", "sparse"]:
            raise ValueError("output format {} is not define".format(output_format))

        cols = value.split(delimitor, -1)

        label = cols[0]
        if label_type == 'int':
            label = int(label)
        elif label_type in ["float", "float64"]:
            label = float(label)

        fid_value = []
        for i in range(1, len(cols)):
            fid, val = cols[i].split(":", -1)

            fid = int(fid)
            if data_type in ["float", "float64"]:
                val = float(val)
            elif data_type in ["int", "int64"]:
                val = int(val)

            fid_value.append((fid, val))

        if output_format == "dense":
            features = [0 for i in range(max_fid + 1)]
            for fid, val in fid_value:
                features[fid] = val
            
            features = np.asarray(features, dtype=data_type)

        else:
            indices = []
            data = []
            for fid, val in fid_value:
                indices.append(fid)
                data.append(val)
            
            features = SparseVector(indices, data, max_fid + 1)

        return Instance(inst_id=None,
                        features=features,
                        label=label)


# =============================================================================
# SparseTagReader: mainly for tag data
# =============================================================================
class SparseTagReader(object):
    def __init__(self, data_io_param):
        DataIOParamChecker.check_param(data_io_param)  
        self.delimitor = data_io_param.delimitor
        self.data_type = data_io_param.data_type
        self.with_label = data_io_param.with_label
        self.label_type = data_io_param.label_type
        self.output_format = data_io_param.output_format

    def agg_tag(self, kvs, delimitor=' '):
        tags_set = set()
        LOGGER.info("delemitor is {}".format(self.delimitor))
        for key, value in kvs:
            if self.with_label:
                cols = value.split(delimitor, -1)[1 : ]
            else:
                cols = value.split(delimitor, -1)[0 : ]
 
            LOGGER.debug("tags is {}, value is {}".format(cols, value))
            tags_set |= set(cols)

        LOGGER.info("tags set is {}".format(tags_set))
        return tags_set

    def generate_header(self, tags_dict):
        label_idx = None
        if self.with_label:
            label_idx = 0

        out_header_dict = {"features": tags_dict,
                           "label": label_idx}

        save_data_header(out_header_dict)

    def read_data(self, table_name, namespace, mode="fit"):
        # input_data = eggroll.table(table_name, namespace)
        input_data = feature.get_feature_data_table(table_name)
        LOGGER.info("start to read sparse data and change data to instance")
        # LOGGER.info("tag data is {}".format(list(input_data.collect())))
        # LOGGER.info("delemitor is {}".format(self.delimitor))
        if mode == "transform":
            tags_dict = load_data_header("header")[0]
        else:
            tag_aggregator = functools.partial(self.agg_tag, delimitor=self.delimitor)
            tags_set_list = list(input_data.mapPartitions(tag_aggregator).collect())
            tags_set = set()
            for _, _tags_set in tags_set_list:
                tags_set |= _tags_set
            tags = list(tags_set)

            if len(tags) == 0:
                raise ValueError("no tags in input data, please check!")

            tags = sorted(tags)
            tags_dict = dict(zip(tags, range(len(tags))))

            self.generate_header(tags_dict)
            
            save_data_transform_result()

        params = [self.delimitor,
                  self.data_type,
                  self.with_label,
                  self.label_type,
                  self.output_format,
                  tags_dict]

        to_instance_with_param = functools.partial(self.to_instance, params)
        data_instance = input_data.mapValues(to_instance_with_param)

        return data_instance

    @staticmethod
    def to_instance(param_list, value):
        delimitor = param_list[0]
        data_type=param_list[1]
        with_label = param_list[2]
        label_type = param_list[3]
        output_format = param_list[4]
        tags_dict = param_list[5]

        if output_format not in ["dense", "sparse"]:
            raise ValueError("output format {} is not define".format(output_format))

        cols = value.split(delimitor, -1)
        start_pos = 0
        label = None

        if with_label:
            start_pos = 1
            label = cols[0]
            if label_type == 'int':
                label = int(label)
            elif label_type in ["float", "float64"]:
                label = float(label)

        if output_format == "dense":
            features = [0 for i in range(len(tags_dict))]
            for tag in cols[start_pos : ]:
                features[tags_dict.get(tag)] = 1
            
            features = np.asarray(features, dtype=data_type)
        else:
            indices = []
            data = []
            for tag in cols[start_pos : ]:
                indices.append(tags_dict.get(tag))
                _data = 1
                if data_type in ["float", "float64"]:
                    _data = float(1)
                data.append(_data)
            
            features = SparseVector(indices, data, len(tags_dict))

        return Instance(inst_id=None,
                        features=features,
                        label=label)


def get_one_line(data_instance):
    return data_instance.collect().__next__()


def load_data_header(mode="meta"):
    if mode == "meta":
        return feature.read_feature_meta()
    else:
        return feature.read_feature_header()


def save_data_header(header):
    feature.save_feature_header(header["features"], header["label"])
    LOGGER.debug("feature header is : {}".format(feature.read_feature_header()))


def save_data_transform_result(missing_fill=False, 
                               missing_replace_method=None, 
                               missing_impute=None, 
                               missing_fill_value=None, 
                               outlier_replace=False, 
                               outlier_replace_method=None,
                               outlier_impute=None, 
                               outlier_replace_value=None,
                               header=None):

    buf = DataTransform()
    buf.missing_fill = missing_fill
    buf.outlier_replace = outlier_replace
   
    if missing_fill:
        if missing_replace_method:
            buf.missing_replace_method = str(missing_replace_method)
        
        if missing_impute is not None:
            if type(missing_impute).__name__ == "list":
                buf.missing_value.extend(map(str, missing_impute))
            else:
                buf.missing_value.append(str(missing_impute))

        if missing_fill_value is not None:
            missing_value_dict = {}
            if type(missing_fill_value).__name__ == "list":
                feature_value_dict = dict(zip(range(len(missing_fill_value)), map(str, missing_fill_value)))
            else:
                feature_value_dict = dict(zip(range(len(header)), [str(missing_fill_value) for idx in range(len(header))]))

            buf.missing_replace_value.update(feature_value_dict)

    if outlier_replace:
        if outlier_replace_method:
            buf.outlier_replace_method = str(outlier_replace_method)
        
        if outlier_impute is not None:
            if type(outlier_impute).__name__ == "list":
                buf.outlier_value.extend(map(str, outlier_impute))
            else:
                buf.outlier_value.append(str(outlier_impute))

        if outlier_replace_value is not None:
            outlier_value_dict = {}

            if type(outlier_replace_value).__name__ == "list":
                outlier_value_dict = dict(zip(range(len(outlier_replace_value)), map(str, outlier_replace_value)))
            else:
                outlier_value_dict = dict(zip(range(len(header)), [str(outlier_replace_value) for idx in range(len(header))]))

            buf.outlier_replace_value.update(outlier_value_dict)
 
    LOGGER.debug("conf protobuf is {}".format(buf))

    core.save_model("data_transform", buf)


def load_data_transform_result():
    buf = DataTransform()
    core.read_model("data_transform", buf)

    LOGGER.debug("buf.replace_method {}".format(buf.missing_replace_method))
    LOGGER.debug("buf.outlier_replace_method {}".format(buf.outlier_replace_method))
    missing_fill = buf.missing_fill
    missing_replace_method = buf.missing_replace_method
    missing_value = buf.missing_value
    missing_replace_value = buf.missing_replace_value
    outlier_replace = buf.outlier_replace
    outlier_replace_method = buf.outlier_replace_method
    outlier_value = buf.outlier_value
    outlier_replace_value = buf.outlier_replace_value

    if missing_fill:
        if not missing_replace_method:
            missing_replace_method = None

        if not missing_value:
            missing_value = None
        else:
            missing_value = list(missing_value)

        if missing_replace_value:
            missing_replace_value = [kv[1] for kv in sorted(missing_replace_value.items(), key=lambda kv: kv[0])]
        else:
            missing_replace_value = None

    if outlier_replace:
        if not outlier_replace_method:
            outlier_replace_method = None

        if not outlier_value:
            outlier_value = None
        else:
            outlier_value = list(outlier_value)

        if outlier_replace_value:
            outlier_replace_value = [kv[1] for kv in sorted(outlier_replace_value.items(), key=lambda kv: kv[0])]
        else:
            outlier_replace_value = None

    return missing_fill, missing_replace_method, missing_value, \
           missing_replace_value, outlier_replace, outlier_replace_method, \
           outlier_value, outlier_replace_value

