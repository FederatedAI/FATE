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
from fate_flow.manager.tracking import Tracking 
from fate_flow.entity.metric import Metric
from fate_flow.entity.metric import MetricMeta
from federatedml.feature.instance import Instance
from federatedml.feature.sparse_vector import SparseVector
from federatedml.util import consts
from federatedml.util import abnormal_detection
from federatedml.statistic import data_overview
from federatedml.model_base import ModelBase
from arch.api.proto.data_io_meta_pb2 import DataIOMeta
from arch.api.proto.data_io_param_pb2 import DataIOParam
from arch.api.proto.data_io_meta_pb2 import ImputerMeta
from arch.api.proto.data_io_param_pb2 import ImputerParam
from arch.api.proto.data_io_meta_pb2 import OutlierMeta
from arch.api.proto.data_io_param_pb2 import OutlierParam
from arch.api import storage

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
        self.missing_impute_rate = None
        self.outlier_replace_rate = None
        self.header = None
        self.sid_name = None
        self.label_name = None
        self.tracker = None

    def set_tracker(self, tracker):
        self.tracker = tracker

    def generate_header(self, input_data, input_data_feature):
        # data_meta = storage.get_data_table_meta(input_data)
        data_meta = None

        if not data_meta:
            feature_shape = data_overview.get_data_shape(input_data_feature)
            self.header = ["fid" + str(i) for i in range(feature_shape)]
            self.sid_name = "sid"
            
            if self.with_label:
                self.label_name = "label"
        else:
            self.sid_name = data_meta.get("sid")
            if self.with_label:
                self.header = self.header.split(self.delimitor, -1)[: self.label_idx] + \
                              self.header.split(self.delimitor, -1)[self.label_idx + 1:]
                self.label_name = self.header.split(self.delimitor, -1)[self.label_idx]
            else:
                self.header = self.header.split(self.delimitor, -1)

        schema = make_schema(self.header, self.sid_name, self.label_name)
        set_schema(input_data_feature, schema)
        
        return schema

    def read_data(self, input_data, mode="fit"):
        LOGGER.info("start to read dense data and change data to instance")

        abnormal_detection.empty_table_detection(input_data)

        input_data_features = None
        input_data_labels = None

        if self.with_label:
            if type(self.label_idx).__name__ != "int":
                raise ValueError("label index should be integer")

            data_shape = data_overview.get_data_shape(input_data)
            if not data_shape or self.label_idx >= data_shape:
                raise ValueError("input data's value is empty, it does not contain a label")

            input_data_features = input_data.mapValues(
                lambda value: [] if data_shape == 1 else value.split(self.delimitor, -1)[:self.label_idx] + value.split(
                    self.delimitor, -1)[
                                                                                                            self.label_idx + 1:])
            input_data_labels = input_data.mapValues(lambda value: value.split(self.delimitor, -1)[self.label_idx])

        else:
            input_data_features = input_data.mapValues(
                lambda value: [] if not value else value.split(self.delimitor, -1))

        if mode == "fit":
            data_instance = self.fit(input_data, input_data_features, input_data_labels)
        else:
            data_instance = self.transform(input_data_features, input_data_labels)

        return data_instance

    def fit(self, input_data, input_data_features, input_data_labels):
        schema = self.generate_header(input_data, input_data_features)
        input_data_features = self.fill_missing_value(input_data_features, "fit")
        input_data_features = self.replace_outlier_value(input_data_features, "fit")

        data_instance = self.gen_data_instance(input_data_features, input_data_labels)

        set_schema(data_instance, schema)
        
        return data_instance

    def transform(self, input_data_features, input_data_labels):
        schema = make_schema(self.header, self.sid_name, self.label_name)

        set_schema(input_data_features, schema)
        input_data_features = self.fill_missing_value(input_data_features, "transform")
        input_data_features = self.replace_outlier_value(input_data_features, "transform")

        data_instance = self.gen_data_instance(input_data_features, input_data_labels)
        set_schema(data_instance, schema)
        
        return data_instance

    def fill_missing_value(self, input_data_features, mode="fit"):
        if self.missing_fill:
            from federatedml.feature.imputer import Imputer
            imputer_processor = Imputer(self.missing_impute)
            if mode == "fit":
                input_data_features, self.default_value = imputer_processor.fit(input_data_features,
                                                                                replace_method=self.missing_fill_method,
                                                                                replace_value=self.default_value)
                if self.missing_impute is None:
                    self.missing_impute = imputer_processor.get_missing_value_list()
            else:
                input_data_features = imputer_processor.transform(input_data_features,
                                                                  transform_value=self.default_value)

            if self.missing_impute is None:
                self.missing_impute = imputer_processor.get_missing_value_list()

            self.missing_impute_rate = imputer_processor.get_impute_rate(mode)
            # callback("missing_value_ratio",
            #         missing_impute_rate,
            #         self.tracker)

            # callback("missing_value_list",
            #           self.missing_impute,
            #           self.tracker)

        return input_data_features

    def replace_outlier_value(self, input_data_features, mode="fit"):
        if self.outlier_replace:
            from federatedml.feature.imputer import Imputer
            imputer_processor = Imputer(self.outlier_impute)
            if mode == "fit":
                input_data_features, self.outlier_replace_value = \
                    imputer_processor.fit(input_data_features,
                                          replace_method=self.outlier_replace_method,
                                          replace_value=self.outlier_replace_value)

                if self.outlier_impute is None:
                    self.outlier_impute = imputer_processor.get_missing_value_list()
            else:
                input_data_features = imputer_processor.transform(input_data_features,
                                                                  transform_value=self.outlier_replace_value)

            self.outlier_replace_rate = imputer_processor.get_impute_rate(mode)
            # callback("outlier_value_ratio",
            #         outlier_replace_rate,
            #         self.tracker)

            # callback("outlier_value_list",
            #          self.outlier_impute,
            #          self.tracker)

        return input_data_features

    def gen_data_instance(self, input_data_features, input_data_labels):
        if self.with_label:
            data_instance = input_data_features.join(input_data_labels,
                                                     lambda features, label:
                                                     self.to_instance(features, label))
        else:
            data_instance = input_data_features.mapValues(lambda features: self.to_instance(features))

        return data_instance

    def to_instance(self, features, label=None):
        if self.with_label:
            if self.label_type == 'int':
                label = int(label)
            elif self.label_type in ["float", "float64"]:
                label = float(label)

            features = DenseFeatureReader.gen_output_format(features, self.data_type, self.output_format,
                                                            missing_impute=self.missing_impute)

        else:
            features = DenseFeatureReader.gen_output_format(features, self.data_type, self.output_format,
                                                            missing_impute=self.missing_impute)

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

    def save_model(self):

        dataio_meta, dataio_param = save_data_io_model(input_format="dense",
                                                       delimitor=self.delimitor,
                                                       data_type=self.data_type,
                                                       with_label=self.with_label,
                                                       label_idx=self.label_idx,
                                                       label_type=self.label_type,
                                                       output_format=self.output_format,
                                                       header=self.header,
                                                       sid_name=self.sid_name,
                                                       label_name=self.label_name,
                                                       model_name="DenseFeatureReader")


        missing_imputer_meta, missing_imputer_param = save_missing_imputer_model(self.missing_fill,
                                                                                 self.missing_fill_method,
                                                                                 self.missing_impute,
                                                                                 self.default_value,
                                                                                 self.missing_impute_rate,
                                                                                 self.header,
                                                                                 "Imputer")

        dataio_meta.imputer_meta.CopyFrom(missing_imputer_meta)
        dataio_param.imputer_param.CopyFrom(missing_imputer_param)

        outlier_meta, outlier_param = save_outlier_model(self.outlier_replace,
                                                         self.outlier_replace_method,
                                                         self.outlier_impute,
                                                         self.outlier_replace_value,
                                                         self.outlier_replace_rate,
                                                         self.header,
                                                         "Outlier")

        dataio_meta.outlier_meta.CopyFrom(outlier_meta)
        dataio_param.outlier_param.CopyFrom(outlier_param)

        return {"DataIOMeta": dataio_meta,
                "DataIOParam": dataio_param
                }

    def load_model(self, model_meta, model_param):
        self.delimitor, self.data_type, _1, _2, self.with_label, \
        self.label_idx, self.label_type, self.output_format, self.header, self.sid_name, self.label_name = load_data_io_model("DenseFeatureReader",
                                                                                              model_meta,
                                                                                              model_param)

        self.missing_fill, self.missing_fill_method, \
        self.missing_impute, self.default_value = load_missing_imputer_model(self.header,
                                                                             "Imputer",
                                                                             model_meta.imputer_meta,
                                                                             model_param.imputer_param)

        self.outlier_replace, self.outlier_replace_method, \
        self.outlier_impute, self.outlier_replace_value = load_outlier_model(self.header,
                                                                             "Outlier",
                                                                             model_meta.outlier_meta,
                                                                             model_param.outlier_param)


# =============================================================================
# SparseFeatureReader: mainly for libsvm input format
# =============================================================================
class SparseFeatureReader(object):
    def __init__(self, data_io_param):
        self.delimitor = data_io_param.delimitor
        self.data_type = data_io_param.data_type
        self.label_type = data_io_param.label_type
        self.output_format = data_io_param.output_format
        self.header = None
        self.sid_name = "sid"
        self.label_name = "label"

    def get_max_feature_index(self, line, delimitor=' '):
        if line.strip() == '':
            raise ValueError("find an empty line, please check!!!")

        cols = line.split(delimitor, -1)
        if len(cols) <= 1:
            return -1

        return max([int(fid_value.split(":", -1)[0]) for fid_value in cols[1:]])

    def generate_header(self, max_feature):
        self.header = [str(i) for i in range(max_feature + 1)]

    def read_data(self, input_data, mode="fit"):
        LOGGER.info("start to read sparse data and change data to instance")

        abnormal_detection.empty_table_detection(input_data)

        if not data_overview.get_data_shape(input_data):
            raise ValueError("input data's value is empty, it does not contain a label")

        if mode == "fit":
            data_instance = self.fit(input_data)
        else:
            data_instance = self.transform(input_data)

        schema = make_schema(self.header, self.sid_name, self.label_name)
        set_schema(data_instance, schema)
        return data_instance

    def fit(self, input_data):
        get_max_fid = functools.partial(self.get_max_feature_index, delimitor=self.delimitor)
        max_feature = input_data.mapValues(get_max_fid).reduce(lambda max_fid1, max_fid2: max(max_fid1, max_fid2))

        if max_feature == -1:
            raise ValueError("no feature value in input data, please check!")

        self.generate_header(max_feature)

        data_instance = self.gen_data_instance(input_data, max_feature)
        return data_instance

    def transform(self, input_data):
        max_feature = len(self.header)

        data_instance = self.gen_data_instance(input_data, max_feature)
        return data_instance

    def gen_data_instance(self, input_data, max_feature):
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

    def save_model(self):

        dataio_meta, dataio_param = save_data_io_model(input_format="sparse",
                                                       delimitor=self.delimitor,
                                                       data_type=self.data_type,
                                                       label_type=self.label_type,
                                                       output_format=self.output_format,
                                                       header=self.header,
                                                       sid_name=self.sid_name,
                                                       label_name=self.label_name,
                                                       model_name="SparseFeatureReader")

        missing_imputer_meta, missing_imputer_param = save_missing_imputer_model(missing_fill=False,
                                                                                 model_name="Imputer")
        dataio_meta.imputer_meta.CopyFrom(missing_imputer_meta)
        dataio_param.imputer_param.CopyFrom(missing_imputer_param)

        outlier_meta, outlier_param = save_outlier_model(outlier_replace=False,
                                                         model_name="Outlier")

        dataio_meta.outlier_meta.CopyFrom(outlier_meta)
        dataio_param.outlier_param.CopyFrom(outlier_param)

        return {"DataIOMeta": dataio_meta,
                "DataIOParam": dataio_param
                }

    def load_model(self, model_meta, model_param):
        self.delimitor, self.data_type, _1, _2, _3, _4, \
        self.label_type, self.output_format, self.header, self.sid_name, self.label_name = load_data_io_model("SparseFeatureReader",
                                                                              model_meta,
                                                                              model_param)


# =============================================================================
# SparseTagReader: mainly for tag data
# =============================================================================
class SparseTagReader(object):
    def __init__(self, data_io_param):
        self.delimitor = data_io_param.delimitor
        self.data_type = data_io_param.data_type
        self.tag_with_value = data_io_param.tag_with_value
        self.tag_value_delimitor = data_io_param.tag_value_delimitor
        self.with_label = data_io_param.with_label
        self.label_type = data_io_param.label_type
        self.output_format = data_io_param.output_format
        self.header = None
        self.sid_name = "sid"
        self.label_name = None

    @staticmethod
    def agg_tag(kvs, delimitor=' ', with_label=True, tag_with_value=False, tag_value_delimitor=":"):
        tags_set = set()
        for key, value in kvs:
            if with_label:
                cols = value.split(delimitor, -1)[1:]
            else:
                cols = value.split(delimitor, -1)[0:]

            if tag_with_value is False:
                tags = cols
            else:
                tags = [fea_value.split(tag_value_delimitor, -1)[0] for fea_value in cols]

            tags_set |= set(tags)

        return tags_set

    def generate_header(self, tags):
        self.header = tags

    def read_data(self, input_data, mode="fit"):
        LOGGER.info("start to read sparse data and change data to instance")

        abnormal_detection.empty_table_detection(input_data)

        if mode == "fit":
            data_instance = self.fit(input_data)
            if self.with_label:
                self.label_name = "label"
        else:
            data_instance = self.transform(input_data)

        schema = make_schema(self.header, self.sid_name, self.label_name)
        set_schema(data_instance, schema)
        return data_instance

    def fit(self, input_data):
        tag_aggregator = functools.partial(SparseTagReader.agg_tag,
                                           delimitor=self.delimitor,
                                           with_label=self.with_label,
                                           tag_with_value=self.tag_with_value,
                                           tag_value_delimitor=self.tag_value_delimitor)
        tags_set_list = list(input_data.mapPartitions(tag_aggregator).collect())
        tags_set = set()
        for _, _tags_set in tags_set_list:
            tags_set |= _tags_set
        tags = list(tags_set)

        tags = sorted(tags)
        tags_dict = dict(zip(tags, range(len(tags))))

        self.generate_header(tags)

        data_instance = self.gen_data_instance(input_data, tags_dict)
        return data_instance

    def transform(self, input_data):
        tags_dict = dict(zip(self.header, range(len(self.header))))

        data_instance = self.gen_data_instance(input_data, tags_dict)
        return data_instance

    def gen_data_instance(self, input_data, tags_dict):
        params = [self.delimitor,
                  self.data_type,
                  self.tag_with_value,
                  self.tag_value_delimitor,
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
        data_type = param_list[1]
        tag_with_value = param_list[2]
        tag_value_delimitor = param_list[3]
        with_label = param_list[4]
        label_type = param_list[5]
        output_format = param_list[6]
        tags_dict = param_list[7]

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
            for fea in cols[start_pos:]:
                if tag_with_value:
                    _tag, _val = fea.split(tag_value_delimitor, -1)
                    if _tag in tags_dict:
                        features[tags_dict.get(_tag)] = _val
                else:
                    if fea in tags_dict:
                        features[tags_dict.get(fea)] = 1

            features = np.asarray(features, dtype=data_type)
        else:
            indices = []
            data = []
            for fea in cols[start_pos:]:
                if tag_with_value:
                    _tag, _val = fea.split(tag_value_delimitor, -1)
                else:
                    _tag = fea
                    _val = 1
                
                if _tag not in tags_dict:
                    continue
                
                indices.append(tags_dict.get(_tag))
                if data_type in ["float", "float64"]:
                    _val = float(_val)
                elif data_type in ["int", "int64", "long"]:
                    _val = int(_val)
                elif data_type == "str":
                    _val = str(_val)

                data.append(_val)

            features = SparseVector(indices, data, len(tags_dict))

        return Instance(inst_id=None,
                        features=features,
                        label=label)

    def save_model(self):
        dataio_meta, dataio_param = save_data_io_model(input_format="tag",
                                                       delimitor=self.delimitor,
                                                       data_type=self.data_type,
                                                       tag_with_value=self.tag_with_value,
                                                       tag_value_delimitor=self.tag_value_delimitor,
                                                       with_label=self.with_label,
                                                       label_type=self.label_type,
                                                       output_format=self.output_format,
                                                       header=self.header,
                                                       sid_name=self.sid_name,
                                                       label_name=self.label_name,
                                                       model_name="Reader")

        missing_imputer_meta, missing_imputer_param = save_missing_imputer_model(missing_fill=False,
                                                                                 model_name="Imputer")

        dataio_meta.imputer_meta.CopyFrom(missing_imputer_meta)
        dataio_param.imputer_param.CopyFrom(missing_imputer_param)

        outlier_meta, outlier_param = save_outlier_model(outlier_replace=False,
                                                         model_name="Outlier")

        dataio_meta.outlier_meta.CopyFrom(outlier_meta)
        dataio_param.outlier_param.CopyFrom(outlier_param)

        return {"DataIOMeta": dataio_meta,
                "DataIOParam": dataio_param
                }

    def load_model(self, model_meta, model_param):
        self.delimitor, self.data_type, self.tag_with_value, self.tag_value_delimitor, self.with_label, \
        _1, self.label_type, self.output_format, self.header, self.sid_name, self.label_name = load_data_io_model("SparseTagReader",
                                                                                  model_meta,
                                                                                  model_param)


class DataIO(ModelBase):
    def __init__(self):
        super(DataIO, self).__init__()
        self.reader = None
        from federatedml.param.dataio_param import DataIOParam
        self.model_param = DataIOParam()

    def _init_model(self, model_param):
        print ("model_param is {}".format(model_param))
        if model_param.input_format == "dense":
            self.reader = DenseFeatureReader(self.model_param)
            self.reader.set_tracker(self.tracker)
        elif model_param.input_format == "sparse":
            self.reader = SparseFeatureReader(self.model_param)
        elif model_param.input_format == "tag":
            self.reader = SparseTagReader(self.model_param)

        self.model_param = model_param

    def _load_model(self, model_dict):
        input_model_param = None
        input_model_meta = None
        for _, value in model_dict["model"].items():
            for model in value:
                if model.endswith("Meta"):
                    input_model_meta = value[model]
                if model.endswith("Param"):
                    input_model_param = value[model]

        if input_model_meta.input_format == "dense":
            self.reader = DenseFeatureReader(self.model_param)
            self.reader.set_tracker(self.tracker)
        elif input_model_meta.input_format == "sparse":
            self.reader = SparseFeatureReader(self.model_param)
        elif input_model_meta.input_format == "tag":
            self.reader = SparseTagReader(self.model_param)

        self.reader.load_model(input_model_meta, input_model_param)

    def fit(self, data_inst):
        return self.reader.read_data(data_inst, "fit")
    def transform(self, data_inst):
        return self.reader.read_data(data_inst, "transform")

    def export_model(self):
        model_dict = self.reader.save_model()
        model_dict["DataIOMeta"].need_run = self.need_run
        return model_dict
        # return self.reader.save_model()

    """
    def run(self, component_parameters, args=None):
        self._init_runtime_parameters(component_parameters)

        stage = None
        if "model" in args:
            self._load_model(args["model"])
            stage = "transform"

        if args["data"] is None:
            return

        self._run_data(stage)
    """

def make_schema(header=None, sid_name=None, label_name=None):
    schema = {}
    if header:
        schema["header"] = header

    if sid_name:
        schema["sid_name"] = sid_name

    if label_name:
        schema["label_name"] = label_name

    return schema


def set_schema(data_instance, schema):
    data_instance.schema = schema


def save_data_io_model(input_format="dense",
                       delimitor=",",
                       data_type="str",
                       tag_with_value=False,
                       tag_value_delimitor=":",
                       with_label=False,
                       label_idx=0,
                       label_type="int",
                       output_format="dense",
                       header=None,
                       sid_name=None,
                       label_name=None,
                       model_name="DataIO"):
    model_meta = DataIOMeta()
    model_param = DataIOParam()

    model_meta.input_format = input_format
    model_meta.delimitor = delimitor
    model_meta.data_type = data_type
    model_meta.tag_with_value = tag_with_value
    model_meta.tag_value_delimitor = tag_value_delimitor
    model_meta.with_label = with_label
    model_meta.label_idx = label_idx
    model_meta.label_type = label_type
    model_meta.output_format = output_format

    if header is not None:
        model_param.header.extend(header)
        
        if sid_name:
            model_param.sid_name = sid_name

        if label_name:
            model_param.label_name = label_name

    return model_meta, model_param


def load_data_io_model(model_name="DataIO",
                       model_meta=None,
                       model_param=None):
    delimitor = model_meta.delimitor
    data_type = model_meta.data_type
    tag_with_value = model_meta.tag_with_value
    tag_value_delimitor = model_meta.tag_value_delimitor
    with_label = model_meta.with_label
    label_idx = model_meta.label_idx
    label_type = model_meta.label_type
    output_format = model_meta.output_format

    header = list(model_param.header)
    sid_name = None
    if model_param.sid_name:
        sid_name = model_param.sid_name
    
    label_name = None
    if model_param.label_name:
        label_name = model_param.label_name

    return delimitor, data_type, tag_with_value, tag_value_delimitor, with_label, label_idx, label_type, output_format, header, sid_name, label_name


def save_missing_imputer_model(missing_fill=False,
                               missing_replace_method=None,
                               missing_impute=None,
                               missing_fill_value=None,
                               missing_replace_rate=None,
                               header=None,
                               model_name="Imputer"):
    model_meta = ImputerMeta()
    model_param = ImputerParam()

    model_meta.is_imputer = missing_fill
    if missing_fill:
        if missing_replace_method:
            model_meta.strategy = str(missing_replace_method)

        if missing_impute is not None:
            if missing_impute is not None:
                model_meta.missing_value.extend(map(str, missing_impute))

        if missing_fill_value is not None:
            feature_value_dict = dict(zip(header, map(str, missing_fill_value)))
            model_param.missing_replace_value.update(feature_value_dict)

        if missing_replace_rate is not None:
            missing_replace_rate_dict = dict(zip(header, missing_replace_rate))
            model_param.missing_value_ratio.update(missing_replace_rate_dict)

    return model_meta, model_param


def load_missing_imputer_model(header=None,
                               model_name="Imputer",
                               model_meta=None,
                               model_param=None):
    missing_fill = model_meta.is_imputer
    missing_replace_method = model_meta.strategy
    missing_value = model_meta.missing_value
    missing_fill_value = model_param.missing_replace_value

    if missing_fill:
        if not missing_replace_method:
            missing_replace_method = None

        if not missing_value:
            missing_value = None
        else:
            missing_value = list(missing_value)

        if missing_fill_value:
            missing_fill_value = [missing_fill_value.get(head) for head in header]
        else:
            missing_fill_value = None
    else:
        missing_replace_method = None
        missing_value = None
        missing_fill_value = None

    return missing_fill, missing_replace_method, missing_value, missing_fill_value


def save_outlier_model(outlier_replace=False,
                       outlier_replace_method=None,
                       outlier_impute=None,
                       outlier_replace_value=None,
                       outlier_replace_rate=None,
                       header=None,
                       model_name="Outlier"):
    model_meta = OutlierMeta()
    model_param = OutlierParam()

    model_meta.is_outlier = outlier_replace
    if outlier_replace:
        if outlier_replace_method:
            model_meta.strategy = str(outlier_replace_method)

        if outlier_impute:
            model_meta.outlier_value.extend(map(str, outlier_impute))

        if outlier_replace_value:
            outlier_value_dict = dict(zip(header, map(str, outlier_replace_value)))
            model_param.outlier_replace_value.update(outlier_value_dict)

        if outlier_replace_rate:
            outlier_value_ratio_dict = dict(zip(header, outlier_replace_rate))
            model_param.outlier_value_ratio.update(outlier_value_ratio_dict)

    return model_meta, model_param


def load_outlier_model(header=None,
                       model_name="Outlier",
                       model_meta=None,
                       model_param=None):
    outlier_replace = model_meta.is_outlier
    outlier_replace_method = model_meta.strategy
    outlier_value = model_meta.outlier_value
    outlier_replace_value = model_param.outlier_replace_value

    if outlier_replace:
        if not outlier_replace_method:
            outlier_replace_method = None

        if not outlier_value:
            outlier_value = None
        else:
            outlier_value = list(outlier_value)

        if outlier_replace_value:
            outlier_replace_value = [outlier_replace_value.get(head) for head in header]
        else:
            outlier_replace_value = None
    else:
        outlier_replace_method = None
        outlier_value = None
        outlier_replace_value = None

    return outlier_replace, outlier_replace_method, outlier_value, outlier_replace_value


def callback(keyword="missing_impute",
             value_list=None,
             tracker=None):
    # tracker = Tracking("abc", "123")
    metric_type=None
    """
    if keyword.endswith("ratio"):
        metric_list = []
        for i in range(len(value_list)):
            metric_list.append(Metric(i, value_list[i]))

        tracker.log_metric_data(keyword, "DATAIO", metric_list)

        metric_type = "DATAIO_TABLE"
    """
    metric_list = []
    for i in range(len(value_list)):
        metric_list.append(Metric(value_list[i], i))

    tracker.log_metric_data(keyword, "DATAIO", metric_list)
        
    metric_type = "DATAIO_TEXT"

    tracker.set_metric_meta(keyword,
                            "DATAIO",
                            MetricMeta(name=keyword,
                                        metric_type=metric_type))

