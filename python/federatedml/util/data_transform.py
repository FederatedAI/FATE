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

import copy
import functools

import numpy as np

from federatedml.feature.instance import Instance
from federatedml.feature.sparse_vector import SparseVector
from federatedml.model_base import ModelBase
from federatedml.protobuf.generated.data_transform_meta_pb2 import DataTransformMeta
from federatedml.protobuf.generated.data_transform_meta_pb2 import DataTransformImputerMeta
from federatedml.protobuf.generated.data_transform_meta_pb2 import DataTransformOutlierMeta
from federatedml.protobuf.generated.data_transform_param_pb2 import DataTransformParam
from federatedml.protobuf.generated.data_transform_param_pb2 import DataTransformImputerParam
from federatedml.protobuf.generated.data_transform_param_pb2 import DataTransformOutlierParam
from federatedml.statistic import data_overview
from federatedml.util import abnormal_detection
from federatedml.util import consts
from federatedml.util import LOGGER
from federatedml.util.io_check import assert_io_num_rows_equal
from federatedml.util.data_format_preprocess import DataFormatPreProcess
from federatedml.util.anonymous_generator_util import Anonymous


# =============================================================================
# DenseFeatureTransformer
# =============================================================================
class DenseFeatureTransformer(object):
    def __init__(self, data_transform_param):
        self.delimitor = data_transform_param.delimitor
        self.data_type = data_transform_param.data_type
        self.missing_fill = data_transform_param.missing_fill
        self.default_value = data_transform_param.default_value
        self.missing_fill_method = data_transform_param.missing_fill_method
        self.missing_impute = data_transform_param.missing_impute
        self.outlier_replace = data_transform_param.outlier_replace
        self.outlier_replace_method = data_transform_param.outlier_replace_method
        self.outlier_impute = data_transform_param.outlier_impute
        self.outlier_replace_value = data_transform_param.outlier_replace_value
        self.with_label = data_transform_param.with_label
        self.label_name = data_transform_param.label_name.lower() if self.with_label else None
        self.label_type = data_transform_param.label_type if self.with_label else None
        self.output_format = data_transform_param.output_format
        self.missing_impute_rate = None
        self.outlier_replace_rate = None
        self.header = None
        self.sid_name = None
        self.exclusive_data_type_fid_map = {}
        self.match_id_name = data_transform_param.match_id_name
        self.match_id_index = 0
        self.with_match_id = data_transform_param.with_match_id
        self.anonymous_generator = None
        self.anonymous_header = None

        if data_transform_param.exclusive_data_type:
            self.exclusive_data_type = dict([(k.lower(), v)
                                             for k, v in data_transform_param.exclusive_data_type.items()])
        else:
            self.exclusive_data_type = None

    def _update_param(self, schema):
        meta = schema["meta"]
        self.delimitor = meta.get("delimiter", ",")
        self.data_type = meta.get("data_type")
        self.with_label = meta.get("with_label", False)
        if self.with_label:
            self.label_type = meta.get("label_type", "int")
            self.label_name = meta.get("label_name", '')
        self.with_match_id = meta.get("with_match_id", False)

        if self.with_match_id:
            match_id_name = schema.get("match_id_name", [])
            if not self.match_id_name:
                if isinstance(match_id_name, list):
                    raise ValueError("Multiple Match ID exist, please specified the one to use")
                self.match_id_name = match_id_name
                self.match_id_index = schema["original_index_info"]["match_id_index"][0]
            else:
                try:
                    idx = match_id_name.index(self.match_id_name)
                except ValueError:
                    raise ValueError(f"Can not find {self.match_id_name} in {match_id_name}")
                self.match_id_index = schema["original_index_info"]["match_id_index"][idx]

            schema["match_id_name"] = self.match_id_name

        header = schema["header"]
        exclusive_data_type = meta.get("exclusive_data_type", None)
        if exclusive_data_type:
            self.exclusive_data_type = dict([(k.lower(), v) for k, v in exclusive_data_type.items()])
            for idx, col_name in enumerate(header):
                if col_name in self.exclusive_data_type:
                    self.exclusive_data_type_fid_map[idx] = self.exclusive_data_type[col_name]

    def extract_feature_value(self, value, header_index=None):
        if not header_index:
            return []

        value = value.split(self.delimitor, -1)
        if len(value) <= header_index[-1]:
            raise ValueError("Feature shape is smaller than header shape")

        feature_values = []
        for idx in header_index:
            feature_values.append(value[idx])

        return feature_values

    def read_data(self, input_data, mode="fit"):
        LOGGER.info("start to read dense data and change data to instance")

        abnormal_detection.empty_table_detection(input_data)

        schema = copy.deepcopy(input_data.schema)
        if not schema.get("meta"):
            LOGGER.warning("Data meta is supported to be set with data uploading or binding, "
                           "please refer to data transform using guides.")
            meta = dict(input_format="dense",
                        delimiter=self.delimitor,
                        with_label=self.with_label,
                        label_name=self.label_name,
                        with_match_id=self.with_match_id,
                        data_type=self.data_type,
                        )
            if mode == "transform" and self.with_label \
                    and self.label_name not in schema["header"].split(self.delimitor, -1):
                del meta["label_name"]
                del meta["with_label"]

            schema["meta"] = meta
            generated_header = DataFormatPreProcess.generate_header(input_data, schema)
            schema.update(generated_header)
            schema = self.anonymous_generator.generate_anonymous_header(schema)
            set_schema(input_data, schema)
        else:
            self._update_param(schema)

        header = schema["header"]
        anonymous_header = schema["anonymous_header"]
        training_header = self.header
        if mode == "transform":
            if (set(self.header) & set(header)) != set(self.header):
                raise ValueError(f"Transform Data's header is {header}, expect {self.header}")
            self.header = header
            if not self.anonymous_header:
                self.anonymous_header = anonymous_header
        else:
            self.header = header
            self.anonymous_header = anonymous_header

        header_index = schema["original_index_info"]["header_index"]
        extract_feature_func = functools.partial(self.extract_feature_value,
                                                 header_index=header_index)
        input_data_features = input_data.mapValues(extract_feature_func)
        # input_data_features.schema = input_data.schema
        input_data_features.schema = schema

        input_data_labels = None
        input_data_match_id = None

        if "label_name" in schema:
            label_index = schema["original_index_info"]["label_index"]
            input_data_labels = input_data.mapValues(lambda value: value.split(self.delimitor, -1)[label_index])

        if self.with_match_id:
            input_data_match_id = input_data.mapValues(
                lambda value: value.split(self.delimitor, -1)[self.match_id_index])

        if mode == "fit":
            data_instance = self.fit(input_data, input_data_features, input_data_labels, input_data_match_id)
            set_schema(data_instance, schema)
        else:
            data_instance = self.transform(input_data_features, input_data_labels, input_data_match_id)
            data_instance = data_overview.header_alignment(data_instance, training_header, self.anonymous_header)
            self.header = training_header

        return data_instance

    def fit(self, input_data, input_data_features, input_data_labels, input_data_match_id):
        input_data_features = self.fill_missing_value(input_data_features, "fit")
        input_data_features = self.replace_outlier_value(input_data_features, "fit")

        data_instance = self.gen_data_instance(input_data_features, input_data_labels, input_data_match_id)

        return data_instance

    @assert_io_num_rows_equal
    def transform(self, input_data_features, input_data_labels, input_data_match_id):
        schema = input_data_features.schema
        input_data_features = self.fill_missing_value(input_data_features, "transform")
        input_data_features = self.replace_outlier_value(input_data_features, "transform")

        data_instance = self.gen_data_instance(input_data_features, input_data_labels, input_data_match_id)
        data_instance.schema = schema

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

        return input_data_features

    def gen_data_instance(self, input_data_features, input_data_labels, input_data_match_id):
        if input_data_labels:
            data_instance = input_data_features.join(input_data_labels,
                                                     lambda features, label: self.to_instance(features, label))
        else:
            data_instance = input_data_features.mapValues(lambda features: self.to_instance(features))

        if self.with_match_id:
            data_instance = data_instance.join(input_data_match_id, self.append_match_id)

        return data_instance

    def append_match_id(self, inst, match_id):
        inst.inst_id = match_id
        return inst

    def to_instance(self, features, label=None):
        if self.header is None and len(features) != 0:
            raise ValueError("features shape {} not equal to header shape 0".format(len(features)))
        elif self.header is not None and len(self.header) != len(features):
            raise ValueError("features shape {} not equal to header shape {}".format(len(features), len(self.header)))

        if label is not None:
            if self.label_type == 'int':
                label = int(label)
            elif self.label_type in ["float", "float64"]:
                label = float(label)

            format_features = DenseFeatureTransformer.gen_output_format(features, self.data_type,
                                                                        self.exclusive_data_type_fid_map,
                                                                        self.output_format,
                                                                        missing_impute=self.missing_impute)

        else:
            format_features = DenseFeatureTransformer.gen_output_format(features, self.data_type,
                                                                        self.exclusive_data_type_fid_map,
                                                                        self.output_format,
                                                                        missing_impute=self.missing_impute)

        return Instance(inst_id=None,
                        features=format_features,
                        label=label)

    @staticmethod
    def gen_output_format(features, data_type='float', exclusive_data_type_fid_map=None,
                          output_format='dense', missing_impute=None):

        if output_format not in ["dense", "sparse"]:
            raise ValueError("output format {} is not define".format(output_format))

        missing_impute_dtype_set = {"int", "int64", "long", "float", "float64", "double"}
        missing_impute_value_set = {'', 'NULL', 'null', "NA"}
        type_mapping = dict()
        if output_format == "dense":
            # format_features = copy.deepcopy(features)
            format_features = [None] * len(features)
            for fid in range(len(features)):
                if exclusive_data_type_fid_map is not None and fid in exclusive_data_type_fid_map:
                    dtype = exclusive_data_type_fid_map[fid]
                else:
                    dtype = data_type

                if dtype in missing_impute_dtype_set:
                    if (missing_impute is not None and features[fid] in missing_impute) or \
                            (missing_impute is None and features[fid] in missing_impute_value_set):
                        format_features[fid] = np.nan
                        continue

                format_features[fid] = features[fid]
                if exclusive_data_type_fid_map:
                    if dtype not in type_mapping:
                        np_type = getattr(np, dtype)
                        type_mapping[dtype] = np_type

                    format_features[fid] = type_mapping[dtype](format_features[fid])

            if exclusive_data_type_fid_map:
                return np.asarray(format_features, dtype=object)
            else:
                return np.asarray(format_features, dtype=data_type)

        indices = []
        data = []
        column_shape = len(features)
        non_zero = 0

        for i in range(column_shape):
            if (missing_impute is not None and features[i] in missing_impute) or \
                    (missing_impute is None and features[i] in missing_impute_value_set):
                indices.append(i)
                data.append(np.nan)
                non_zero += 1

            elif data_type in ['float', 'float64', "double"]:
                if np.fabs(float(features[i])) < consts.FLOAT_ZERO:
                    continue

                indices.append(i)
                data.append(float(features[i]))
                non_zero += 1

            elif data_type in ['int', "int64", "long"]:
                if int(features[i]) == 0:
                    continue
                indices.append(i)
                data.append(int(features[i]))

            else:
                indices.append(i)
                data.append(features[i])

        return SparseVector(indices, data, column_shape)

    def get_summary(self):
        if not self.missing_fill and not self.outlier_replace:
            return {}

        summary_buf = {}
        if self.missing_fill:
            missing_summary = dict()
            missing_summary["missing_value"] = list(self.missing_impute)
            missing_summary["missing_impute_value"] = dict(zip(self.header, self.default_value))
            missing_summary["missing_impute_rate"] = dict(zip(self.header, self.missing_impute_rate))
            summary_buf["missing_fill_info"] = missing_summary

        if self.outlier_replace:
            outlier_replace_summary = dict()
            outlier_replace_summary["outlier_value"] = list(self.outlier_impute)
            outlier_replace_summary["outlier_replace_value"] = dict(zip(self.header, self.outlier_replace_value))
            outlier_replace_summary["outlier_replace_rate"] = dict(zip(self.header, self.outlier_replace_rate))
            summary_buf["outlier_replace_rate"] = outlier_replace_summary

        return summary_buf

    def save_model(self):
        transform_meta, transform_param = save_data_transform_model(input_format="dense",
                                                                    delimitor=self.delimitor,
                                                                    data_type=self.data_type,
                                                                    exclusive_data_type=self.exclusive_data_type,
                                                                    with_label=self.with_label,
                                                                    label_type=self.label_type,
                                                                    output_format=self.output_format,
                                                                    header=self.header,
                                                                    sid_name=self.sid_name,
                                                                    label_name=self.label_name,
                                                                    with_match_id=self.with_match_id,
                                                                    model_name="DenseFeatureTransformer",
                                                                    anonymous_header=self.anonymous_header)

        missing_imputer_meta, missing_imputer_param = save_missing_imputer_model(self.missing_fill,
                                                                                 self.missing_fill_method,
                                                                                 self.missing_impute,
                                                                                 self.default_value,
                                                                                 self.missing_impute_rate,
                                                                                 self.header,
                                                                                 "Imputer")

        transform_meta.imputer_meta.CopyFrom(missing_imputer_meta)
        transform_param.imputer_param.CopyFrom(missing_imputer_param)

        outlier_meta, outlier_param = save_outlier_model(self.outlier_replace,
                                                         self.outlier_replace_method,
                                                         self.outlier_impute,
                                                         self.outlier_replace_value,
                                                         self.outlier_replace_rate,
                                                         self.header,
                                                         "Outlier")

        transform_meta.outlier_meta.CopyFrom(outlier_meta)
        transform_param.outlier_param.CopyFrom(outlier_param)

        return {"DataTransformMeta": transform_meta,
                "DataTransformParam": transform_param
                }

    def load_model(self, model_meta, model_param):
        self.delimitor, self.data_type, self.exclusive_data_type, _1, _2, self.with_label, \
            self.label_type, self.output_format, self.header, self.sid_name, self.label_name, self.with_match_id, self.anonymous_header = \
            load_data_transform_model("DenseFeatureTransformer", model_meta, model_param)

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
# SparseFeatureTransformer: mainly for libsvm input format
# =============================================================================
class SparseFeatureTransformer(object):
    def __init__(self, data_transform_param):
        self.delimitor = data_transform_param.delimitor
        self.data_type = data_transform_param.data_type
        self.label_type = data_transform_param.label_type
        self.output_format = data_transform_param.output_format
        self.header = None
        self.sid_name = "sid"
        self.with_match_id = data_transform_param.with_match_id
        self.match_id_name = "match_id" if self.with_match_id else None
        self.match_id_index = data_transform_param.match_id_index
        self.with_label = data_transform_param.with_label
        self.label_name = data_transform_param.label_name.lower() if self.with_label else None
        self.anonymous_generator = None
        self.anonymous_header = None

    def _update_param(self, schema):
        meta = schema["meta"]
        self.delimitor = meta.get("delimiter", ",")
        self.data_type = meta.get("data_type")
        self.with_label = meta.get("with_label", False)
        if self.with_label:
            self.label_type = meta.get("label_type", "int")
            self.label_name = meta.get("label_name", "")
        self.with_match_id = meta.get("with_match_id", False)
        if self.with_match_id:
            match_id_name = schema.get("match_id_name")
            if isinstance(match_id_name, list):
                self.match_id_name = match_id_name[self.match_id_index]
            else:
                self.match_id_name = match_id_name

            schema["match_id_name"] = self.match_id_name

    def read_data(self, input_data, mode="fit"):
        LOGGER.info("start to read sparse data and change data to instance")

        abnormal_detection.empty_table_detection(input_data)

        schema = copy.deepcopy(input_data.schema)
        if not schema.get("meta", {}):
            LOGGER.warning("Data meta is supported to be set with data uploading or binding, "
                           "please refer to data transform using guides.")
            meta = dict(input_format="sparse",
                        delimiter=self.delimitor,
                        with_label=self.with_label,
                        with_match_id=self.with_match_id,
                        data_type=self.data_type)

            schema["meta"] = meta
            generated_header = DataFormatPreProcess.generate_header(input_data, schema)
            schema.update(generated_header)
            schema = self.anonymous_generator.generate_anonymous_header(schema)
            set_schema(input_data, schema)
        else:
            self._update_param(schema)

        if mode == "fit":
            self.header = schema["header"]
            self.anonymous_header = schema["anonymous_header"]
            data_instance = self.fit(input_data)
        else:
            if not self.anonymous_header:
                header_set = set(self.header)
                self.anonymous_header = []
                for column, anonymous_column in zip(schema["header"], schema["anonymous_header"]):
                    if column not in header_set:
                        continue
                    self.anonymous_header.append(anonymous_column)

            schema["header"] = self.header
            schema["anonymous_header"] = self.anonymous_header
            set_schema(input_data, schema)
            data_instance = self.transform(input_data)

        set_schema(data_instance, schema)
        return data_instance

    def fit(self, input_data):
        max_feature = len(self.header)

        if max_feature == 0:
            raise ValueError("no feature value in input data, please check!")

        data_instance = self.gen_data_instance(input_data, max_feature)
        return data_instance

    def transform(self, input_data):
        max_feature = len(self.header)

        data_instance = self.gen_data_instance(input_data, max_feature)
        return data_instance

    def gen_data_instance(self, input_data, max_feature):
        id_range = input_data.schema["meta"].get("id_range", 0)
        params = [self.delimitor, self.data_type,
                  self.label_type, self.with_match_id,
                  self.match_id_index, id_range,
                  self.output_format,
                  self.with_label, max_feature]

        to_instance_with_param = functools.partial(self.to_instance, params)
        data_instance = input_data.mapValues(to_instance_with_param)

        return data_instance

    @staticmethod
    def to_instance(param_list, value):
        delimitor = param_list[0]
        data_type = param_list[1]
        label_type = param_list[2]
        with_match_id = param_list[3]
        match_id_index = param_list[4]
        id_range = param_list[5]
        output_format = param_list[6]
        with_label = param_list[7]
        max_fid = param_list[8]

        if output_format not in ["dense", "sparse"]:
            raise ValueError("output format {} is not define".format(output_format))

        cols = value.split(delimitor, -1)
        offset = 0
        if with_match_id:
            offset = id_range if id_range else 1
            match_id = cols[match_id_index]
        else:
            match_id = None

        label = None
        if with_label:
            label = cols[offset]
            if label_type == 'int':
                label = int(label)
            elif label_type in ["float", "float64"]:
                label = float(label)
            offset += 1

        fid_value = []
        for i in range(offset, len(cols)):
            fid, val = cols[i].split(":", -1)

            fid = int(fid)
            if data_type in ["float", "float64"]:
                val = float(val)
            elif data_type in ["int", "int64"]:
                val = int(val)

            fid_value.append((fid, val))

        if output_format == "dense":
            features = [0 for i in range(max_fid)]
            for fid, val in fid_value:
                features[fid] = val

            features = np.asarray(features, dtype=data_type)

        else:
            indices = []
            data = []
            for fid, val in fid_value:
                indices.append(fid)
                data.append(val)

            features = SparseVector(indices, data, max_fid)

        return Instance(inst_id=match_id,
                        features=features,
                        label=label)

    def save_model(self):

        transform_meta, transform_param = save_data_transform_model(input_format="sparse",
                                                                    delimitor=self.delimitor,
                                                                    data_type=self.data_type,
                                                                    label_type=self.label_type,
                                                                    output_format=self.output_format,
                                                                    header=self.header,
                                                                    sid_name=self.sid_name,
                                                                    label_name=self.label_name,
                                                                    with_match_id=self.with_match_id,
                                                                    with_label=self.with_label,
                                                                    model_name="SparseFeatureTransformer",
                                                                    anonymous_header=self.anonymous_header)

        missing_imputer_meta, missing_imputer_param = save_missing_imputer_model(missing_fill=False,
                                                                                 model_name="Imputer")
        transform_meta.imputer_meta.CopyFrom(missing_imputer_meta)
        transform_param.imputer_param.CopyFrom(missing_imputer_param)

        outlier_meta, outlier_param = save_outlier_model(outlier_replace=False,
                                                         model_name="Outlier")

        transform_meta.outlier_meta.CopyFrom(outlier_meta)
        transform_param.outlier_param.CopyFrom(outlier_param)

        return {"DataTransformMeta": transform_meta,
                "DataTransformParam": transform_param
                }

    def load_model(self, model_meta, model_param):
        self.delimitor, self.data_type, _0, _1, _2, self.with_label, self.label_type, self.output_format, \
            self.header, self.sid_name, self.label_name, self.with_match_id, self.anonymous_header = \
            load_data_transform_model(
                "SparseFeatureTransformer",
                model_meta,
                model_param)


# =============================================================================
# SparseTagTransformer: mainly for tag data
# =============================================================================
class SparseTagTransformer(object):
    def __init__(self, data_transform_param):
        self.delimitor = data_transform_param.delimitor
        self.data_type = data_transform_param.data_type
        self.tag_with_value = data_transform_param.tag_with_value
        self.tag_value_delimitor = data_transform_param.tag_value_delimitor
        self.with_label = data_transform_param.with_label
        self.label_type = data_transform_param.label_type if self.with_label else None
        self.output_format = data_transform_param.output_format
        self.header = None
        self.sid_name = "sid"
        self.label_name = data_transform_param.label_name.lower() if data_transform_param.label_name else None
        self.missing_fill = data_transform_param.missing_fill
        self.missing_fill_method = data_transform_param.missing_fill_method
        self.default_value = data_transform_param.default_value
        self.with_match_id = data_transform_param.with_match_id
        self.match_id_index = data_transform_param.match_id_index
        self.match_id_name = "match_id" if self.with_match_id else None
        self.missing_impute_rate = None
        self.missing_impute = None
        self.anonymous_generator = None
        self.anonymous_header = None

    def _update_param(self, schema):
        meta = schema["meta"]
        self.delimitor = meta.get("delimiter", ",")
        self.data_type = meta.get("data_type")
        self.tag_with_value = meta.get("tag_with_value")
        self.tag_value_delimitor = meta.get("tag_value_delimiter", ":")
        self.with_label = meta.get("with_label", False)
        if self.with_label:
            self.label_type = meta.get("label_type", "int")
            self.label_name = meta.get("label_name")
        self.with_match_id = meta.get("with_match_id", False)
        if self.with_match_id:
            match_id_name = schema.get("match_id_name")
            if isinstance(match_id_name, list):
                self.match_id_name = match_id_name[self.match_id_index]
            else:
                self.match_id_name = match_id_name

            schema["match_id_name"] = self.match_id_name

    def read_data(self, input_data, mode="fit"):
        LOGGER.info("start to read sparse data and change data to instance")

        abnormal_detection.empty_table_detection(input_data)
        schema = copy.deepcopy(input_data.schema)
        if not schema.get("meta", {}):
            LOGGER.warning("Data meta is supported to be set with data uploading or binding, "
                           "please refer to data transform using guides.")
            meta = dict(input_format="tag",
                        delimiter=self.delimitor,
                        with_label=self.with_label,
                        with_match_id=self.with_match_id,
                        tag_with_value=self.tag_with_value,
                        tag_value_delimiter=self.tag_value_delimitor,
                        data_type=self.data_type)
            schema["meta"] = meta
            generated_header = DataFormatPreProcess.generate_header(input_data, schema)
            schema.update(generated_header)
            schema = self.anonymous_generator.generate_anonymous_header(schema)
            set_schema(input_data, schema)
        else:
            self._update_param(schema)

        if mode == "fit":
            self.header = schema["header"]
            self.anonymous_header = schema["anonymous_header"]
            data_instance = self.fit(input_data)
        else:
            if not self.anonymous_header:
                header_set = set(self.header)
                self.anonymous_header = []
                for column, anonymous_column in zip(schema["header"], schema["anonymous_header"]):
                    if column not in header_set:
                        continue
                    self.anonymous_header.append(anonymous_column)

            schema["header"] = self.header
            schema["anonymous_header"] = self.anonymous_header
            set_schema(input_data, schema)
            data_instance = self.transform(input_data)

        set_schema(data_instance, schema)

        return data_instance

    @staticmethod
    def change_tag_to_str(value, tags_dict=None, delimitor=",", feature_offset=0,
                          tag_value_delimitor=":"):
        vals = value.split(delimitor, -1)
        ret = [''] * len(tags_dict)

        vals = vals[feature_offset:]

        for i in range(len(vals)):
            tag, value = vals[i].split(tag_value_delimitor, -1)
            idx = tags_dict.get(tag, None)
            if idx is not None:
                ret[idx] = value

        return ret

    @staticmethod
    def change_str_to_tag(value, tags_dict=None, delimitor=",", tag_value_delimitor=":"):
        ret = [None] * len(tags_dict)
        tags = sorted(list(tags_dict.keys()))
        for i in range(len(value)):
            tag, val = tags[i], value[i]
            ret[i] = tag_value_delimitor.join([tag, val])

        return delimitor.join(ret)

    def fill_missing_value(self, input_data, tags_dict, schema, mode="fit"):
        feature_offset = DataFormatPreProcess.get_feature_offset(schema)
        str_trans_method = functools.partial(self.change_tag_to_str,
                                             tags_dict=tags_dict,
                                             delimitor=self.delimitor,
                                             feature_offset=feature_offset,
                                             tag_value_delimitor=self.tag_value_delimitor)

        input_data = input_data.mapValues(str_trans_method)
        set_schema(input_data, schema)

        from federatedml.feature.imputer import Imputer
        imputer_processor = Imputer()
        if mode == "fit":
            data, self.default_value = imputer_processor.fit(input_data,
                                                             replace_method=self.missing_fill_method,
                                                             replace_value=self.default_value)
            LOGGER.debug("self.default_value is {}".format(self.default_value))
        else:
            data = imputer_processor.transform(input_data,
                                               transform_value=self.default_value)
        if self.missing_impute is None:
            self.missing_impute = imputer_processor.get_missing_value_list()

        LOGGER.debug("self.missing_impute is {}".format(self.missing_impute))

        self.missing_impute_rate = imputer_processor.get_impute_rate(mode)

        str_trans_tag_method = functools.partial(self.change_str_to_tag,
                                                 tags_dict=tags_dict,
                                                 delimitor=self.delimitor,
                                                 tag_value_delimitor=self.tag_value_delimitor)

        data = data.mapValues(str_trans_tag_method)

        return data

    def fit(self, input_data):
        schema = input_data.schema
        tags_dict = dict(zip(schema["header"], range(len(schema["header"]))))
        if self.tag_with_value and self.missing_fill:
            input_data = self.fill_missing_value(input_data, tags_dict, schema, mode="fit")

        data_instance = self.gen_data_instance(input_data, schema["meta"], tags_dict)

        return data_instance

    def transform(self, input_data):
        schema = input_data.schema
        tags_dict = dict(zip(self.header, range(len(self.header))))

        if self.tag_with_value and self.missing_fill:
            input_data = self.fill_missing_value(input_data, tags_dict, schema, mode="transform")

        data_instance = self.gen_data_instance(input_data, schema["meta"], tags_dict)

        return data_instance

    def gen_data_instance(self, input_data, meta, tags_dict):
        params = [self.delimitor,
                  self.data_type,
                  self.tag_with_value,
                  self.tag_value_delimitor,
                  self.with_label,
                  self.with_match_id,
                  self.match_id_index,
                  meta.get("id_range", 0),
                  self.label_type,
                  self.output_format,
                  tags_dict]

        to_instance_with_param = functools.partial(self.to_instance, params)
        data_instance = input_data.mapValues(to_instance_with_param)

        return data_instance

    def get_summary(self):
        if not self.missing_fill:
            return {}

        missing_summary = dict()
        missing_summary["missing_value"] = list(self.missing_impute)
        missing_summary["missing_impute_value"] = dict(zip(self.header, self.default_value))
        missing_summary["missing_impute_rate"] = dict(zip(self.header, self.missing_impute_rate))
        summary_buf = {"missing_fill_info": missing_summary}
        return summary_buf

    @staticmethod
    def to_instance(param_list, value):
        delimitor = param_list[0]
        data_type = param_list[1]
        tag_with_value = param_list[2]
        tag_value_delimitor = param_list[3]
        with_label = param_list[4]
        with_match_id = param_list[5]
        match_id_index = param_list[6]
        id_range = param_list[7]
        label_type = param_list[8]
        output_format = param_list[9]
        tags_dict = param_list[10]

        if output_format not in ["dense", "sparse"]:
            raise ValueError("output format {} is not define".format(output_format))

        cols = value.split(delimitor, -1)
        offset = 0
        label = None
        match_id = None

        if with_match_id:
            offset = id_range if id_range else 1
            if offset == 0:
                offset = 1
            match_id = cols[match_id_index]

        if with_label:
            label = cols[offset]
            offset += 1
            if label_type == 'int':
                label = int(label)
            elif label_type in ["float", "float64"]:
                label = float(label)

        if output_format == "dense":
            features = [0 for i in range(len(tags_dict))]
            for fea in cols[offset:]:
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
            for fea in cols[offset:]:
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

        return Instance(inst_id=match_id,
                        features=features,
                        label=label)

    def save_model(self):
        transform_meta, transform_param = save_data_transform_model(input_format="tag",
                                                                    delimitor=self.delimitor,
                                                                    data_type=self.data_type,
                                                                    tag_with_value=self.tag_with_value,
                                                                    tag_value_delimitor=self.tag_value_delimitor,
                                                                    with_label=self.with_label,
                                                                    label_type=self.label_type,
                                                                    with_match_id=self.with_match_id,
                                                                    output_format=self.output_format,
                                                                    header=self.header,
                                                                    sid_name=self.sid_name,
                                                                    label_name=self.label_name,
                                                                    model_name="Transformer",
                                                                    anonymous_header=self.anonymous_header)

        missing_imputer_meta, missing_imputer_param = save_missing_imputer_model(self.missing_fill,
                                                                                 self.missing_fill_method,
                                                                                 self.missing_impute,
                                                                                 self.default_value,
                                                                                 self.missing_impute_rate,
                                                                                 self.header,
                                                                                 "Imputer")

        transform_meta.imputer_meta.CopyFrom(missing_imputer_meta)
        transform_param.imputer_param.CopyFrom(missing_imputer_param)

        outlier_meta, outlier_param = save_outlier_model(outlier_replace=False,
                                                         model_name="Outlier")

        transform_meta.outlier_meta.CopyFrom(outlier_meta)
        transform_param.outlier_param.CopyFrom(outlier_param)

        return {"DataTransformMeta": transform_meta,
                "DataTransformParam": transform_param
                }

    def load_model(self, model_meta, model_param):
        self.delimitor, self.data_type, _0, self.tag_with_value, self.tag_value_delimitor, self.with_label, \
            self.label_type, self.output_format, self.header, self.sid_name, self.label_name, self.with_match_id, \
            self.anonymous_header = load_data_transform_model(
                "SparseTagTransformer",
                model_meta,
                model_param)

        self.missing_fill, self.missing_fill_method, \
            self.missing_impute, self.default_value = load_missing_imputer_model(self.header,
                                                                                 "Imputer",
                                                                                 model_meta.imputer_meta,
                                                                                 model_param.imputer_param)


class DataTransform(ModelBase):
    def __init__(self):
        super(DataTransform, self).__init__()
        self.transformer = None
        from federatedml.param.data_transform_param import DataTransformParam
        self.model_param = DataTransformParam()
        self._input_model_meta = None
        self._input_model_param = None

    def _load_reader(self, schema=None):
        if schema is None or not schema.get("meta", {}):
            input_format = self.model_param.input_format
        else:
            input_format = schema["meta"].get("input_format")

        if input_format == "dense":
            self.transformer = DenseFeatureTransformer(self.model_param)
        elif input_format == "sparse" or input_format == "svmlight":
            self.transformer = SparseFeatureTransformer(self.model_param)
        elif input_format == "tag":
            self.transformer = SparseTagTransformer(self.model_param)
        else:
            raise ValueError("Cannot recognize input format")

        if self._input_model_meta:
            self.transformer.load_model(self._input_model_meta, self._input_model_param)
            self._input_model_meta, self._input_model_param = None, None

        self.transformer.anonymous_generator = Anonymous(self.role, self.component_properties.local_partyid)

    def _init_model(self, model_param):
        self.model_param = model_param

    def load_model(self, model_dict):
        for _, value in model_dict["model"].items():
            for model in value:
                if model.endswith("Meta"):
                    self._input_model_meta = value[model]
                if model.endswith("Param"):
                    self._input_model_param = value[model]

    def fit(self, data):
        self._load_reader(data.schema)
        data_inst = self.transformer.read_data(data, "fit")
        if isinstance(self.transformer, (DenseFeatureTransformer, SparseTagTransformer)):
            summary_buf = self.transformer.get_summary()
            if summary_buf:
                self.set_summary(summary_buf)

        clear_schema(data_inst)
        return data_inst

    def transform(self, data):
        self._load_reader(data.schema)
        data_inst = self.transformer.read_data(data, "transform")
        clear_schema(data_inst)
        return data_inst

    def export_model(self):
        if not self.need_run:
            model_meta = DataTransformMeta()
            model_meta.need_run = False
            model_param = DataTransformParam()
            model_dict = dict(DataTransformMeta=model_param,
                              DataTransformParam=model_param)
        else:
            model_dict = self.transformer.save_model()

        return model_dict


def clear_schema(data_inst):
    ret_schema = copy.deepcopy(data_inst.schema)
    key_words = {"sid", "header", "anonymous_header", "label_name",
                 "anonymous_label", "match_id_name"}
    for key in data_inst.schema:
        if key not in key_words:
            del ret_schema[key]

    data_inst.schema = ret_schema


def set_schema(data_instance, schema):
    data_instance.schema = schema


def save_data_transform_model(input_format="dense",
                              delimitor=",",
                              data_type="str",
                              exclusive_data_type=None,
                              tag_with_value=False,
                              tag_value_delimitor=":",
                              with_label=False,
                              label_name='',
                              label_type="int",
                              output_format="dense",
                              header=None,
                              sid_name=None,
                              with_match_id=False,
                              model_name="DataTransform",
                              anonymous_header=None):
    model_meta = DataTransformMeta()
    model_param = DataTransformParam()

    model_meta.input_format = input_format
    model_meta.delimitor = delimitor
    model_meta.data_type = data_type
    model_meta.tag_with_value = tag_with_value
    model_meta.tag_value_delimitor = tag_value_delimitor
    model_meta.with_label = with_label
    if with_label:
        model_meta.label_name = label_name
        model_meta.label_type = label_type
    model_meta.output_format = output_format
    model_meta.with_match_id = with_match_id

    if header is not None:
        model_param.header.extend(header)

    if anonymous_header is not None:
        model_param.anonymous_header.extend(anonymous_header)

    if sid_name:
        model_param.sid_name = sid_name

    if label_name:
        model_param.label_name = label_name

    if exclusive_data_type is not None:
        model_meta.exclusive_data_type.update(exclusive_data_type)

    return model_meta, model_param


def load_data_transform_model(model_name="DataTransform",
                              model_meta=None,
                              model_param=None):
    delimitor = model_meta.delimitor
    data_type = model_meta.data_type
    tag_with_value = model_meta.tag_with_value
    tag_value_delimitor = model_meta.tag_value_delimitor
    with_label = model_meta.with_label
    label_name = model_meta.label_name if with_label else None
    label_type = model_meta.label_type if with_label else None
    try:
        with_match_id = model_meta.with_match_id
    except AttributeError:
        with_match_id = False

    output_format = model_meta.output_format

    header = list(model_param.header) or None

    try:
        anonymous_header = list(model_param.anonymous_header)
    except AttributeError:
        anonymous_header = None

    sid_name = None
    if model_param.sid_name:
        sid_name = model_param.sid_name

    exclusive_data_type = None

    if model_meta.exclusive_data_type:
        exclusive_data_type = {}
        for col_name in model_meta.exclusive_data_type:
            exclusive_data_type[col_name] = model_meta.exclusive_data_type.get(col_name)

    return delimitor, data_type, exclusive_data_type, tag_with_value, tag_value_delimitor, with_label, \
        label_type, output_format, header, sid_name, label_name, with_match_id, anonymous_header


def save_missing_imputer_model(missing_fill=False,
                               missing_replace_method=None,
                               missing_impute=None,
                               missing_fill_value=None,
                               missing_replace_rate=None,
                               header=None,
                               model_name="Imputer"):
    model_meta = DataTransformImputerMeta()
    model_param = DataTransformImputerParam()

    model_meta.is_imputer = missing_fill
    if missing_fill:
        if missing_replace_method:
            model_meta.strategy = str(missing_replace_method)

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
    model_meta = DataTransformOutlierMeta()
    model_param = DataTransformOutlierParam()

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
