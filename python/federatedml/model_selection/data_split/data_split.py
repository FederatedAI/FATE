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

import collections

from sklearn.model_selection import train_test_split

from fate_arch.session import computing_session
from federatedml.model_base import Metric, MetricMeta
from federatedml.feature.binning.base_binning import BaseBinning
from federatedml.model_base import ModelBase
from federatedml.param.data_split_param import DataSplitParam
from federatedml.util import LOGGER
from federatedml.util import data_io
from federatedml.util.consts import FLOAT_ZERO

ROUND_NUM = 3


class DataSplitter(ModelBase):
    def __init__(self):
        super().__init__()
        self.metric_name = "data_split"
        self.metric_namespace = "train"
        self.metric_type = "DATA_SPLIT"
        self.model_param = DataSplitParam()
        self.role = None
        self.need_transform = None

    def _init_model(self, params):
        self.random_state = params.random_state
        self.test_size = params.test_size
        self.train_size = params.train_size
        self.validate_size = params.validate_size
        self.stratified = params.stratified
        self.shuffle = params.shuffle
        self.split_points = params.split_points
        if self.split_points:
            self.split_points = sorted(self.split_points)
        self.need_run = params.need_run

    @staticmethod
    def _safe_divide(n, d):
        result = n / d if d > FLOAT_ZERO else 0.0
        if result >= 1:
            result = 1.0
        return result

    def _split(self, ids, y, test_size, train_size):
        if test_size <= FLOAT_ZERO:
            return ids, [], y, []
        if train_size <= FLOAT_ZERO:
            return [], ids, [], y
        stratify = y if self.stratified else None
        if not isinstance(test_size, int):
            train_size = round(train_size * len(ids))
            test_size = len(ids) - train_size
        id_train, id_test, y_train, y_test = train_test_split(ids, y,
                                                              test_size=test_size, train_size=train_size,
                                                              random_state=self.random_state,
                                                              shuffle=self.shuffle, stratify=stratify)
        return id_train, id_test, y_train, y_test

    def _get_ids(self, data_inst):
        ids = sorted([i for i, v in data_inst.mapValues(lambda v: None).collect()])
        return ids

    def _get_y(self, data_inst):
        if self.stratified:
            y = [v for i, v in data_inst.mapValues(lambda v: v.label).collect()]
            if self.need_transform:
                y = self.transform_regression_label(data_inst)
        else:
            # make dummy y
            y = [0] * (data_inst.count())
        return y

    def check_need_transform(self):
        if self.split_points is not None:
            if len(self.split_points) == 0:
                self.need_transform = False
            else:
                # only need to produce binned labels if stratified split needed
                if self.stratified:
                    self.need_transform = True
        return

    @staticmethod
    def get_train_test_size(train_size, test_size):
        LOGGER.debug(f"original train is {train_size}, original test_size is {test_size}")
        # return original set size if int
        if isinstance(test_size, int) and isinstance(train_size, int):
            return train_size, test_size
        total_size = test_size + train_size
        new_train_size = DataSplitter._safe_divide(train_size, total_size)
        new_test_size = DataSplitter._safe_divide(test_size, total_size)
        LOGGER.debug(f"new_train_size is {new_train_size}, new_test_size is {new_test_size}")
        return new_train_size, new_test_size

    def param_validator(self, data_inst):
        """
        Validate & transform param inputs

        """
        # check if need label transform
        self.check_need_transform()

        # check & transform data set sizes
        n_count = data_inst.count()
        if isinstance(self.test_size, float) or isinstance(self.train_size, float) or isinstance(self.validate_size,
                                                                                                 float):
            total_size = 1.0
        else:
            total_size = n_count
        if self.train_size is None:
            if self.validate_size is None:
                self.train_size = total_size - self.test_size
                self.validate_size = total_size - (self.test_size + self.train_size)
            else:
                if self.test_size is None:
                    self.test_size = 0
                self.train_size = total_size - (self.validate_size + self.test_size)
        elif self.test_size is None:
            if self.validate_size is None:
                self.test_size = total_size - self.train_size
                self.validate_size = total_size - (self.test_size + self.train_size)
            else:
                self.test_size = total_size - (self.validate_size + self.train_size)
        elif self.validate_size is None:
            if self.train_size is None:
                self.train_size = total_size - self.test_size
            self.validate_size = total_size - (self.test_size + self.train_size)

        if abs((abs(self.train_size) + abs(self.test_size) + abs(self.validate_size)) - total_size) > FLOAT_ZERO:
            raise ValueError(f"train_size, test_size, validate_size should sum up to 1.0 or data count")
        return

    def transform_regression_label(self, data_inst):
        edge = self.split_points[-1] + 1
        split_points_bin = self.split_points + [edge]
        bin_labels = data_inst.mapValues(lambda v: BaseBinning.get_bin_num(v.label, split_points_bin))
        binned_y = [v for k, v in bin_labels.collect()]
        return binned_y

    @staticmethod
    def get_class_freq(y, split_points=None, label_names=None):
        """
        get frequency info of a given y set; only called when stratified is true
        :param y: list, y sample
        :param split_points: list, split points used to bin regression values
        :param label_names: list, label names of all data
        :return: dict
        """
        freq_dict = collections.Counter(y)
        freq_keys = freq_dict.keys()
        # continuous label
        if split_points is not None:
            label_count = len(split_points) + 1
            # fill in count for missing bins
            if len(freq_keys) < label_count:
                for i in range(label_count):
                    if i not in freq_keys:
                        freq_dict[i] = 0
        # categorical label
        else:
            if label_names is None:
                raise ValueError("No label values collected.")
            label_count = len(label_names)
            # fill in count for missing labels
            if len(freq_keys) < label_count:
                for label in label_names:
                    if label not in freq_keys:
                        freq_dict[label] = 0
        return freq_dict

    def callback_count_info(self, id_train, id_validate, id_test, all_metas):
        """
        Tool to callback returned data count & ratio information
        Parameters
        ----------
        id_train: list, id of data set
        id_validate: list, id of data set
        id_test: list, id of data set
        all_metas: dict, all meta info

        Returns
        -------
        None
        """
        metas = {}

        train_count = len(id_train)
        metas["train"] = train_count

        validate_count = len(id_validate)
        metas["validate"] = validate_count

        test_count = len(id_test)
        metas["test"] = test_count

        original_count = train_count + validate_count + test_count
        metas["original"] = original_count

        metric_name = f"{self.metric_name}_count_info"
        all_metas[metric_name] = metas

        metas = {}

        train_ratio = train_count / original_count
        validate_ratio = validate_count / original_count
        test_ratio = test_count / original_count

        metas["train"] = round(train_ratio, ROUND_NUM)
        metas["validate"] = round(validate_ratio, ROUND_NUM)
        metas["test"] = round(test_ratio, ROUND_NUM)

        metric_name = f"{self.metric_name}_ratio_info"
        all_metas[metric_name] = metas

        # stratified
        all_metas["stratified"] = self.stratified

        return all_metas

    def callback_label_info(self, y_train, y_validate, y_test, all_metas):
        """
        Tool to callback returned data label information
        Parameters
        ----------
        y_train: list, y
        y_validate: list, y
        y_test: list, y
        all_metas: dict, all meta info

        Returns
        -------
        None

        """
        metas = {}
        y_all = y_train + y_validate + y_test

        label_names = None
        if self.split_points is None:
            label_names = list(set(y_all))

        original_freq_dict = DataSplitter.get_class_freq(y_all, self.split_points, label_names)
        metas["original"] = original_freq_dict

        train_freq_dict = DataSplitter.get_class_freq(y_train, self.split_points, label_names)
        metas["train"] = train_freq_dict

        validate_freq_dict = DataSplitter.get_class_freq(y_validate, self.split_points, label_names)
        metas["validate"] = validate_freq_dict

        test_freq_dict = DataSplitter.get_class_freq(y_test, self.split_points, label_names)
        metas["test"] = test_freq_dict

        if self.split_points is not None:
            metas["split_points"] = self.split_points
            metas["continuous_label"] = True
        else:
            metas["label_names"] = label_names
            metas["continuous_label"] = False

        metric_name = f"{self.metric_name}_label_info"
        all_metas[metric_name] = metas

        return all_metas

    def callback(self, metas):
        metric = [Metric(self.metric_name, 0)]
        self.callback_metric(metric_name=self.metric_name, metric_namespace=self.metric_namespace, metric_data=metric)
        self.tracker.set_metric_meta(metric_name=self.metric_name, metric_namespace=self.metric_namespace,
                                     metric_meta=MetricMeta(name=self.metric_name, metric_type=self.metric_type,
                                                            extra_metas=metas))

    @staticmethod
    def _match_id(data_inst, ids):
        ids = [(i, None) for i in ids]
        id_table = computing_session.parallelize(ids, include_key=True, partition=data_inst.partitions)
        return data_inst.join(id_table, lambda v1, v2: v1)

    @staticmethod
    def _set_output_table_schema(data_inst, schema):
        if schema is not None and data_inst.count() > 0:
            data_io.set_schema(data_inst, schema)

    def split_data(self, data_inst, id_train, id_validate, id_test):
        train_data = DataSplitter._match_id(data_inst, id_train)
        validate_data = DataSplitter._match_id(data_inst, id_validate)
        test_data = DataSplitter._match_id(data_inst, id_test)

        schema = getattr(data_inst, "schema", None)
        self._set_output_table_schema(train_data, schema)
        self._set_output_table_schema(validate_data, schema)
        self._set_output_table_schema(test_data, schema)
        return train_data, validate_data, test_data

    def fit(self, data_inst):
        raise NotImplementedError("fit method in data_split should not be called here.")
