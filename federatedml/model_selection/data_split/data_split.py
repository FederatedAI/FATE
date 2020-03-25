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

from sklearn.model_selection import train_test_split

from arch.api.utils import log_utils
from federatedml.feature.binning.base_binning import Binning
from federatedml.model_base import ModelBase
from federatedml.param.data_split_param import DataSplitParam
from federatedml.util import data_io

LOGGER = log_utils.getLogger()


class DataSplitter(ModelBase):
    def __init__(self):
        super().__init__()
        self.metric_name = "data_split"
        self.metric_namespace = "train"
        self.metric_type = "DATASPLIT"
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
        return

    @staticmethod
    def _safe_divide(n, d):
        result = n / d if d != 0 else 0
        return result

    def _split(self, ids, y, test_size, train_size):
        if test_size == 0:
            return ids, [], y, []
        stratify = y if self.stratified else None
        id_train, id_test, y_train, y_test = train_test_split(ids, y,
                                                              test_size=test_size, train_size=train_size,
                                                              random_state=self.random_state,
                                                              shuffle=self.shuffle, stratify=stratify)
        return id_train, id_test, y_train, y_test

    def _get_ids(self, data_inst):
        ids = [i for i, v in data_inst.mapValues(lambda v: None).collect()]
        return ids

    def _get_y(self, data_inst):
        if self.stratified:
            y = [v for i, v in data_inst.mapValues(lambda v: v.label).collect()]
            if self.need_transform:
                y = self.transform_regression_label(data_inst, y)
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
    def get_train_test_size(test_size, validate_size):
        # return original set size if int
        if isinstance(test_size, int):
            return validate_size, test_size
        test_validate_size = test_size + validate_size
        new_train_size = DataSplitter._safe_divide(test_size, test_validate_size)
        new_test_size = DataSplitter._safe_divide(validate_size, test_validate_size)
        return new_test_size, new_train_size

    def param_validator(self, data_inst):
        """
        Validate & transform param inputs

        """
        # check if need label transform
        self.check_need_transform()

        # check & transform data set sizes
        n_count = data_inst.count()
        # only output train data
        if self.train_size is None and self.test_size is None and self.validate_size is None:
            self.train_size = 1.0
            self.test_size, self.validate_size = 0.0, 0.0
        if isinstance(self.test_size, float) or isinstance(self.train_size, float) or isinstance(self.validate_size, float):
            total_size = 1.0
        else:
            total_size = n_count
        if self.train_size is None:
            if self.validate_size is None:
                self.train_size = total_size - self.test_size
                self.validate_size = total_size - (self.test_size + self.train_size)
            else:
                self.train_size = total_size - (self.test_size + self.validate_size)
        elif self.test_size is None:
            if self.validate_size is None:
                self.test_size = total_size - self.train_size
                self.validate_size = total_size - (self.test_size + self.train_size)
            else:
                self.test_size = total_size - (self.validate_size + self.train_size)
        elif self.validate_size is None:
            if self.train_size is None:
                self.train_size = total_size - self.test_size
            else:
                self.test_size = total_size -  self.train_size
            self.validate_size = total_size - (self.test_size + self.train_size)
        if self.train_size + self.test_size + self.validate_size != total_size:
            raise ValueError(f"train_size, test_size, validate_size should sum up to 1.0 or data count")
        return

    def transform_regression_label(self, data_inst, y):
        split_points_bin = self.split_points + [max(y)]
        bin_labels = data_inst.mapValues(lambda v: Binning.get_bin_num(v.label, split_points_bin))
        binned_y = [v for k, v in bin_labels.collect()]
        return binned_y

    @staticmethod
    def _match_id(data_inst, ids):
        return data_inst.filter(lambda k, v: k in ids)

    def split_data(self, data_inst, id_train, id_test, id_validate):
        train_data = DataSplitter._match_id(data_inst, id_train)
        test_data = DataSplitter._match_id(data_inst, id_test)
        validate_data = DataSplitter._match_id(data_inst, id_validate)

        schema = data_inst.schema
        data_io.set_schema(train_data, schema)
        data_io.set_schema(test_data, schema)
        data_io.set_schema(validate_data, schema)
        return train_data, test_data, validate_data


    def fit(self, data_inst):
        LOGGER.debug("fit method in data_split should not be called here.")
        return