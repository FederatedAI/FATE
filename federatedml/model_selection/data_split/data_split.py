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
from arch.api.utils import log_utils
from arch.api import session
from fate_flow.entity.metric import Metric, MetricMeta
from federatedml.model_base import ModelBase
from federatedml.param.data_split_param import DataSplitParam

import numpy as np
from sklearn.model_selection import train_test_split

LOGGER = log_utils.getLogger()

session.init("data split")

class DataSplitter(ModelBase):
    def __init__(self):
        super().__init__()
        self.metric_name = "data_split"
        self.metric_namespace = "train"
        self.metric_type = "DATASPLIT"
        self.model_param = DataSplitParam()
        self.role = None

    def _init_model(self, params):
        self.random_state = params.random_state
        self.test_size = params.test_size
        self.train_size = params.train_size
        self.validate_size = params.validate_size
        self.stratified = params.stratified
        self.shuffle = params.shuffle
        self.bin_interval = params.bin_interval
        return

    def _split(self, ids, y):
        id_train, id_test, y_train, y_test = train_test_split(ids, y, test_size=self.test_size + self.validate_size,
                                                              train_size=self.train_size, random_state=self.random_state,
                                                              shuffle=self.shuffle, stratify=self.stratified)
        id_test, id_validate, y_test, y_validate = train_test_split(id_test, y_test, test_size=self.test_size + self.validate_size,
                                                              train_size=self.train_size, random_state=self.random_state,
                                                              shuffle=self.shuffle, stratify=self.stratified)
        return id_train, id_test, id_validate

    def _get_ids(self, data_inst):
        ids = np.array([i for i, v in data_inst.mapValues(lambda v: None).collect()])
        return ids

    def _get_y(self, data_inst):
        # @TODO: check whether y is categorical or continuous, produce tags based on binning if continuous
        y = np.array([v for i, v in data_inst.mapValues(lambda v: v.label).collect()])
        return y

    def param_validater(self, data_inst):
        """
        Validate & transform data set size inputs

        """
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
                self.validate_size = 0
            else:
                self.train_size = total_size - (self.validate_size + self.test_size)
        elif self.test_size is None:
            if self.validate_size is None:
                self.test_size = total_size - self.train_size
                self.validate_size = 0
            else:
                self.test_size = total_size - (self.validate_size + self.train_size)
        elif self.validate_size is None:
            if self.train_size is None:
                self.train_size = total_size - self.test_size
                self.validate_size = 0
            else:
                self.test_size = total_size -  self.train_size
                self.validate_size = 0
        if self.train_size + self.test_size + self.validate_size != total_size:
            raise ValueError(f"train_size, test_size, validate_size should sum up to 1.0 or data count")

    @staticmethod
    def match_id(data_inst, ids):
        return data_inst.filter(lambda k, v: k in ids)

    def split_data(self, data_inst, id_train, id_test, id_validate):
        # @TODO: consider clean up code & reduce
        id_train_table = session.parallelize(id_train)
        id_test_table = session.parallelize(id_test)
        id_validate_table = session.parallelize(id_validate)
        train_data = DataSplitter.match_id(data_inst, id_train_table)
        test_data = DataSplitter.match_id(data_inst, id_test_table)
        validate_data = DataSplitter.match_id(data_inst, id_validate_table)
        return train_data, test_data, validate_data


    def fit(self, data_inst):
        LOGGER.debug("fit method in data_split should not be called here.")
        return