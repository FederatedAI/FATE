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
from fate_flow.entity.metric import Metric, MetricMeta
from federatedml.model_base import ModelBase
from federatedml.param.data_inst_split_param import DataSplitParam
from federatedml.util import consts

import numpy as np
from sklearn.model_selection import ShuffleSplit

LOGGER = log_utils.getLogger()


class DataSplitBase(ModelBase):
    def __init__(self):
        super().__init__()
        self.metric_name = "data_inst_split"
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

    def __split(self, ids):
        """
        Output generators for splitting data
        """
        spliter = ShuffleSplit(n_splits=1., test_size=self.test_size + self.validate_size,
                               train_size=self.train_size, random_state=self.random_state)
        return

    def _get_ids(self, data_inst):
        ids = np.array([i for i,v in data_inst.mapValues(lambda v: None).collect()])
        return ids

    def param_validater(self, data_inst):
        """
        Validate data set size inputs & transform data set sizes

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

    def fit(self, data_inst):
        self.param_validater(data_inst)
        ids = self._get_ids(data_inst)
        return


    def save_data(self):
        return


class HeteroDataSplit(DataSplitBase):
    def __init__(self):
        super().__init__()

class HomoSplitGuest(DataSplitBase):
    def __init__(self):
        super().__init__()
