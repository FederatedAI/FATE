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
from federatedml.model_selection.data_split.data_split import DataSplitter
from federatedml.transfer_variable.transfer_class.data_split_transfer_variable_transfer_variable import DataSplitTransferVariable
from federatedml.util import consts

LOGGER = log_utils.getLogger()


class HeteroDataSplitHost(DataSplitter):
    def __init__(self):
        super().__init__()
        self.transfer_variable = DataSplitTransferVariable()

    def fit(self, data_inst):
        LOGGER.debug(f"Enter Hetero {self.role} Data Split fit")

        id_train = self.transfer_variable.id_train.get(idx=0)
        id_test = self.transfer_variable.id_test.get(idx=0)
        id_validate = self.transfer_variable.id_validate.get(idx=0)

        train_data, test_data, validate_data = self.split_data(data_inst, id_train, id_test, id_validate)

        # return train_data, test_data, validate_data
        LOGGER.debug(f"train_data: {list(train_data.collect())}")
        LOGGER.debug(f"train_data schema: {train_data.schema}")
        LOGGER.debug(f"test_data schema: {test_data.schema}")
        return validate_data

class HeteroDataSplitGuest(DataSplitter):
    def __init__(self):
        super().__init__()
        self.transfer_variable = DataSplitTransferVariable()

    def fit(self, data_inst):
        LOGGER.debug(f"Enter Hetero {self.role} Data Split fit")
        self.param_validator(data_inst)

        ids = self._get_ids(data_inst)
        y = self._get_y(data_inst)

        id_train, id_test_validate, y_train, y_test_validate = self._split(ids, y,
                                                                           test_size=self.test_size + self.validate_size,
                                                                           train_size = self.train_size)

        validate_size, test_size = DataSplitter.get_train_test_size(self.test_size, self.validate_size)
        id_test, id_validate, _, _ = self._split(id_test_validate, y_test_validate, validate_size, test_size)

        self.transfer_variable.id_train.remote(obj=id_train,role=consts.HOST, idx=-1)
        self.transfer_variable.id_test.remote(obj=id_test, role=consts.HOST, idx=-1)
        self.transfer_variable.id_validate.remote(obj=id_validate, role=consts.HOST, idx=-1)

        train_data, test_data, validate_data = self.split_data(data_inst, id_train, id_test, id_validate)

        # return train_data, test_data, validate_data
        LOGGER.debug(f"train_data: {list(train_data.collect())}")
        LOGGER.debug(f"train_data schema: {train_data.schema}")
        LOGGER.debug(f"test_data schema: {test_data.schema}")
        return validate_data
