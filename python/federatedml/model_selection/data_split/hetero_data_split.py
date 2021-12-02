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

from federatedml.model_selection.data_split.data_split import DataSplitter
from federatedml.transfer_variable.transfer_class.data_split_transfer_variable import \
    DataSplitTransferVariable
from federatedml.util import LOGGER
from federatedml.util import consts


class HeteroDataSplitHost(DataSplitter):
    def __init__(self):
        super().__init__()
        self.transfer_variable = DataSplitTransferVariable()

    def fit(self, data_inst):
        if self.need_run is False:
            return
        LOGGER.debug(f"Enter Hetero {self.role} Data Split fit")

        id_train = self.transfer_variable.id_train.get(idx=0)
        id_test = self.transfer_variable.id_test.get(idx=0)
        id_validate = self.transfer_variable.id_validate.get(idx=0)

        train_data, validate_data, test_data = self.split_data(data_inst, id_train, id_validate, id_test)

        all_metas = {}
        all_metas = self.callback_count_info(id_train, id_validate, id_test, all_metas)
        self.callback(all_metas)
        self.set_summary(all_metas)

        return [train_data, validate_data, test_data]


class HeteroDataSplitGuest(DataSplitter):
    def __init__(self):
        super().__init__()
        self.transfer_variable = DataSplitTransferVariable()

    def fit(self, data_inst):
        LOGGER.debug(f"Enter Hetero {self.role} Data Split fit")
        if self.need_run is False:
            return
        self.param_validator(data_inst)

        ids = self._get_ids(data_inst)
        y = self._get_y(data_inst)

        id_train, id_test_validate, y_train, y_test_validate = self._split(
            ids, y, test_size=self.test_size + self.validate_size, train_size=self.train_size)

        validate_size, test_size = DataSplitter.get_train_test_size(self.validate_size, self.test_size)
        id_validate, id_test, y_validate, y_test = self._split(id_test_validate, y_test_validate,
                                                               test_size=test_size, train_size=validate_size)

        self.transfer_variable.id_train.remote(obj=id_train, role=consts.HOST, idx=-1)
        self.transfer_variable.id_test.remote(obj=id_test, role=consts.HOST, idx=-1)
        self.transfer_variable.id_validate.remote(obj=id_validate, role=consts.HOST, idx=-1)

        train_data, validate_data, test_data = self.split_data(data_inst, id_train, id_validate, id_test)

        all_metas = {}

        all_metas = self.callback_count_info(id_train, id_validate, id_test, all_metas)
        # summary["data_split_count_info"] = all_metas
        if self.stratified:
            all_metas = self.callback_label_info(y_train, y_validate, y_test, all_metas)
            #summary["data_split_label_info"] = all_metas
        self.callback(all_metas)
        self.set_summary(all_metas)

        return [train_data, validate_data, test_data]
