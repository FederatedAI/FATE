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
from federatedml.util import consts

import copy

LOGGER = log_utils.getLogger()


class Step(object):
    def __init__(self):
        self.feature_list = []
        self.step_direction = ""
        self.n_step = 0
        self.n_model = 0

    def set_step_info(self, step_info):
        step_direction, n_step, n_model = step_info
        self.n_step = n_step
        self.n_model = n_model
        self.self_direction = step_direction

    def get_flowid(self):
        flowid = "train.{}.{}.{}".format(self.step_direction, self.n_step, self.n_model)
        return flowid

    @staticmethod
    def get_new_header(header, feature_list):
        """
        Make new header, called by Host or Guest
        :param header: old header
        :param feature_list: list of feature indices to be included in model
        :return: a new header with de sired features
        """
        new_header = [header[i] for i in range(len(header)) if i in feature_list]
        return new_header

    @staticmethod
    def slice_data_instance_list(data_instance, feature_list):
        """
        return data_instance with features at given indices
        Parameters
        ----------
        data_instance: data Instance object, input data
        feature_list: list of desired indices
        """
        data_instance.features = data_instance.features[feature_list]
        return data_instance

    @staticmethod
    def slice_data_instance(data_instance, feature_mask):
        """
        return data_instance with features at given indices
        Parameters
        ----------
        data_instance: data Instance object, input data
        feature_mask: mask to filter data_instance
        """
        new_data_instance = copy.deepcopy(data_instance)
        new_data_instance.features = new_data_instance.features[feature_mask > 0]
        return new_data_instance

    def run(self, original_model, train_data, test_data, feature_mask):
        if original_model.model_param.early_stop != 'loss':
            raise ValueError("Stepwise only accepts 'loss' as early stop criteria.")
        model = copy.deepcopy(original_model)
        current_flowid = self.get_flowid()
        model.set_flowid(current_flowid)
        if original_model.role != consts.ARBITER:
            curr_train_data = train_data.map(lambda k, v: (k, self.slice_data_instance(v, feature_mask)))
        else:
            curr_train_data = train_data
        model.fit(curr_train_data)
        return model
