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
import copy

from federatedml.util  import consts

LOGGER = log_utils.getLogger()


class Step():
    def __init__(self):
        self.feature_list = []
        self.step_direction = ""
        self.n_step = 0
        self.n_model = 0

    def _set_step_info(self, n_step, n_model, step_direction):
        self.n_step = n_step
        self.n_model = n_model
        self.self_direction = step_direction

    def get_new_header(self, header, feature_list):
        """
        Make new header, called by Host or Guest
        :param header: old header
        :param feature_list: list of feature indices to be included in model
        :return: a new header with desired features
        """
        new_header = [header[i] for i in range(len(header)) if i in feature_list]
        return new_header

    def slice_data_instance(self, data_instance, feature_list):
        """
        return data_instance with features at given indices
        Parameters
        ----------
        data_instance: data Instance object, input data
        feature_list: list of desired indices
        """
        data_instance.features = data_instance.features[feature_list]
        return data_instance


    def run(self, stepwise_param, original_model, train_data, test_data, feature_list):
        if stepwise_param.role == consts.ARBITER:
            return self._arbiter_run(original_model)
        model = copy.deepcopy(original_model)
        this_flowid = 'train.' + self.step_direction + '.' + str(self.n_step) + '.' + str(self.n_model)
        model.set_flowid(this_flowid)
        curr_train_data = train_data.map(lambda k, v: (k, self.slice_data_instance(v, feature_list)))
        model.fit(curr_train_data)
        return

    def _arbiter_run(self, original_model):
        model = copy.deepcopy(original_model)
        this_flowid = 'train.' + self.step_direction + '.' + str(self.n_step) + '.' + str(self.n_model)
        model.set_flowid(this_flowid)
        model.fit(None)
        if original_model.model_param.early_stop != 'loss':
            raise ValueError("Stepwise only accepts 'loss' as early stop criteria.")
        #@TODO: (in future) use valdiaton data for calcualtion if needed
        # get final loss from loss history for criteria calculation
        loss = model.loss_history[-1]
        return loss









    

