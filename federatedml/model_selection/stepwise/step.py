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
from federatedml.model_selection.stepwise.stepwise import Stepwise
from federatedml.util  import consts

LOGGER = log_utils.getLogger()


class Step(Stepwise):
    def __init__(self):
        super(Step, self).__init__()

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


    def run(self, stepwise_param, train_data, validate_data, model):
        #@TODO: drop_one & add_one for each step
        #@TODO use "map" to make new dTable
        if self.role == consts.ARBITER:
            self._arbiter_run(stepwise_param, model)
        return

    def _arbiter_run(self, stepwise_param, model):
        #@TODO: add calculate AIC/BIC here, return calculate value









    

