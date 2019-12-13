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
from federatedml.model_selection.stepwise import Stepwise

LOGGER = log_utils.getLogger()


class Step(Stepwise):
    def __init__(self):
        super(Step, self).__init__()
        self.model_param = None
        self.forward = False
        self.backward = False
        self.best_list = []

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
        return





    

