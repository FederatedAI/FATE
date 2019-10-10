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
from federatedml.ftl.data_util.common_data_util import load_model_parameters, save_model_parameters
from federatedml.linear_model.logistic_regression import HeteroLRGuest
from research.hetero_dnn_logistic_regression.local_model_proxy import BaseLocalModelUpdateProxy, \
    PlainLocalModelUpdateProxy

LOGGER = log_utils.getLogger()


class HeteroDNNLRGuest(HeteroLRGuest):

    def __init__(self, local_model, logistic_params):
        super(HeteroDNNLRGuest, self).__init__(logistic_params)
        self.data_shape = local_model.get_encode_dim()
        self.index_tracking_list = []
        self.local_model = local_model
        self.local_model_update_proxy = PlainLocalModelUpdateProxy()
        self.local_model_update_proxy.set_model(local_model)

    def set_local_model_update_proxy(self, local_model_proxy: BaseLocalModelUpdateProxy):
        self.local_model_update_proxy = local_model_proxy
        self.local_model_update_proxy.set_model(self.local_model)

    def transform(self, instance_table):
        """
        Extract features from instances

        Parameters
        ----------
        :param instance_table: dtable consists of a collection of (index, instance) pairs
        :return: instance_table: dtable consists of a collection of (index, instance) pairs,
        that each instance holds newly extracted features.
        """

        LOGGER.info("@ extract representative features from guest raw input")

        # delegate to local_model_proxy for performing the feature extraction task
        dtable, self.index_tracking_list = self.local_model_update_proxy.transform(instance_table)
        return dtable

    def update_local_model(self, fore_gradient_table, instance_table, coef, **training_info):
        """
        Update local model (i.e., the parameters of local model) based on specified fore_gradient_table, instance_table,
        and coef.

        Parameters
        ----------
        :param fore_gradient_table: dtable consists of a collection of (index, gradient) pairs
        :param instance_table: dtable consists of a collection of (index, instance) pairs
        :param coef: the coefficients of the logistic regression model
        :param training_info: a dictionary holding information on states of training process
        """

        LOGGER.info("@ update guest local model")

        # delegate to local_model_proxy for performing the local model update task
        training_info["index_tracking_list"] = self.index_tracking_list
        training_info["is_host"] = False
        self.local_model_update_proxy.update_local_model(fore_gradient_table, instance_table, coef, **training_info)

    def save_model(self, model_table_name, model_namespace):
        LOGGER.info("@ save guest model to name/ns" + ", " + str(model_table_name) + ", " + str(model_namespace))
        lr_model_name = model_table_name + "_lr_model"
        local_model_name = model_table_name + "_local_model"
        super(HeteroDNNLRGuest, self).save_model(lr_model_name, model_namespace)
        save_model_parameters(self.local_model.get_model_parameters(), local_model_name, model_namespace)

    def load_model(self, model_table_name, model_namespace):
        LOGGER.info("@ load guest model from name/ns" + ", " + str(model_table_name) + ", " + str(model_namespace))
        lr_model_name = model_table_name + "_lr_model"
        local_model_name = model_table_name + "_local_model"
        super(HeteroDNNLRGuest, self).load_model(lr_model_name, model_namespace)
        model_parameters = load_model_parameters(local_model_name, model_namespace)
        self.local_model.restore_model(model_parameters)