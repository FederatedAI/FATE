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

import sys

import tensorflow as tf

from arch.api.utils import log_utils
from federatedml.ftl.autoencoder import Autoencoder
from federatedml.ftl.hetero_ftl.hetero_ftl_guest import GuestFactory
from workflow.hetero_ftl_workflow.hetero_workflow import FTLWorkFlow

LOGGER = log_utils.getLogger()


class FTLGuestWorkFlow(FTLWorkFlow):

    def __init__(self):
        super(FTLGuestWorkFlow, self).__init__()

    def _do_initialize_model(self, ftl_model_param, ftl_local_model_param, ftl_data_model_param):
        self.ftl_local_model = self._create_local_model(ftl_local_model_param, ftl_data_model_param)
        self.model = GuestFactory.create(ftl_model_param, self._get_transfer_variable(), self.ftl_local_model)

    @staticmethod
    def _create_local_model(ftl_local_model_param, ftl_data_model_param):
        autoencoder = Autoencoder("local_ftl_guest_model_01")
        autoencoder.build(input_dim=ftl_data_model_param.n_feature_guest, hidden_dim=ftl_local_model_param.encode_dim,
                          learning_rate=ftl_local_model_param.learning_rate)
        return autoencoder

    def train(self, train_data_instance, validation_data=None):
        LOGGER.debug("@ enter guest workflow train function")
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            self.ftl_local_model.set_session(sess)
            sess.run(init)
            self.model.fit(train_data_instance)
            self.model.save_model(self.workflow_param.model_table, self.workflow_param.model_namespace)
            self.model.predict(train_data_instance)
            if validation_data is not None:
                LOGGER.debug("@ validation")
                self.model.predict(validation_data)

    def predict(self, data_instance):
        LOGGER.debug("@ enter guest workflow predict function")
        tf.reset_default_graph()
        self.model.load_model(self.workflow_param.model_table, self.workflow_param.model_namespace)
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            self.ftl_local_model.set_session(sess)
            sess.run(init)
            self.model.predict(data_instance)


if __name__ == "__main__":
    conf = sys.argv[1]
    guest_wf = FTLGuestWorkFlow()
    guest_wf.run()
