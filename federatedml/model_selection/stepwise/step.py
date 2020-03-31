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
from federatedml.util.data_io import make_schema, set_schema

import copy
import numpy as np

LOGGER = log_utils.getLogger()


class Step(object):
    def __init__(self):
        self.feature_list = []
        self.step_direction = ""
        self.n_step = 0
        self.n_model = 0

    def set_step_info(self, step_info):
        n_step, n_model = step_info
        self.n_step = n_step
        self.n_model = n_model

    def get_flowid(self):
        flowid = "train.step{}.model{}".format(self.n_step, self.n_model)
        return flowid

    @staticmethod
    def slice_data_instance(data_instance, feature_mask):
        """
        return data_instance with features at given indices
        Parameters
        ----------
        data_instance: data Instance object, input data
        feature_mask: mask to filter data_instance
        """
        data_instance.features = data_instance.features[feature_mask]
        return data_instance

    @staticmethod
    def get_new_schema(original_data, feature_mask):
        old_header = original_data.schema.get("header")
        sid_name = original_data.schema.get("sid_name")
        label_name = original_data.schema.get("label_name")
        new_header = [old_header[i] for i in np.where(feature_mask > 0)[0]]
        schema = make_schema(new_header, sid_name, label_name)
        return schema

    def run(self, original_model, train_data, validate_data, feature_mask):
        model = copy.deepcopy(original_model)
        current_flowid = self.get_flowid()
        model.set_flowid(current_flowid)
        if original_model.role != consts.ARBITER:
            curr_train_data = train_data.mapValues(lambda v: Step.slice_data_instance(v, feature_mask))
            new_schema = Step.get_new_schema(train_data, feature_mask)
            LOGGER.debug("new schema is: {}".format(new_schema))
            set_schema(curr_train_data, new_schema)
            model.header = new_schema.get("header")
        else:
            curr_train_data = train_data
        model.fit(curr_train_data)
        return model
