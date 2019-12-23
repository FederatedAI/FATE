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

LOGGER = log_utils.getLogger()


class HeteroNNModel(object):
    def __init__(self):
        self.partition = 1

    def load_model(self):
        pass

    def predict(self, data):
        pass

    def export_model(self):
        pass

    def get_hetero_nn_model_meta(self):
        pass

    def get_hetero_nn_model_param(self):
        pass

    def set_hetero_nn_model_meta(self, model_meta):
        pass

    def set_hetero_nn_model_param(self, model_param):
        pass

    def set_partition(self, partition):
        pass


class HeteroNNHostModel(HeteroNNModel):
    def __init__(self):
        super(HeteroNNHostModel, self).__init__()
        self.role = "host"

    def train(self, x, epoch, batch):
        pass

    def evaluate(self, x, epoch ,batch):
        pass


class HeteroNNGuestModel(HeteroNNModel):
    def __init__(self):
        super(HeteroNNGuestModel, self).__init__()
        self.role = "guest"

    def train(self, x, y, epoch, batch):
        pass

    def evaluate(self, x, y, epoch ,batch):
        pass
