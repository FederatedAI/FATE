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

from federatedml.ftl.hetero_ftl.hetero_ftl_arbiter import HeteroFTLArbiter
from workflow.hetero_ftl_workflow.hetero_workflow import FTLWorkFlow


class FTLArbiterWorkFlow(FTLWorkFlow):

    def __init__(self):
        super(FTLArbiterWorkFlow, self).__init__()

    def _do_initialize_model(self, ftl_model_param, ftl_local_model_param, ftl_data_model_param):
        self.ftl_arbiter = HeteroFTLArbiter(ftl_model_param)

    def train(self, data_instance, validation_data=None):
        self.ftl_arbiter.fit()


if __name__ == "__main__":
    conf = sys.argv[1]
    guest_wf = FTLArbiterWorkFlow()
    guest_wf.run()