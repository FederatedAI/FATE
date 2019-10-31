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

import json
import sys
from workflow.hetero_ftl_workflow.hetero_arbiter_workflow import FTLArbiterWorkFlow
from arch.api import session
from arch.api import federation
from arch.api.utils import log_utils
LOGGER = log_utils.getLogger()

config_path = "./conf/arbiter_runtime_conf.json"


class TestFTLArbiter(FTLArbiterWorkFlow):

    def __init__(self):
        super(TestFTLArbiter, self).__init__()

    def _init_argument(self):
        with open(config_path) as conf_f:
            runtime_json = json.load(conf_f)
        self._initialize(runtime_json)

        LOGGER.debug("The Arbiter job id is {}".format(job_id))
        LOGGER.debug("The Arbiter work mode id is {}".format(self.workflow_param.work_mode))
        session.init(job_id, self.workflow_param.work_mode)
        federation.init(job_id, runtime_json)
        LOGGER.debug("Finish eggroll and federation init")


if __name__ == '__main__':
    job_id = sys.argv[1]
    workflow = TestFTLArbiter()
    workflow.run()
