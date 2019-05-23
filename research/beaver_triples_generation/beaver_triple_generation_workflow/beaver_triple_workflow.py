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

from arch.api import eggroll
from arch.api import federation
from arch.api.utils import log_utils
from federatedml.param import WorkFlowParam
from federatedml.util import ParamExtract
from federatedml.util import WorkFlowParamChecker

LOGGER = log_utils.getLogger()


class BeaverTripleGenerationWorkflow(object):

    def init_argument(self, config_path, job_id):
        self._initialize_workflow_param(config_path)
        self._initialize_beaver_triple_generator(config_path)
        with open(config_path) as conf_f:
            runtime_json = json.load(conf_f)

        LOGGER.debug("The job id is {}".format(job_id))
        LOGGER.debug("The work mode id is {}".format(self.workflow_param.work_mode))
        eggroll.init(job_id, self.workflow_param.work_mode)
        federation.init(job_id, runtime_json)
        LOGGER.debug("Finish eggroll and federation init")

    def _initialize_workflow_param(self, config_path):
        workflow_param = WorkFlowParam()
        self.workflow_param = ParamExtract.parse_param_from_config(workflow_param, config_path)
        # WorkFlowParamChecker.check_param(self.workflow_param)

    def _initialize_beaver_triple_generator(self, config):
        pass
