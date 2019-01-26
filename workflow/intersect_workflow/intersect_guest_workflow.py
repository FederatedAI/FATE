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
from federatedml.param import IntersectParam
from federatedml.statistic.intersect.intersect_guest import RsaIntersectionGuest, RawIntersectionGuest
from federatedml.util.param_extract import ParamExtract
from workflow.workflow import WorkFlow

LOGGER = log_utils.getLogger()


class IntersectGuestWorkFlow(WorkFlow):
    def _initialize(self, config_path):
        self._initialize_workflow_param(config_path)
        self._initialize_intersect(config_path)

    def _initialize_intersect(self, config):
        intersect_param = IntersectParam()
        self.intersect_param = ParamExtract.parse_param_from_config(intersect_param, config)

    def intersect(self, data_instance):
        if self.intersect_param.intersect_method == "rsa":
            LOGGER.info("Using rsa intersection")
            self.intersection = RsaIntersectionGuest(self.intersect_param)
        elif self.intersect_param.intersect_method == "raw":
            LOGGER.info("Using raw intersection")
            self.intersection = RawIntersectionGuest(self.intersect_param)
        else:
            raise TypeError("intersect_method {} is not support yet".format(self.workflow_param.intersect_method))

        intersect_ids = self.intersection.run(data_instance)
        
        self.save_intersect_result(intersect_ids)
        LOGGER.info("Save intersect results")

if __name__ == "__main__":
    intersect_guest_wf = IntersectGuestWorkFlow()
    intersect_guest_wf.run()
