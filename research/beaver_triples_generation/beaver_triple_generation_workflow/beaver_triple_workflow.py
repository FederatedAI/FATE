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
from workflow.hetero_ftl_workflow.hetero_guest_workflow import FTLGuestWorkFlow
from federatedml.ftl.data_util.uci_credit_card_util import load_guest_host_dtable_from_UCI_Credit_Card
from arch.api import eggroll
from arch.api import federation
from federatedml.util.transfer_variable import BeaverTripleTransferVariable
from research.beaver_triples_generation.beaver_triple import PartyABeaverTripleGenerationHelper
from research.beaver_triples_generation.bt_guest import BeaverTripleGenerationGuest
from arch.api.utils import log_utils
from federatedml.param import WorkFlowParam
from federatedml.param import param as param_generator
from federatedml.param.param import OneVsRestParam
from federatedml.param.param import SampleParam
from federatedml.param.param import ScaleParam
from federatedml.statistic.intersect import RawIntersectionHost, RawIntersectionGuest
from federatedml.util import ParamExtract, DenseFeatureReader, SparseFeatureReader
from federatedml.util import WorkFlowParamChecker

LOGGER = log_utils.getLogger()

config_path = "./conf/guest_runtime_conf.json"


class BeaverTripleGenerationGuestWorkflow(object):

    def _initialize_workflow_param(self, config_path):
        workflow_param = WorkFlowParam()
        self.workflow_param = ParamExtract.parse_param_from_config(workflow_param, config_path)
        WorkFlowParamChecker.check_param(self.workflow_param)

    def _init_argument(self):
        self._initialize_workflow_param(config_path)
        self._initialize_beaver_triple_generator(config_path)
        with open(config_path) as conf_f:
            runtime_json = json.load(conf_f)

        LOGGER.debug("The Guest job id is {}".format(job_id))
        LOGGER.debug("The Guest work mode id is {}".format(self.workflow_param.work_mode))
        eggroll.init(job_id, self.workflow_param.work_mode)
        federation.init(job_id, runtime_json)
        LOGGER.debug("Finish eggroll and federation init")

    def _initialize_beaver_triple_generator(self, config):
        LOGGER.debug("@ initialize guest beaver triple generator")
        self.ftl_transfer_variable = BeaverTripleTransferVariable()
        mul_ops, global_iters, num_batch = fill_beaver_triple_matrix_shape(mul_op_def, num_epoch)
        party_a_bt_gene_helper = PartyABeaverTripleGenerationHelper(mul_ops, global_iters, num_batch)
        self.guest = BeaverTripleGenerationGuest(party_a_bt_gene_helper, self.ftl_transfer_variable)

    def run(self):
        self.guest.generate()


if __name__ == '__main__':
    job_id = sys.argv[1]
    workflow = BeaverTripleGenerationGuestWorkflow()
    workflow.run()
