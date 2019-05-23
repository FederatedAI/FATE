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

from arch.api.utils import log_utils
from federatedml.util.transfer_variable import BeaverTripleTransferVariable
from research.beaver_triples_generation.beaver_triple import PartyBBeaverTripleGenerationHelper, \
    fill_beaver_triple_matrix_shape
from research.beaver_triples_generation.beaver_triple_generation_workflow import BeaverTripleGenerationWorkflow
from research.beaver_triples_generation.bt_host import BeaverTripleGenerationHost
from research.examples.beaver_triple_generation.operation_definition import get_mul_op_def_example

LOGGER = log_utils.getLogger()


class HostBeaverTripleGenerationWorkflow(BeaverTripleGenerationWorkflow):

    def _initialize_beaver_triple_generator(self, config):
        LOGGER.debug("@ initialize host beaver triple generator")
        self.ftl_transfer_variable = BeaverTripleTransferVariable()

        mul_op_def = get_mul_op_def_example()
        num_epoch = 1

        mul_ops, global_iters, num_batch = fill_beaver_triple_matrix_shape(mul_op_def, num_epoch)
        party_b_bt_gene_helper = PartyBBeaverTripleGenerationHelper(mul_ops, global_iters, num_batch)
        self.host = BeaverTripleGenerationHost(party_b_bt_gene_helper, self.ftl_transfer_variable)

    def run(self):
        self.init_argument(config_path=config_path, job_id=job_id)
        self.host.generate()


if __name__ == '__main__':
    job_id = sys.argv[1]
    config_path = "./conf/host_runtime_conf.json"
    workflow = HostBeaverTripleGenerationWorkflow()
    workflow.run()
