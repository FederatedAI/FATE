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

import argparse
from fate_test.utils import parse_summary_result
from fate_client.pipeline.utils import test_utils
from fate_client.pipeline.components.fate import HeteroSecureBoost, PSI, Reader
from fate_client.pipeline.components.fate.evaluation import Evaluation
from fate_client.pipeline import FateFlowPipeline


def main(config="../../config.yaml", param="./sbt_breast_config.yaml", namespace=""):
    # obtain config
    if isinstance(config, str):
        config = test_utils.load_job_config(config)
    parties = config.parties
    guest = parties.guest[0]
    host = parties.host[0]

    if isinstance(param, str):
        param = test_utils.JobConfig.load_from_file(param)

    assert isinstance(param, dict)

    guest_data_table = param.get("data_guest")
    host_data_table = param.get("data_host")

    guest_train_data = {"name": guest_data_table, "namespace": f"experiment{namespace}"}
    host_train_data = {"name": host_data_table, "namespace": f"experiment{namespace}"}
    pipeline = FateFlowPipeline().set_parties(guest=guest, host=host)

    reader_0 = Reader("reader_0")
    reader_0.guest.task_parameters(
        namespace=guest_train_data['namespace'],
        name=guest_train_data['name']
    )
    reader_0.hosts[0].task_parameters(
        namespace=host_train_data['namespace'],
        name=guest_train_data['name']
    )

    psi_0 = PSI("psi_0", input_data=reader_0.outputs["output_data"])

    config_param = {
        "num_trees": param["num_trees"],
        "max_depth": param["max_depth"],
        "max_bin": param["max_bin"],
        "objective": param.get("objective", "binary:bce"),
    }
    hetero_sbt_0 = HeteroSecureBoost('sbt_0', train_data=psi_0.outputs['output_data'], num_trees=config_param["num_trees"], 
                             max_bin=config_param["max_bin"], max_depth=config_param["max_depth"], he_param={'kind': 'paillier', 'key_length': 1024},
                             objective=config_param["objective"])
    
    hetero_sbt_1 = HeteroSecureBoost('sbt_1',
                                     test_data=psi_0.outputs['output_data'],
                                     input_model=hetero_sbt_0.outputs['output_model'],
                                     )

    if config_param['objective'] == 'regression:l2':
        evaluation_0 = Evaluation(
            'eval_0',
            runtime_parties=dict(guest=guest),
            input_data=[hetero_sbt_0.outputs['train_output_data']],
            default_eval_setting='regression',
        )


    else:
        evaluation_0 = Evaluation(
            'eval_0',
            runtime_parties=dict(guest=guest),
            metrics=['auc'],
            input_data=[hetero_sbt_0.outputs['train_output_data']]
        )

    pipeline.add_task(reader_0)
    pipeline.add_task(psi_0)
    pipeline.add_task(hetero_sbt_0)
    pipeline.add_task(hetero_sbt_1)
    pipeline.add_task(evaluation_0)

    if config.task_cores:
        pipeline.conf.set("task_cores", config.task_cores)
    if config.timeout:
        pipeline.conf.set("timeout", config.timeout)

    pipeline.compile()
    pipeline.fit()

    result_summary = parse_summary_result(pipeline.get_task_info("eval_0").get_output_metric()[0]["data"])
    print(f"result_summary: {result_summary}")

    return pipeline.model_info.job_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser("BENCHMARK-QUALITY PIPELINE JOB")
    parser.add_argument("-c", "--config", type=str,
                        help="config file", default="../../config.yaml")
    parser.add_argument("-p", "--param", type=str,
                        help="config file for params", default="./sbt_breast_config.yaml")
    args = parser.parse_args()
    main(args.config, args.param)