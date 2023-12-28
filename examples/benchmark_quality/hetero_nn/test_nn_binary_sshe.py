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
from fate_client.pipeline.utils import test_utils
from fate_client.pipeline import FateFlowPipeline
from fate_client.pipeline.components.fate.nn.torch import nn, optim
from fate_client.pipeline.components.fate.nn.torch.base import Sequential
from fate_client.pipeline.components.fate.hetero_nn import HeteroNN, get_config_of_default_runner
from fate_client.pipeline.components.fate.reader import Reader
from fate_client.pipeline.components.fate.psi import PSI
from fate_client.pipeline.components.fate.nn.algo_params import TrainingArguments, SSHEArgument
from fate_client.pipeline.components.fate import Evaluation


def main(config="../../config.yaml", param="./breast_config.yaml", namespace=""):

    # obtain config
    if isinstance(config, str):
        config = test_utils.load_job_config(config)
    parties = config.parties
    guest = parties.guest[0]
    host = parties.host[0]

    if isinstance(param, str):
        param = test_utils.JobConfig.load_from_file(param)


    pipeline = FateFlowPipeline().set_parties(guest=guest, host=host)

    reader_0 = Reader("reader_0", runtime_parties=dict(guest=guest, host=host))
    reader_0.guest.task_parameters(
        namespace=f"experiment{namespace}",
        name=param['guest_table']
    )
    reader_0.hosts[0].task_parameters(
        namespace=f"experiment{namespace}",
        name=param['host_table']
    )
    psi_0 = PSI("psi_0", input_data=reader_0.outputs["output_data"])

    training_args = TrainingArguments(
        num_train_epochs=param['epochs'],
        per_device_train_batch_size=param['batch_size'],
        logging_strategy='epoch',
    )

    guest_top = param['guest_model']['top']
    guest_bottom = param['guest_model']['bottom']
    host_bottom = param['host_model']['bottom']
    agg_layer_guest = param['guest_model']['agg_layer']
    agg_layer_host = param['host_model']['agg_layer']
    lr = param['lr']

    guest_conf = get_config_of_default_runner(
        bottom_model=nn.Linear(guest_bottom[0], guest_bottom[1]),
        top_model=Sequential(
            nn.Linear(guest_top[0], guest_top[1]),
            nn.Sigmoid()
        ),
        training_args=training_args,
        optimizer=optim.Adam(lr=lr),
        loss=nn.BCELoss(),
        agglayer_arg=SSHEArgument(
            guest_in_features=agg_layer_guest[0],
            host_in_features=agg_layer_host[0],
            out_features=agg_layer_guest[1]
        )
    )

    host_conf = get_config_of_default_runner(
        bottom_model=nn.Linear(host_bottom[0], host_bottom[1]),
        optimizer=optim.Adam(lr=lr),
        training_args=training_args,
        agglayer_arg=SSHEArgument(
            guest_in_features=agg_layer_guest[0],
            host_in_features=agg_layer_host[0],
            out_features=agg_layer_host[1]
        )
    )

    hetero_nn_0 = HeteroNN(
        'hetero_nn_0',
        train_data=psi_0.outputs['output_data'], validate_data=psi_0.outputs['output_data']
    )

    hetero_nn_0.guest.task_parameters(runner_conf=guest_conf)
    hetero_nn_0.hosts[0].task_parameters(runner_conf=host_conf)

    hetero_nn_1 = HeteroNN(
        'hetero_nn_1',
        test_data=psi_0.outputs['output_data'],
        predict_model_input=hetero_nn_0.outputs['train_model_output']
    )

    if param['is_binary']:
        metrics = ['auc']
    else:
        metrics = ['accuracy']

    evaluation_0 = Evaluation(
        'eval_0',
        runtime_parties=dict(guest=guest),
        metrics=metrics,
        input_data=[hetero_nn_0.outputs['train_data_output'], hetero_nn_1.outputs['predict_data_output']]
    )

    pipeline.add_tasks([reader_0, psi_0, hetero_nn_0, hetero_nn_1, evaluation_0])
    pipeline.compile()
    pipeline.fit()

    result_summary = pipeline.get_task_info("eval_0").get_output_metric()[0]["data"]
    print(f"result_summary: {result_summary}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser("BENCHMARK-QUALITY PIPELINE JOB")
    parser.add_argument("-c", "--config", type=str,
                        help="config file", default="../../config.yaml")
    parser.add_argument("-p", "--param", type=str,
                        help="config file for params", default="./breast_config.yaml")
    args = parser.parse_args()
    main(args.config, args.param)
