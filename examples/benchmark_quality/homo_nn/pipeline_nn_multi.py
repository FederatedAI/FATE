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
from fate_client.pipeline.components.fate import Reader
from fate_client.pipeline import FateFlowPipeline
from fate_client.pipeline.utils import test_utils
from fate_client.pipeline.components.fate.evaluation import Evaluation
from fate_client.pipeline import FateFlowPipeline
from fate_client.pipeline.components.fate.nn.torch import nn, optim
from fate_client.pipeline.components.fate.nn.loader import ModelLoader
from fate_client.pipeline.components.fate.homo_nn import HomoNN, get_config_of_default_runner
from fate_client.pipeline.components.fate.nn.algo_params import TrainingArguments, FedAVGArguments



def main(config="../../config.yaml", param="", namespace=""):
    # obtain config
    if isinstance(config, str):
        config = test_utils.load_job_config(config)
    parties = config.parties
    guest = parties.guest[0]
    host = parties.host[0]
    arbiter = parties.arbiter[0]

    if isinstance(param, str):
        param = test_utils.JobConfig.load_from_file(param)

    assert isinstance(param, dict)

    epochs = param.get('epochs')
    batch_size = param.get('batch_size')
    in_feat = param.get('in_feat')
    out_feat = param.get('out_feat')
    class_num = param.get('class_num')
    lr = param.get('lr')

    guest_data_table = param.get("data_guest")
    host_data_table = param.get("data_host")
    test_data_table = param.get("data_test")

    guest_train_data = {"name": guest_data_table, "namespace": f"experiment{namespace}"}
    host_train_data = {"name": host_data_table, "namespace": f"experiment{namespace}"}
    test_data = {"name": test_data_table, "namespace": f"experiment{namespace}"}
    pipeline = FateFlowPipeline().set_parties(guest=guest, host=host, arbiter=arbiter)

    conf = get_config_of_default_runner(
        algo='fedavg',
        model=ModelLoader('multi_model', 'Multi', feat=in_feat, class_num=class_num), 
        loss=nn.CrossEntropyLoss(),
        optimizer=optim.Adam(lr=lr),
        training_args=TrainingArguments(num_train_epochs=epochs, per_device_train_batch_size=batch_size, seed=114514),
        fed_args=FedAVGArguments(),
        task_type='multi'
        )
    
    reader_0 = Reader("reader_0", runtime_parties=dict(guest=guest, host=host))
    reader_0.guest.task_parameters(
        namespace=guest_train_data['namespace'],
        name=guest_train_data['name']
    )
    reader_0.hosts[0].task_parameters(
        namespace=host_train_data['namespace'],
        name=guest_train_data['name']
    )

    reader_1 = Reader("reader_1", runtime_parties=dict(guest=guest, host=host), namespace=test_data["namespace"], name=test_data["name"])

    homo_nn_0 = HomoNN(
        'nn_0',
        runner_conf=conf,
        train_data=reader_0.outputs["output_data"]
    )

    homo_nn_1 = HomoNN(
        'nn_1',
        input_model=homo_nn_0.outputs['output_model'],
        test_data=reader_1.outputs["output_data"]
    )

    evaluation_0 = Evaluation(
        'eval_0',
        default_eval_setting='multi',
        runtime_parties=dict(guest=guest),
        input_datas=[homo_nn_1.outputs['test_output_data'], homo_nn_0.outputs['train_output_data']]
    )

    if config.task_cores:
        pipeline.conf.set("task_cores", config.task_cores)
    if config.timeout:
        pipeline.conf.set("timeout", config.timeout)

    pipeline.add_task(reader_0)
    pipeline.add_task(reader_1)
    pipeline.add_task(homo_nn_0)
    pipeline.add_task(homo_nn_1)
    pipeline.add_task(evaluation_0)

    pipeline.compile()
    pipeline.fit()

    result_summary = parse_summary_result(pipeline.get_task_info("eval_0").get_output_metric()[0]["data"])
    data_summary = {"train": {"guest": guest_train_data["name"], "host": host_train_data["name"]},
                    "test": {"guest": guest_train_data["name"], "host": host_train_data["name"]}
                    }
    
    return data_summary, result_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser("BENCHMARK-QUALITY PIPELINE JOB")
    parser.add_argument("-c", "--config", type=str,
                        help="config file", default="../../config.yaml")
    parser.add_argument("-p", "--param", type=str,
                        help="config file for params", default="")
    args = parser.parse_args()
    main(args.config, args.param)