#
#  Copyright 2021 The FATE Authors. All Rights Reserved.
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
import kfserving
import os
import subprocess
import tempfile
import torch

from .base import KFServingDeployer

CONFIG_FILE_DIR = "config"
CONFIG_FILE_NAME = "config.properties"
CONFIG_FILE = CONFIG_FILE_DIR + "/" + CONFIG_FILE_NAME

MODEL_STORE_DIR = "model-store"

_ts_config = '''inference_address=http://0.0.0.0:8085
management_address=http://0.0.0.0:8081
metrics_address=http://0.0.0.0:8082
enable_metrics_api=true
metrics_format=prometheus
number_of_netty_threads=4
job_queue_size=10
service_envelope=kfserving
model_store=/mnt/models/model-store
model_snapshot={}'''


def generate_files(working_dir,
                   model_name,
                   torch_model,
                   handler="",
                   min_workers=1,
                   max_workers=5,
                   batch_size=1,
                   max_batch_delay=5000,
                   response_timeout=120):
    """generate mar file and config file

    The defaults are chosen from
    https://github.com/kubeflow/kfserving/blob/v0.5.1/docs/samples/v1beta1/torchserve/config.properties
    """
    torch_script = torch.jit.script(torch_model)
    torch_script_file = os.path.join(working_dir, model_name + ".pt")
    torch_script.save(torch_script_file)
    if not handler:
        handler = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_dummy_base_handler.py")

    # there is no official support of using the model-archiver package via python function calls
    command = ["torch-model-archiver",
               "--serialized-file",
               torch_script_file,
               "--model-name",
               model_name,
               "--export-path",
               working_dir,
               "--handler",
               handler,
               "-v",
               "1.0",
               "-f"]
    ret = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if ret.returncode != 0:
        raise RuntimeError("error running torch-model-archiver: {}, command output: {}".format(ret.args, ret.stdout))
    mar_file = os.path.join(working_dir, model_name + ".mar")

    model_snapshot_dict = {
        "name": "startup.cfg",
        "modelCount": 1,
        "models": {
            model_name: {
                "1.0": {
                    "defaultVersion": True,
                    "marName": os.path.basename(mar_file),
                    "minWorkers": min_workers,
                    "maxWorkers": max_workers,
                    "batchSize": batch_size,
                    "maxBatchDelay": max_batch_delay,
                    "responseTimeout": response_timeout
                }
            }
        }
    }
    config = _ts_config.format(json.dumps(model_snapshot_dict))
    config_file = os.path.join(working_dir, CONFIG_FILE_NAME)
    with open(config_file, "w") as f:
        f.write(config)

    return mar_file, config_file


class TorchServeKFDeployer(KFServingDeployer):
    def _do_prepare_model(self):
        working_dir = tempfile.mkdtemp()
        mar_file, config_file = generate_files(working_dir,
                                               self.service_id,
                                               self.model_object)
        objects = [{"dest": CONFIG_FILE, "obj": config_file},
                   {"dest": MODEL_STORE_DIR + "/" + os.path.basename(mar_file), "obj": mar_file}]
        return objects

    def _do_prepare_predictor(self):
        self.isvc.spec.predictor.pytorch = kfserving.V1beta1TorchServeSpec(storage_uri=self.storage_uri)
