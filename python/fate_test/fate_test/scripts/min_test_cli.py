#
#  Copyright 2022 The FATE Authors. All Rights Reserved.
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
import os
import time
from datetime import datetime

import click

from fate_test.scripts._options import SharedOptions
from flow_sdk.client import FlowClient


@click.command("min")
@click.option("-t", "--data_type", type=click.Choice(["fast", "normal"]), default="fast",
              help="fast for breast data, normal for default credit data")
@click.option("-gid", "--guest_id", type=click.STRING, required=True, help="guest party id")
@click.option("-hid", "--host_id", type=click.STRING, required=True, help="host party id")
@click.option("-aid", "--arbiter_id", type=click.STRING, required=True, help="arbiter party id")
@click.option("--sbt", type=click.BOOL, default=True, help="test sbt or not")
@click.option("--serving", type=click.BOOL, default=False, help="test fate serving or not")
@SharedOptions.get_shared_options(hidden=True)
@click.pass_context
def run_min_test(ctx, data_type, guest_id, host_id, arbiter_id, sbt, serving, **kwargs):
    ctx.obj.update(**kwargs)
    ctx.obj.post_process()

    if not ctx.obj["yes"] and not click.confirm("running?"):
        return

    flow_ip, http_port = ctx.obj["config"].serving_setting["flow_services"][0]["address"].rsplit(":", 1)
    flow_client = FlowClient(flow_ip, http_port, "v1")

    data_dir = ctx.obj["config"].data_base_dir
    cache_dir = ctx.obj["config"].cache_directory

    task = TrainLRTask(flow_client, data_dir, cache_dir, data_type, guest_id, host_id, arbiter_id)
    task.run(serving)

    if sbt:
        task = TrainSBTTask(flow_client, data_dir, cache_dir, data_type, guest_id, host_id)
        task.run()


class TaskManager:

    # hetero-lr task
    hetero_lr_config_file = "test_hetero_lr_train_job_conf.json"
    hetero_lr_dsl_file = "test_hetero_lr_train_job_dsl.json"

    # hetero-sbt task
    hetero_sbt_config_file = "test_secureboost_train_binary_conf.json"
    hetero_sbt_dsl_file = "test_secureboost_train_dsl.json"

    publish_conf_file = "publish_load_model.json"
    bind_conf_file = "bind_model_service.json"

    predict_task_file = "test_predict_conf.json"

    def __init__(self, flow_client: FlowClient, data_dir: str, cache_dir: str):
        self.flow_client = flow_client

        self.data_dir = data_dir
        self.cache_dir = cache_dir

        self.job_id = datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]

    def load_config_file(self, filename):
        if not filename.startswith("/"):
            filename = f"{self.data_dir}/examples/min_test_task/config/{filename}"

        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_tmp_config(self, config, prefix):
        cache_dir = f"{self.cache_dir}/min_test/{self.job_id}"
        os.makedirs(cache_dir, exist_ok=True)

        filename = f"{cache_dir}/{prefix}.{round(datetime.now().timestamp() * 1000)}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)

        return filename

    def start_block_task(self, cmd, max_waiting_time=7200):
        click.echo(f"Starting block task, cmd is {cmd}")
        start_time = time.time()

        while True:
            stdout = self.flow_client.component.metric_all(
                job_id=cmd.get("job_id"), role="guest", party_id=cmd.get("party_id"),
                component_name=cmd.get("component_name"),
            )

            if stdout:
                break

            waited_time = time.time() - start_time
            if waited_time >= max_waiting_time:
                return None

            click.echo(f"job cmd: component metric_all, waited time: {waited_time}")
            time.sleep(10)

        try:
            json.dumps(stdout)
        except json.decoder.JSONDecodeError:
            click.echo(f"start task error, return value: {stdout}", err=True)
            raise click.Abort()

        return stdout

    def start_block_func(self, run_func, params, exit_func, max_waiting_time=7200):
        start_time = time.time()

        while True:
            result = run_func(*params)
            if exit_func(result):
                return result

            end_time = time.time()
            if end_time - start_time >= max_waiting_time:
                return None

            time.sleep(10)

    def task_status(self, stdout, msg):
        status = stdout.get("retcode")

        if status == 0:
            return status

        click.echo((
            f"start task error, return value: {stdout}" if status is None
            else f"{msg}, status: {status}, stdout: {stdout}"
        ), err=True)
        raise click.Abort()

    def get_table_info(self, name, namespace):
        time_print(f"Start task: table_info")

        stdout = self.flow_client.table.info(namespace=str(namespace), table_name=str(name))
        self.task_status(stdout, "query data info task exec fail")

        return stdout


class TrainTask(TaskManager):

    def __init__(self, flow_client, data_dir, cache_dir, data_type, guest_id, host_id, arbiter_id=0):
        super().__init__(flow_client, data_dir, cache_dir)

        self.method = "all"
        self.guest_id = guest_id
        self.host_id = host_id
        self.arbiter_id = arbiter_id
        self._data_type = data_type
        self.model_id = None
        self.model_version = None
        self.dsl_file = None
        self.train_component_name = None

        if self._data_type == "fast":
            self.task_data_count = 569
            self.task_intersect_count = 569
            self.auc_base = 0.98
            self.guest_table_name = "breast_hetero_guest"
            self.guest_namespace = "experiment"
            self.host_name = "breast_hetero_host"
            self.host_namespace = "experiment"
        elif self._data_type == "normal":
            self.task_data_count = 30000
            self.task_intersect_count = 30000
            self.auc_base = 0.69
            self.guest_table_name = "default_credit_hetero_guest"
            self.guest_namespace = "experiment"
            self.host_name = "default_credit_hetero_host"
            self.host_namespace = "experiment"
        else:
            click.echo(f"Unknown data type: {self._data_type}", err=True)
            raise click.Abort()

    def _make_runtime_conf(self, conf_type="train"):
        pass

    def _check_status(self, job_id):
        pass

    def _deploy_model(self):
        pass

    def run(self, test_serving=False):
        time_print(f"Start task: train job submit")

        runtime_conf_file = self._make_runtime_conf()
        stdout = self.flow_client.job.submit(
            config_data=self.load_config_file(runtime_conf_file),
            dsl_data=self.load_config_file(self.dsl_file),
        )
        self.task_status(stdout, "Train task failed")

        click.echo(json.dumps(stdout, indent=4))

        job_id = stdout.get("jobId")
        self._check_status(job_id)

        auc = self._get_auc(job_id)
        time_print(
            f"Train auc: {auc}" if auc >= self.auc_base
            else f"Warning: Train auc {auc} is lower than expect value {self.auc_base}"
        )

        self.model_id = stdout["data"]["model_info"]["model_id"]
        self.model_version = stdout["data"]["model_info"]["model_version"]

        time.sleep(30)
        self.start_predict_task()

        if test_serving:
            self._load_model()
            self._bind_model()

    def start_predict_task(self):
        self._deploy_model()
        time_print(f"Start task: predict job submit")

        runtime_conf_file = self._make_runtime_conf("predict")
        stdout = self.flow_client.job.submit(config_data=self.load_config_file(runtime_conf_file))
        self.task_status(stdout, "Predict task failed")

        job_id = stdout.get("jobId")
        self._check_status(job_id)

        time_print("Predict task success")

    def _parse_dsl_components(self):
        json_info = self.load_config_file(self.hetero_lr_dsl_file)
        return list(json_info["components"].keys())

    def _check_cpn_status(self, job_id):
        time_print("Start task: job query")

        stdout = self.flow_client.job.query(job_id=job_id)

        try:
            if stdout["retcode"] != 0:
                return "running"

            task_status = stdout["data"][0]["f_status"]
            time_print(f"Current task status: {task_status}")
            return task_status
        except (KeyError, IndexError):
            pass

    def _deploy_model(self):
        time_print("Start task: model deploy")

        stdout = self.flow_client.model.deploy(
            model_id=self.model_id, model_version=self.model_version,
            cpn_list=["reader_0", "data_transform_0", "intersection_0", self.train_component_name],
        )
        self.task_status(stdout, "Deploy model failed")

        time_print(stdout)
        time_print("Deploy model success")

        self.predict_model_id = stdout["data"]["model_id"]
        self.predict_model_version = stdout["data"]["model_version"]

    def _check_exit(self, status):
        return status not in {"running", "start", "waiting"}

    def _get_auc(self, job_id):
        cmd = {"job_id": job_id, "party_id": self.guest_id, "component_name": "evaluation_0"}
        eval_results = self.start_block_task(cmd)
        eval_results = eval_results["data"]["train"][self.train_component_name]["data"]
        time_print(f"Get auc eval results: {eval_results}")

        auc = 0
        for metric_name, metric_value in eval_results:
            if metric_name == "auc":
                auc = metric_value
        return auc

    def _bind_model(self):
        time_print("Start task: model bind")

        config_path = self.__config_bind_load(self.bind_conf_file)
        stdout = self.flow_client.model.bind(config_data=self.load_config_file(config_path))
        self.task_status(stdout, "Bind model failed")

        time_print("Bind model Success")

    def __config_bind_load(self, template):
        json_info = self.load_config_file(template)
        json_info["service_id"] = self.model_id
        json_info["initiator"]["party_id"] = str(self.guest_id)
        json_info["role"]["guest"] = [str(self.guest_id)]
        json_info["role"]["host"] = [str(self.host_id)]
        json_info["role"]["arbiter"] = [str(self.arbiter_id)]
        json_info["job_parameters"]["model_id"] = self.predict_model_id
        json_info["job_parameters"]["model_version"] = self.predict_model_version
        json_info.pop("servings", None)

        config_path = self.save_tmp_config(json_info, "bind_model")
        return config_path

    def _load_model(self):
        time_print("Start task: model load")

        config_path = self.__config_bind_load(self.publish_conf_file)
        stdout = self.flow_client.model.load(config_data=self.load_config_file(config_path))
        status = self.task_status(stdout, "Load model failed")

        try:
            guest_retcode = stdout["data"]["detail"]["guest"][self.guest_id]["retcode"]
            host_retcode = stdout["data"]["detail"]["host"][self.host_id]["retcode"]
        except KeyError:
            guest_retcode = 1
            host_retcode = 1

        if guest_retcode != 0 or host_retcode != 0:
            click.echo(f"Load model failed, status: {status}, stdout: {stdout}", err=True)
            raise click.Abort()

        time_print("Load model Success")


class TrainLRTask(TrainTask):

    def __init__(self, flow_client, data_dir, cache_dir, data_type, guest_id, host_id, arbiter_id):
        super().__init__(flow_client, data_dir, cache_dir, data_type, guest_id, host_id, arbiter_id)

        self.dsl_file = self.hetero_lr_dsl_file
        self.train_component_name = "hetero_lr_0"

    def _make_runtime_conf(self, conf_type="train"):
        json_info = self.load_config_file(self.hetero_lr_config_file if conf_type == "train" else self.predict_task_file)
        json_info["role"]["guest"] = [self.guest_id]
        json_info["role"]["host"] = [self.host_id]
        json_info["role"]["arbiter"] = [self.arbiter_id]

        json_info["initiator"]["party_id"] = self.guest_id

        if self.model_id is not None:
            json_info["job_parameters"]["common"]["model_id"] = self.predict_model_id
            json_info["job_parameters"]["common"]["model_version"] = self.predict_model_version

        table_info = {"name": self.guest_table_name, "namespace": self.guest_namespace}
        json_info["component_parameters"]["role"]["guest"]["0"]["reader_0"]["table"] = table_info
        if conf_type == "train":
            json_info["component_parameters"]["role"]["guest"]["0"]["reader_1"]["table"] = table_info

        table_info = {"name": self.host_name, "namespace": self.host_namespace}
        json_info["component_parameters"]["role"]["host"]["0"]["reader_0"]["table"] = table_info
        if conf_type == "train":
            json_info["component_parameters"]["role"]["host"]["0"]["reader_1"]["table"] = table_info

        config_path = self.save_tmp_config(json_info, "submit_job_lr")
        return config_path

    def _check_status(self, job_id):
        job_status = self.start_block_func(self._check_cpn_status, [job_id], exit_func=self._check_exit)

        if job_status == "failed":
            click.echo(f"Job {job_id} failed", err=True)
            raise click.Abort()


class TrainSBTTask(TrainTask):

    def __init__(self, flow_client, data_dir, cache_dir, data_type, guest_id, host_id):
        super().__init__(flow_client, data_dir, cache_dir, data_type, guest_id, host_id)

        self.dsl_file = self.hetero_sbt_dsl_file
        self.train_component_name = "hetero_secure_boost_0"

    def _make_runtime_conf(self, conf_type="train"):
        json_info = self.load_config_file(self.hetero_sbt_config_file if conf_type == "train" else self.predict_task_file)
        json_info["role"]["guest"] = [self.guest_id]
        json_info["role"]["host"] = [self.host_id]

        if "arbiter" in json_info["role"]:
            del json_info["role"]["arbiter"]

        json_info["initiator"]["party_id"] = self.guest_id

        if self.model_id is not None:
            json_info["job_parameters"]["common"]["model_id"] = self.predict_model_id
            json_info["job_parameters"]["common"]["model_version"] = self.predict_model_version

        table_info = {"name": self.guest_table_name, "namespace": self.guest_namespace}
        json_info["component_parameters"]["role"]["guest"]["0"]["reader_0"]["table"] = table_info
        if conf_type == "train":
            json_info["component_parameters"]["role"]["guest"]["0"]["reader_1"]["table"] = table_info

        table_info = {"name": self.host_name, "namespace": self.host_namespace}
        json_info["component_parameters"]["role"]["host"]["0"]["reader_0"]["table"] = table_info
        if conf_type == "train":
            json_info["component_parameters"]["role"]["host"]["0"]["reader_1"]["table"] = table_info

        config_path = self.save_tmp_config(json_info, "submit_job_sbt")
        return config_path

    def _check_status(self, job_id):
        job_status = self.start_block_func(self._check_cpn_status, [job_id], exit_func=self._check_exit)

        if job_status == "failed":
            click.echo(f"Job {job_id} failed", err=True)
            raise click.Abort()


def time_print(msg):
    click.echo(f"[{time.strftime('%Y-%m-%d %X')}] {msg}")
