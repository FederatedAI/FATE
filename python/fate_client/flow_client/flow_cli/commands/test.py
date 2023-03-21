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
import time

import click

from flow_client.flow_cli.utils import cli_args
from flow_client.flow_cli.utils.cli_utils import prettify
from flow_sdk.client import FlowClient

from pipeline.backend.pipeline import PipeLine
from pipeline.component import (
    DataTransform, Evaluation, HeteroLR,
    HeteroSecureBoost, Intersection, Reader,
)
from pipeline.interface import Data


@click.group(short_help="FATE Flow Test Operations")
@click.pass_context
def test(ctx):
    """
    \b
    Provides numbers of component operational commands, including metrics, parameters and etc.
    For more details, please check out the help text.
    """
    pass


@test.command("toy", short_help="Toy Test Command")
@cli_args.GUEST_PARTYID_REQUIRED
@cli_args.HOST_PARTYID_REQUIRED
@cli_args.TIMEOUT
@cli_args.TASK_CORES
@click.pass_context
def toy(ctx, **kwargs):
    flow_sdk = FlowClient(ip=ctx.obj["ip"], port=ctx.obj["http_port"], version=ctx.obj["api_version"],
                          app_key=ctx.obj.get("app_key"), secret_key=ctx.obj.get("secret_key"))
    submit_result = flow_sdk.test.toy(**kwargs)
    if submit_result["retcode"] == 0:
        for t in range(kwargs["timeout"]):
            job_id = submit_result["jobId"]
            r = flow_sdk.job.query(job_id=job_id, role="guest", party_id=kwargs["guest_party_id"])
            if r["retcode"] == 0 and len(r["data"]):
                job_status = r["data"][0]["f_status"]
                print(f"toy test job {job_id} is {job_status}")
                if job_status in {"success", "failed", "canceled"}:
                    check_log(flow_sdk, kwargs["guest_party_id"], job_id, job_status)
                    break
            time.sleep(1)
        else:
            print(f"check job status timeout")
            check_log(flow_sdk, kwargs["guest_party_id"], job_id, job_status)
    else:
        prettify(submit_result)


def check_log(flow_sdk, party_id, job_id, job_status):
    r = flow_sdk.job.log(job_id=job_id, output_path="./logs/toy")
    if r["retcode"] == 0:
        log_msg = flow_sdk.test.check_toy(party_id, job_status, r["directory"])
        try:
            for msg in log_msg:
                print(msg)
        except BaseException:
            print(f"auto check log failed, please check {r['directory']}")
    else:
        print(f"get log failed, please check PROJECT_BASE/logs/{job_id} on the fateflow server machine")


@test.command("min", short_help="Min Test Command")
@click.option("-t", "--data-type", type=click.Choice(["fast", "normal"]), default="fast", show_default=True,
              help="fast for breast data, normal for default credit data")
@click.option("--sbt/--no-sbt", is_flag=True, default=True, show_default=True, help="run sbt test or not")
@cli_args.GUEST_PARTYID_REQUIRED
@cli_args.HOST_PARTYID_REQUIRED
@cli_args.ARBITER_PARTYID_REQUIRED
@click.pass_context
def run_min_test(ctx, data_type, sbt, guest_party_id, host_party_id, arbiter_party_id, **kwargs):
    guest_party_id = int(guest_party_id)
    host_party_id = int(host_party_id)
    arbiter_party_id = int(arbiter_party_id)

    if data_type == "fast":
        guest_train_data = {"name": "breast_hetero_guest", "namespace": "experiment"}
        host_train_data = {"name": "breast_hetero_host", "namespace": "experiment"}
        auc_base = 0.98
    elif data_type == "normal":
        guest_train_data = {"name": "default_credit_hetero_guest", "namespace": "experiment"}
        host_train_data = {"name": "default_credit_hetero_host", "namespace": "experiment"}
        auc_base = 0.69
    else:
        click.echo(f"data type {data_type} not supported", err=True)
        raise click.Abort()

    lr_pipeline = lr_train_pipeline(guest_party_id, host_party_id, arbiter_party_id, guest_train_data, host_train_data)
    lr_auc = get_auc(lr_pipeline, "hetero_lr_0")

    if lr_auc < auc_base:
        click.echo(f"Warning: The LR auc {lr_auc} is lower than expect value {auc_base}")

    predict_pipeline(lr_pipeline, guest_party_id, host_party_id, guest_train_data, host_train_data)

    if sbt:
        sbt_pipeline = sbt_train_pipeline(guest_party_id, host_party_id, guest_train_data, host_train_data)
        sbt_auc = get_auc(sbt_pipeline, "hetero_secureboost_0")

        if sbt_auc < auc_base:
            click.echo(f"Warning: The SBT auc {sbt_auc} is lower than expect value {auc_base}")

        predict_pipeline(sbt_pipeline, guest_party_id, host_party_id, guest_train_data, host_train_data)


def lr_train_pipeline(guest, host, arbiter, guest_train_data, host_train_data):
    pipeline = PipeLine().set_initiator(role="guest", party_id=guest).set_roles(guest=guest, host=host, arbiter=arbiter)

    reader_0 = Reader(name="reader_0")
    reader_0.get_party_instance(role="guest", party_id=guest).component_param(table=guest_train_data)
    reader_0.get_party_instance(role="host", party_id=host).component_param(table=host_train_data)

    data_transform_0 = DataTransform(name="data_transform_0")
    data_transform_0.get_party_instance(role="guest", party_id=guest).component_param(
        with_label=True, output_format="dense")
    data_transform_0.get_party_instance(role="host", party_id=host).component_param(with_label=False)

    intersection_0 = Intersection(name="intersection_0")

    lr_param = {
        "penalty": "L2",
        "tol": 0.0001,
        "alpha": 0.01,
        "optimizer": "rmsprop",
        "batch_size": -1,
        "learning_rate": 0.15,
        "init_param": {
            "init_method": "zeros",
            "fit_intercept": True,
        },
        "max_iter": 30,
        "early_stop": "diff",
        "encrypt_param": {
            "key_length": 1024,
        },
        "cv_param": {
            "n_splits": 5,
            "shuffle": False,
            "random_seed": 103,
            "need_cv": False,
        },
        "validation_freqs": 3,
    }
    hetero_lr_0 = HeteroLR(name="hetero_lr_0", **lr_param)

    evaluation_0 = Evaluation(name="evaluation_0", eval_type="binary")

    pipeline.add_component(reader_0)
    pipeline.add_component(data_transform_0, data=Data(data=reader_0.output.data))
    pipeline.add_component(intersection_0, data=Data(data=data_transform_0.output.data))
    pipeline.add_component(hetero_lr_0, data=Data(train_data=intersection_0.output.data))
    pipeline.add_component(evaluation_0, data=Data(data=hetero_lr_0.output.data))

    pipeline.compile()
    pipeline.fit()

    return pipeline


def sbt_train_pipeline(guest, host, guest_train_data, host_train_data):
    pipeline = PipeLine().set_initiator(role="guest", party_id=guest).set_roles(guest=guest, host=host)

    reader_0 = Reader(name="reader_0")
    reader_0.get_party_instance(role="guest", party_id=guest).component_param(table=guest_train_data)
    reader_0.get_party_instance(role="host", party_id=host).component_param(table=host_train_data)

    data_transform_0 = DataTransform(name="data_transform_0")
    data_transform_0.get_party_instance(role="guest", party_id=guest).component_param(
        with_label=True, output_format="dense")
    data_transform_0.get_party_instance(role="host", party_id=host).component_param(with_label=False)

    intersection_0 = Intersection(name="intersection_0")

    sbt_param = {
        "task_type": "classification",
        "objective_param": {
            "objective": "cross_entropy",
        },
        "num_trees": 3,
        "validation_freqs": 1,
        "encrypt_param": {
            "method": "paillier",
        },
        "tree_param": {
            "max_depth": 3,
        }
    }
    hetero_secure_boost_0 = HeteroSecureBoost(name="hetero_secureboost_0", **sbt_param)

    evaluation_0 = Evaluation(name="evaluation_0", eval_type="binary")

    pipeline.add_component(reader_0)
    pipeline.add_component(data_transform_0, data=Data(data=reader_0.output.data))
    pipeline.add_component(intersection_0, data=Data(data=data_transform_0.output.data))
    pipeline.add_component(hetero_secure_boost_0, data=Data(train_data=intersection_0.output.data))
    pipeline.add_component(evaluation_0, data=Data(data=hetero_secure_boost_0.output.data))

    pipeline.compile()
    pipeline.fit()

    return pipeline


def get_auc(pipeline, component_name):
    cpn_summary = pipeline.get_component(component_name).get_summary()
    auc = cpn_summary.get("validation_metrics").get("train").get("auc")[-1]
    return auc


def predict_pipeline(train_pipeline, guest, host, guest_train_data, host_train_data):
    cpn_list = train_pipeline.get_component_list()[1:]
    train_pipeline.deploy_component(cpn_list)

    pipeline = PipeLine()
    reader_0 = Reader(name="reader_0")
    reader_0.get_party_instance(role="guest", party_id=guest).component_param(table=guest_train_data)
    reader_0.get_party_instance(role="host", party_id=host).component_param(table=host_train_data)
    pipeline.add_component(reader_0)
    pipeline.add_component(train_pipeline, data=Data(predict_input={
        train_pipeline.data_transform_0.input.data: reader_0.output.data}))
    pipeline.predict()
