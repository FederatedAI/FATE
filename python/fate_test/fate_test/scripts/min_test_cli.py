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

import os
from time import time

import click

from fate_test.scripts._options import SharedOptions
from flow_sdk.client import FlowClient


@click.command("min")
@click.option("-gid", "--guest-party-id", type=click.STRING, required=True, help="guest party id")
@click.option("-hid", "--host-party-id", type=click.STRING, required=True, help="host party id")
@click.option("--task-cores", type=click.INT, default=2, help="job running cores, default 2 cores")
@click.option("--timeout", type=click.INT, default=300, help="job running timeout, default 300 seconds")
@SharedOptions.get_shared_options(hidden=True)
@click.pass_context
def run_min_test(ctx, guest_party_id, host_party_id, task_cores, timeout, **kwargs):
    ctx.obj.update(**kwargs)
    ctx.obj.post_process()

    if not ctx.obj["yes"] and not click.confirm("running?"):
        return

    ip, http_port = ctx.obj["config"].serving_setting["flow_services"][0]["address"].rsplit(":", 1)
    flow_client = FlowClient(ip, http_port, "v1")

    submit_result = flow_client.test.toy(guest_party_id, host_party_id, task_cores=task_cores, timeout=timeout)

    if submit_result["retcode"] != 0:
        print(submit_result)
        return

    job_id = submit_result["jobId"]
    log_dir = os.path.join(ctx.obj["config"].fate_base, "logs", "toy")

    for _ in range(timeout):
        r = flow_client.job.query(job_id, "guest", guest_party_id)
        if r["retcode"] != 0 or not r["data"]:
            time.sleep(1)
            continue

        job_status = r["data"][0]["f_status"]
        print(f"min test job {job_id} is {job_status}")
        if job_status in {"success", "failed", "canceled"}:
            check_log(flow_client, log_dir, guest_party_id, job_id, job_status)
            break
    else:
        print(f"check job status timeout")
        check_log(flow_client, log_dir, guest_party_id, job_id, job_status)


def check_log(flow_sdk, log_dir, party_id, job_id, job_status):
    r = flow_sdk.job.log(job_id, log_dir)
    if r["retcode"] != 0:
        print(f"get log failed, please check {os.path.join(log_dir, job_id)} on the fateflow server machine")
        return

    log_msg = flow_sdk.test.check_toy(party_id, job_status, r["directory"])
    try:
        for msg in log_msg:
            print(msg)
    except BaseException:
        print(f"auto check log failed, please check {r['directory']}")
