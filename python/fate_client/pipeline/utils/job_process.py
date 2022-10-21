import copy
import json
import multiprocessing
import os
import subprocess
import time

from ..entity.runtime_entity import FateStandaloneRuntimeEntity
from .file_utils import create_parent_dir
from pathlib import Path
from types import SimpleNamespace


def run_subprocess(exec_cmd, std_log_path):
    create_parent_dir(std_log_path)
    std = open(std_log_path, "w")
    process = subprocess.Popen(
        exec_cmd,
        stderr=std,
        stdout=std
    )
    return process


def run_task_in_party(exec_cmd, std_log_path, done_tag_path):
    process = run_subprocess(exec_cmd, std_log_path)
    process.communicate()
    try:
        os.kill(process.pid, 0)
    except ProcessLookupError:
        pass
    finally:
        create_parent_dir(done_tag_path)
        with open(done_tag_path, "w") as fout:
            fout.write("task done")


def run_detect_task(task_done_tag_paths: list):
    while True:
        finish = True
        for file in task_done_tag_paths:
            if not os.path.exists(file):
                finish = False
                break

        if not finish:
            time.sleep(1)
            continue

        break


def process_task(task_type: str, exec_cmd_prefix: list, runtime_entity: FateStandaloneRuntimeEntity):
    role_party_list = runtime_entity.runtime_role_with_party

    task_pools = list()
    task_status_paths = list()
    task_done_tag_paths = list()
    for role_party_obj in role_party_list:
        role = role_party_obj.role
        party_id = role_party_obj.party_id
        status_path = runtime_entity.get_status_output_path(role, party_id)[7:]
        done_tag_path = str(Path(status_path).parent.joinpath(f"{task_type}.done"))
        task_done_tag_paths.append(done_tag_path)
        task_status_paths.append(
            SimpleNamespace(
                role=role,
                party_id=party_id,
                status_path=status_path
            )
        )
        conf_path = runtime_entity.get_job_conf_path(role, party_id)
        log_path = str(Path(runtime_entity.get_log_path(role, party_id)).joinpath("std.log").resolve())
        exec_cmd = copy.deepcopy(exec_cmd_prefix)
        exec_cmd.extend(
            [
                "--task-type",
                task_type,
                "--address",
                conf_path
            ]
        )
        task_pools.append(multiprocessing.Process(target=run_task_in_party, kwargs=dict(
            exec_cmd=exec_cmd,
            std_log_path=log_path,
            done_tag_path=done_tag_path
        )))

        task_pools[-1].start()

    detect_task = multiprocessing.Process(target=run_detect_task,
                                          kwargs=dict(task_done_tag_paths=task_done_tag_paths))

    detect_task.start()

    for func in task_pools:
        func.join()

    detect_task.join()

    return get_task_status(task_status_paths)


def get_task_status(task_status_paths):
    ret_code = 0
    summary_msg = dict()
    for obj in task_status_paths:
        try:
            with open(obj.status_path, "r") as fin:
                party_status = json.loads(fin.read())
                if party_status["retcode"]:
                    ret_code = 10000
        except FileNotFoundError:
            ret_code = 10001
            party_status = dict(retcode=10001,
                                retmsg="can not start task")

        if obj.role not in summary_msg:
            summary_msg[obj.role] = dict()
        summary_msg[obj.role][obj.party_id] = party_status

    ret = dict(retcode=ret_code,
               retmsg=summary_msg)

    return ret
