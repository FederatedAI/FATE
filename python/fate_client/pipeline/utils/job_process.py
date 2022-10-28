import copy
import json
import multiprocessing
import os
import subprocess
import time

from ..entity.runtime_entity import FateStandaloneRuntimeEntity
from pathlib import Path
from types import SimpleNamespace


def run_subprocess(exec_cmd, std_log_path):
    std = open(std_log_path, "w")
    process = subprocess.Popen(
        exec_cmd,
        stderr=std,
        stdout=std
    )
    return process


def run_task_in_party(exec_cmd, std_log_path, status_manager, status_uri):
    process = run_subprocess(exec_cmd, std_log_path)
    process.communicate()
    try:
        os.kill(process.pid, 0)
    except ProcessLookupError:
        pass
    finally:
        status_manager.record_finish_status(status_uri)


def run_detect_task(status_manager, status_uris):
    while True:
        is_finish = status_manager.monitor_status(status_uris)
        if is_finish:
            break

        time.sleep(0.1)


def process_task(task_type: str, exec_cmd_prefix: list, runtime_entity: FateStandaloneRuntimeEntity, log_dir: Path):
    role_party_list = runtime_entity.runtime_role_with_party
    status_manager = runtime_entity.status_manager

    task_pools = list()
    task_status_uris = list()
    # task_done_tag_paths = list()
    for role_party_obj in role_party_list:
        role = role_party_obj.role
        party_id = role_party_obj.party_id
        status_uri = runtime_entity.get_status_output_uri(role, party_id)
        # done_tag_path = str(Path(status_uri.parent.joinpath(f"{task_type}.done")))
        # task_done_tag_paths.append(done_tag_path)
        task_status_uris.append(
            SimpleNamespace(
                role=role,
                party_id=party_id,
                status_uri=status_uri
            )
        )
        conf_path = runtime_entity.get_job_conf_uri(role, party_id)
        log_path = log_dir.joinpath(role).joinpath(party_id)
        log_path.mkdir(parents=True, exist_ok=True)
        log_path = str(log_path.joinpath("std.log"))
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
            status_manager=status_manager,
            status_uri=status_uri
        )))

        task_pools[-1].start()

    detect_task = multiprocessing.Process(target=run_detect_task,
                                          kwargs=dict(status_manager=status_manager,
                                                      status_uris=task_status_uris))

    detect_task.start()

    for func in task_pools:
        func.join()

    detect_task.join()

    return status_manager.get_tasks_status(task_status_uris)


