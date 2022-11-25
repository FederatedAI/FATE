import copy
import multiprocessing
import os
import subprocess
import time

from pathlib import Path
from types import SimpleNamespace
from ..scheduler.runtime_constructor import RuntimeConstructor


def run_subprocess(exec_cmd, std_log_fd):
    process = subprocess.Popen(
        exec_cmd,
        stderr=std_log_fd,
        stdout=std_log_fd
    )
    return process


def run_task_in_party(exec_cmd, std_log_fd):
    process = run_subprocess(exec_cmd, std_log_fd)
    process.communicate()
    process.terminate()
    try:
        os.kill(process.pid, 0)
    except ProcessLookupError:
        pass


def run_detect_task(status_manager, execution_ids):
    while True:
        is_finish = status_manager.monitor_finish_status(execution_ids)
        if is_finish:
            status_manager.record_terminate_status(execution_ids)
            break

        time.sleep(0.5)


def process_task(task_type: str, task_name: str, exec_cmd_prefix: list, runtime_constructor: RuntimeConstructor):
    parties = runtime_constructor.runtime_parties
    task_pools = list()
    status_manager = runtime_constructor.status_manager
    # task_done_tag_paths = list()
    mp_ctx = multiprocessing.get_context("fork")
    std_log_fds = []
    execution_ids = []
    task_infos = []
    for party in parties:
        role = party.role
        party_id = party.party_id

        conf_path = runtime_constructor.task_conf_path(role, party_id)
        execution_id = runtime_constructor.execution_id(role, party_id)
        execution_ids.append(execution_id)
        task_infos.append(SimpleNamespace(execution_id=execution_id, role=role, party_id=party_id))

        log_path = runtime_constructor.log_path(role, party_id)
        std_log_path = Path(log_path).joinpath("std.log").resolve()
        std_log_path.parent.mkdir(parents=True, exist_ok=True)
        std_log_fd = open(std_log_path, "w")
        std_log_fds.append(std_log_fd)

        exec_cmd = copy.deepcopy(exec_cmd_prefix)
        exec_cmd.extend(
            [
                "--process-tag",
                execution_id,
                "--config",
                conf_path
            ]
        )
        task_pools.append(mp_ctx.Process(target=run_task_in_party, kwargs=dict(
            exec_cmd=exec_cmd,
            std_log_fd=std_log_fd
        )))

        task_pools[-1].start()

    detect_task = mp_ctx.Process(target=run_detect_task,
                                 kwargs=dict(status_manager=status_manager,
                                             execution_ids=execution_ids))

    detect_task.start()

    detect_task.join()
    for func in task_pools:
        func.join()

    for std_log_fd in std_log_fds:
        std_log_fd.close()

    return status_manager.get_task_results(task_infos)
