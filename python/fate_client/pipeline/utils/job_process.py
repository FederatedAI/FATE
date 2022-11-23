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


def run_task_in_party(exec_cmd, std_log_fd, status_manager, status_uri):
    process = run_subprocess(exec_cmd, std_log_fd)
    process.communicate()
    process.terminate()
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


def process_task(task_type: str, task_name: str, exec_cmd_prefix: list, runtime_constructor: RuntimeConstructor):
    parties = runtime_constructor.runtime_parties
    task_pools = list()
    task_status_uris = list()
    status_manager = runtime_constructor.status_manager
    # task_done_tag_paths = list()
    mp_ctx = multiprocessing.get_context("fork")
    std_log_fds = []
    for party in parties:
        role = party.role
        party_id = party.party_id

        # TODO: mlmd should be optimized later
        mlmd = runtime_constructor.mlmd(role, party_id)
        status_uri = mlmd.metadata["state_path"]
        terminate_status_uri = mlmd.metadata["terminate_state_path"]

        conf_path = runtime_constructor.task_conf_uri(role, party_id)
        execution_id = runtime_constructor.execution_id(role, party_id)
        std_log_path = Path(status_uri).parent.joinpath("std.log").resolve()
        std_log_path.parent.mkdir(parents=True, exist_ok=True)
        std_log_fd = open(std_log_path, "w")
        std_log_fds.append(std_log_fd)

        done_status_path = str(Path(status_uri).parent.joinpath("done").resolve())

        exec_cmd = copy.deepcopy(exec_cmd_prefix)
        exec_cmd.extend(
            [
                "--execution_id",
                execution_id,
                "--config",
                conf_path
            ]
        )
        task_pools.append(mp_ctx.Process(target=run_task_in_party, kwargs=dict(
            exec_cmd=exec_cmd,
            std_log_fd=std_log_fd,
            status_manager=status_manager,
            status_uri=done_status_path
        )))

        task_status_uris.append(
            SimpleNamespace(
                role=role,
                party_id=party_id,
                status_uri=done_status_path,
                task_terminate_status_uri=terminate_status_uri
            )
        )

        task_pools[-1].start()

    detect_task = mp_ctx.Process(target=run_detect_task,
                                 kwargs=dict(status_manager=status_manager,
                                             status_uris=task_status_uris))

    detect_task.start()

    for func in task_pools:
        func.join()

    detect_task.join()

    for std_log_fd in std_log_fds:
        std_log_fd.close()

    return status_manager.get_tasks_status(task_status_uris)
