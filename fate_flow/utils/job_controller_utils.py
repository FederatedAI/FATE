import time
from threading import Lock

from fate_arch.common import WorkMode
from fate_flow.entity.runtime_config import RuntimeConfig
from fate_flow.entity.constant import JobStatus
from fate_flow.operation.job_saver import JobSaver
from fate_flow.settings import LIMIT_ROLE, MAX_CONCURRENT_JOB_RUN_HOST
from fate_flow.utils import job_utils


def job_quantity_constraint(job_id, role, party_id):
    lock = Lock()
    with lock:
        time.sleep(1)
        if RuntimeConfig.WORK_MODE == WorkMode.CLUSTER:
            if role == LIMIT_ROLE:
                running_jobs = job_utils.query_job(status=JobStatus.RUNNING, role=role)
                ready_jobs = job_utils.query_job(tag='ready', role=role)
                if len(running_jobs)+len(ready_jobs) >= MAX_CONCURRENT_JOB_RUN_HOST:
                    return False
                else:
                    JobSaver.update_job(job_info={"tag": "ready"})
        return True