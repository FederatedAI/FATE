import time
from threading import Lock

from arch.standalone import WorkMode
from fate_flow.entity.runtime_config import RuntimeConfig
from fate_flow.manager.tracking_manager import Tracking
from fate_flow.settings import LIMIT_ROLE, MAX_CONCURRENT_JOB_RUN_HOST
from fate_flow.utils import job_utils


def job_quantity_constraint(job_id, role, party_id, job_info):
    lock = Lock()
    with lock:
        time.sleep(1)
        if RuntimeConfig.WORK_MODE == WorkMode.CLUSTER:
            if role == LIMIT_ROLE:
                running_jobs = job_utils.query_job(status='running', role=role)
                ready_jobs = job_utils.query_job(tag='ready', role=role)
                if len(running_jobs)+len(ready_jobs) >= MAX_CONCURRENT_JOB_RUN_HOST:
                    return False
                else:
                    tracker = Tracking(job_id=job_id, role=role, party_id=party_id)
                    tracker.save_job_info(role=role, party_id=party_id, job_info={'f_tag': 'ready'})
        return True