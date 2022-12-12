import time
import random
import uuid


def get_uuid() -> str:
    return str(uuid.uuid1())


def time_str() -> str:
    return time.strftime("%Y%m%d%H%M%S", time.localtime())


def gen_job_id(*suffix) -> str:
    if not suffix:
        return time_str() + str(random.randint(0, 100000))
    else:
        return "_".join(list(suffix))


def gen_computing_id(job_id, task_name, role, party_id) -> str:
    return "_".join([job_id, task_name, role, party_id, "computing"])


def gen_task_id(job_id, task_name, role, party_id) -> str:
    return "_".join([job_id, task_name, role, party_id, "execution"])


def gen_federation_id(job_id, task_name) -> str:
    return "_".join([job_id, task_name, "federation"])
