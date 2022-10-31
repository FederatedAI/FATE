import time
import random
import uuid


def get_uuid() -> str:
    return str(uuid.uuid1())


def time_str() -> str:
    return time.strftime("%Y%m%d%H%M%S", time.localtime())


def get_session_id(*suffix) -> str:
    if not suffix:
        return time_str() + str(random.randint(0, 100000))
    else:
        return "_".join(list(suffix))


def get_computing_id(session_id) -> str:
    return "_".join([session_id, "computing"])


def get_federation_id(session_id) -> str:
    return "_".join([session_id, "federation"])
