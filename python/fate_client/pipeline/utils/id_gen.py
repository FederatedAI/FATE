import time
import random
import uuid


def get_uuid():
    return str(uuid.uuid1())


def time_str():
    return time.strftime("%Y%m%d%H%M%S", time.localtime())


def get_session_id(*suffix):
    prefix = [time_str() + str(random.randint(0, 100000))]
    session_parts = prefix + list(suffix)
    return "_".join(session_parts)
