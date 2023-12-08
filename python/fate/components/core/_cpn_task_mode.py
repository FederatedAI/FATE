import enum
import os


class TaskMode(enum.StrEnum):
    SIMPLE = "SIMPLE"
    DEEPSPEED = "DEEPSPEED"


def is_deepspeed_mode():
    return os.getenv("FATE_TASK_TYPE", "").upper() == TaskMode.DEEPSPEED


def is_root_worker():
    if is_deepspeed_mode():
        return os.getenv("RANK", "0") == "0"
    return True
