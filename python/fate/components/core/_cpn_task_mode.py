import enum
import os


class TaskMode(enum.Enum):
    SIMPLE = "SIMPLE"
    DEEPSPEED = "DEEPSPEED"

    def __str__(self):
        return self.value


def is_deepspeed_mode():
    return os.getenv("FATE_TASK_TYPE", "").upper() == TaskMode.DEEPSPEED.value


def is_root_worker():
    if is_deepspeed_mode():
        return os.getenv("RANK", "0") == "0"
    return True
