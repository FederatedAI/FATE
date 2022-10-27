#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

"""
execute with python -m fate.components.runner --session_id xxx --task_config xxx
"""
import logging
from typing import Dict

from fate.components.entrypoint.uri.uri import URI
from sklearn import sys

logger = logging.getLogger(__name__)

import rich

# temporary for pretty debug
import rich.logging

# from rich.traceback import install
# install(show_locals=True, width=None)
# handler = rich.logging.RichHandler(rich_tracebacks=True)
handler = logging.StreamHandler(sys.stderr)
logging.basicConfig(handlers=[handler], level=logging.DEBUG)


def run(config: Dict):
    task_id = config["task_id"]
    task = config["task"]
    task_type = task["type"]
    extra = task.get("task_extra", {})
    task_params = task["task_params"]
    if task_type == "run_component":
        from ..runner.entrypoint.exec_component import (
            FATEComponentTaskConfig,
            task_execute,
        )

        task_config = FATEComponentTaskConfig(
            task_id=task_id, extra=extra, **task_params
        )
        logger.info(f"{task_config=}")
        task_execute(task_config)
    else:
        raise RuntimeError(f"task type `{args.type}` unknown")


if __name__ == "__main__":
    import argparse

    arguments = argparse.ArgumentParser()
    arguments.add_argument("session_id")
    arguments.add_argument("task_config")
    args = arguments.parse_args()
    task_config = run(URI.from_string(args.task_config).read_json())
