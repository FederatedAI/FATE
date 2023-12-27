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

import logging.config


def set_up_logging(rank, log_level="DEBUG"):
    if rank < 0:
        message_header = "[[bold green blink] Main [/]]"
    else:
        message_header = f"[[bold green blink]Rank:{rank}[/]]"

    logging.config.dictConfig(
        dict(
            version=1,
            formatters={"with_rank": {"format": f"{message_header} %(message)s", "datefmt": "[%X]"}},
            handlers={
                "base": {
                    "class": "rich.logging.RichHandler",
                    "level": log_level,
                    "filters": [],
                    "formatter": "with_rank",
                    "tracebacks_show_locals": True,
                    "markup": True,
                }
            },
            loggers={},
            root=dict(handlers=["base"], level=log_level),
            disable_existing_loggers=False,
        )
    )
    if rank < 0:
        from rich.traceback import install
        import click

        install(suppress=[click])
