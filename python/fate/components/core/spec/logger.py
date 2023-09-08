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

import logging
import logging.config
import os
from typing import Optional

import pydantic


class LoggerConfig(pydantic.BaseModel):
    config: Optional[dict] = None

    def install(self, debug=False):
        if debug or self.config is None:
            level = os.getenv("DEBUG_MODE_LOG_LEVEL", "DEBUG")
            try:
                import rich.logging

                logging_class = "rich.logging.RichHandler"
                logging_formatters = {}
                handlers = {
                    "console": {
                        "class": logging_class,
                        "level": level,
                        "filters": [],
                    }
                }
            except ImportError:
                logging_class = "logging.StreamHandler"
                logging_formatters = {
                    "console": {
                        "format": "[%(levelname)s][%(asctime)-8s][%(process)s][%(module)s.%(funcName)s][line:%(lineno)d]: %(message)s"
                    }
                }
                handlers = {
                    "console": {
                        "class": logging_class,
                        "level": level,
                        "formatter": "console",
                    }
                }
            self.config = dict(
                version=1,
                formatters=logging_formatters,
                handlers=handlers,
                filters={},
                loggers={},
                root=dict(handlers=["console"], level="DEBUG"),
                disable_existing_loggers=False,
            )
        logging.config.dictConfig(self.config)
