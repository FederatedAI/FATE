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
from typing import Optional

import pydantic


class LoggerConfig(pydantic.BaseModel):
    config: Optional[dict] = None

    def install(self):
        if self.config is None:
            handler_name = "rich_handler"
            self.config = dict(
                version=1,
                formatters={},
                handlers={
                    handler_name: {
                        "class": "rich.logging.RichHandler",
                        "level": "DEBUG",
                        "filters": [],
                    }
                },
                filters={},
                loggers={},
                root=dict(handlers=[handler_name], level="DEBUG"),
                disable_existing_loggers=False,
            )
            logging.config.dictConfig(self.config)
