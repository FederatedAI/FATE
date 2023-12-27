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
import typing

from fate.arch.computing.api import ComputingEngine
from fate.arch.config import cfg


class ComputingBuilder:
    def __init__(
        self,
        computing_session_id: str,
    ):
        self._computing_session_id = computing_session_id

    def build(self, t: ComputingEngine, conf: dict):
        if t == ComputingEngine.STANDALONE:
            data_dir = cfg.get_option(conf, "computing.standalone.data_dir")
            options = cfg.get_option(conf, "computing.standalone.options", None)
            return self.build_standalone(data_dir=data_dir, options=options)
        elif t == ComputingEngine.EGGROLL:
            host = cfg.get_option(conf, "computing.eggroll.host")
            port = cfg.get_option(conf, "computing.eggroll.port")
            options = cfg.get_option(conf, "computing.eggroll.options")
            config = cfg.get_option(conf, "computing.eggroll.config")
            config_options = cfg.get_option(conf, "computing.eggroll.config_options")
            config_properties_file = cfg.get_option(conf, "computing.eggroll.config_properties_file")
            return self.build_eggroll(
                host=host,
                port=port,
                options=options,
                config=config,
                config_options=config_options,
                config_properties_file=config_properties_file,
            )
        elif t == ComputingEngine.SPARK:
            return self.build_spark()
        else:
            raise ValueError(f"computing engine={t} not support")

    def build_standalone(self, data_dir: typing.Optional[str], options=None, logger_config=None):
        from fate.arch.computing.backends.standalone import CSession

        return CSession(
            session_id=self._computing_session_id,
            data_dir=data_dir,
            logger_config=logger_config,
            options=options,
        )

    def build_eggroll(
        self, host: str, port: int, options: dict, config=None, config_options=None, config_properties_file=None
    ):
        from fate.arch.computing.backends.eggroll import CSession

        return CSession(
            session_id=self._computing_session_id,
            host=host,
            port=port,
            options=options,
            config=config,
            config_options=config_options,
            config_properties_file=config_properties_file,
        )

    def build_spark(self):
        from fate.arch.computing.backends.spark import CSession

        return CSession(self._computing_session_id)
