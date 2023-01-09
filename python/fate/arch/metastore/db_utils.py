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
import operator

from ..common.base_utils import current_timestamp
from ..metastore.db_models import DB, StorageConnectorModel


class StorageConnector:
    def __init__(self, connector_name, engine=None, connector_info=None):
        self.name = connector_name
        self.engine = engine
        self.connector_info = connector_info

    @DB.connection_context()
    def create_or_update(self):
        defaults = {
            "f_name": self.name,
            "f_engine": self.engine,
            "f_connector_info": self.connector_info,
            "f_create_time": current_timestamp(),
        }
        connector, status = StorageConnectorModel.get_or_create(f_name=self.name, defaults=defaults)
        if status is False:
            for key in defaults:
                setattr(connector, key, defaults[key])
            connector.save(force_insert=False)

    @DB.connection_context()
    def get_info(self):
        connectors = [
            connector
            for connector in StorageConnectorModel.select().where(
                operator.attrgetter("f_name")(StorageConnectorModel) == self.name
            )
        ]
        if connectors:
            return connectors[0].f_connector_info
        else:
            return {}
