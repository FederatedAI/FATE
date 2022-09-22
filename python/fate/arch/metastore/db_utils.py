import operator

from fate_arch.common.base_utils import current_timestamp
from fate_arch.metastore.db_models import DB, StorageConnectorModel


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
        connector, status = StorageConnectorModel.get_or_create(
            f_name=self.name,
            defaults=defaults)
        if status is False:
            for key in defaults:
                setattr(connector, key, defaults[key])
            connector.save(force_insert=False)

    @DB.connection_context()
    def get_info(self):
        connectors = [connector for connector in StorageConnectorModel.select().where(
            operator.attrgetter("f_name")(StorageConnectorModel) == self.name)]
        if connectors:
            return connectors[0].f_connector_info
        else:
            return {}
