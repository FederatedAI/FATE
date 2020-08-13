from collections import Iterable

from fate_arch.storage import StorageEngine
from fate_arch.storage import StorageTableBase


class StorageTable(StorageTableBase):
    def __init__(self,
                 context,
                 address=None,
                 name: str = None,
                 namespace: str = None,
                 partitions: int = 1,
                 data_name: str = ""):
        self._context = context
        self._address = address
        self._name = name
        self._namespace = namespace
        self._partitions = partitions
        self._data_name = data_name
        self._storage_engine = StorageEngine.SIMPLE

    def get_partitions(self):
        return self.get_meta(meta_type='partitions')

    def get_name(self):
        return self._name

    def get_data_name(self):
        return self._data_name

    def get_namespace(self):
        return self._namespace

    def get_storage_engine(self):
        pass

    def get_address(self):
        pass

    def put_all(self, kv_list: Iterable, **kwargs):
        pass

    def count(self):
        return self.get_meta(meta_type='count')

    def save_as(self, name, namespace, partition=None, schema=None, **kwargs):
        pass

    def close(self):
        pass

    def collect(self, **kwargs):
        part_of_data = self.get_meta(meta_type='part_of_data')
        for k_v in part_of_data:
            yield k_v
