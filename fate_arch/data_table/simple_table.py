from collections import Iterable

from fate_arch.abc import TableABC


class SimpleTable(TableABC):
    def __init__(self, name, namespace, data_name, **kwargs):
        self._name = name,
        self._namespace = namespace
        self.data_name = data_name

    def get_partitions(self):
        return self.get_schema(_type='partitions')

    def get_name(self):
        pass

    def get_data_name(self):
        return self.data_name

    def get_namespace(self):
        pass

    def get_storage_engine(self):
        pass

    def get_address(self):
        pass

    def put_all(self, kv_list: Iterable, **kwargs):
        pass

    def count(self):
        return self.get_schema(_type='count')

    def save_as(self, name, namespace, partition=None, schema_data=None, **kwargs):
        pass

    def close(self):
        pass

    def collect(self, **kwargs):
        data = self.get_schema(_type='data')
        for k_v in data:
            yield k_v
