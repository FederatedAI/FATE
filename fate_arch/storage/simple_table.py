from collections import Iterable

from fate_arch.abc import TableABC


class SimpleTable(TableABC):
    def __init__(self, name, namespace, data_name, **kwargs):
        self._name = name,
        self._namespace = namespace
        self.data_name = data_name

    def get_partitions(self):
        return self.get_meta(_type='partitions')

    def get_name(self):
        return self._name

    def get_data_name(self):
        return self.data_name

    def get_namespace(self):
        return self._namespace

    def get_storage_engine(self):
        pass

    def get_address(self):
        pass

    def put_all(self, kv_list: Iterable, **kwargs):
        pass

    def count(self):
        return self.get_meta(_type='count')

    def save_as(self, name, namespace, partition=None, schema=None, **kwargs):
        pass

    def close(self):
        pass

    def collect(self, **kwargs):
        part_of_data = self.get_meta(_type='part_of_data')
        for k_v in part_of_data:
            yield k_v
