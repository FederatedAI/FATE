from fate_arch import storage


def get_table_info(name, namespace):
    data_table_meta = storage.StorageTableMeta(name=name, namespace=namespace)
    address = data_table_meta.get_address()
    schema = data_table_meta.get_schema()
    return address, schema
