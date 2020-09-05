from fate_arch.abc import AddressABC


class StandaloneAddress(AddressABC):
    def __init__(self, home=None, name=None, namespace=None, storage_type=None):
        self.home = home
        self.name = name
        self.namespace = namespace
        self.storage_type = storage_type


class EggRollAddress(AddressABC):
    def __init__(self, host=None, port=None, home=None, name=None, namespace=None, storage_type=None):
        self.host = host
        self.port = port
        self.name = name
        self.namespace = namespace
        self.storage_type = storage_type
        self.home = home


class HDFSAddress(AddressABC):
    def __init__(self, name_node, path_prefix=None, path=None):
        self.name_node = name_node
        self.path_prefix = path_prefix
        self.path = path


class MysqlAddress(AddressABC):
    def __init__(self, user, passwd, host, port, db, name):
        self.user = user
        self.passwd = passwd
        self.host = host
        self.port = port
        self.db = db
        self.name = name


class FileAddress(AddressABC):
    def __init__(self, path, path_type):
        self.path = path
        self.path_type = path_type
