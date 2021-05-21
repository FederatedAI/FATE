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
    def __init__(self, name_node, path=None):
        self.name_node = name_node
        self.path = path


class PathAddress(AddressABC):
    def __init__(self, path=None):
        self.path = path


class MysqlAddress(AddressABC):
    def __init__(self, user, passwd, host, port, db, name):
        self.user = user
        self.passwd = passwd
        self.host = host
        self.port = port
        self.db = db
        self.name = name


class HiveAddress(AddressABC):
    def __init__(self, host, name, port=10000, username=None, database='default', auth='NONE', configuration=None,
                 kerberos_service_name=None, password=None):
        self.host = host
        self.username = username
        self.port = port
        self.database = database
        self.auth = auth
        self.configuration = configuration
        self.kerberos_service_name = kerberos_service_name
        self.password=password
        self.name = name


class FileAddress(AddressABC):
    def __init__(self, path, path_type):
        self.path = path
        self.path_type = path_type
