from fate_arch.abc import AddressABC


class StandaloneAddress(AddressABC):
    def __init__(self, home=None, name=None, namespace=None, storage_type=None):
        self.home = home
        self.name = name
        self.namespace = namespace
        self.storage_type = storage_type

    def __hash__(self):
        return (self.home, self.name, self.namespace, self.storage_type).__hash__()

    def __str__(self):
        return f"StandaloneAddress(name={self.name}, namespace={self.namespace})"

    def __repr__(self):
        return self.__str__()


class EggRollAddress(AddressABC):
    def __init__(self, home=None, name=None, namespace=None):
        self.name = name
        self.namespace = namespace
        self.home = home

    def __hash__(self):
        return (self.home, self.name, self.namespace).__hash__()

    def __str__(self):
        return f"EggRollAddress(name={self.name}, namespace={self.namespace})"

    def __repr__(self):
        return self.__str__()


class HDFSAddress(AddressABC):
    def __init__(self, name_node, path=None):
        self.name_node = name_node
        self.path = path

    def __hash__(self):
        return (self.name_node, self.path).__hash__()

    def __str__(self):
        return f"HDFSAddress(name_node={self.name_node}, path={self.path})"

    def __repr__(self):
        return self.__str__()


class PathAddress(AddressABC):
    def __init__(self, path=None):
        self.path = path

    def __hash__(self):
        return self.path.__hash__()

    def __str__(self):
        return f"PathAddress(path={self.path})"

    def __repr__(self):
        return self.__str__()


class MysqlAddress(AddressABC):
    def __init__(self, user, passwd, host, port, db, name):
        self.user = user
        self.passwd = passwd
        self.host = host
        self.port = port
        self.db = db
        self.name = name

    def __hash__(self):
        return (self.host, self.port, self.db, self.name).__hash__()

    def __str__(self):
        return f"MysqlAddress(db={self.db}, name={self.name})"

    def __repr__(self):
        return self.__str__()


class HiveAddress(AddressABC):
    def __init__(self, host, name, port=10000, username=None, database='default', auth_mechanism='PLAIN', password=None):
        self.host = host
        self.username = username
        self.port = port
        self.database = database
        self.auth_mechanism = auth_mechanism
        self.password = password
        self.name = name

    def __hash__(self):
        return (self.host, self.port, self.database, self.name).__hash__()

    def __str__(self):
        return f"HiveAddress(database={self.database}, name={self.name})"

    def __repr__(self):
        return self.__str__()


class LinkisHiveAddress(AddressABC):
    def __init__(self, host="127.0.0.1", port=9001, username='', database='', name='', run_type='hql',
                 execute_application_name='hive', source={}, params={}):
        self.host = host
        self.port = port
        self.username = username
        self.database = database if database else f"{username}_ind"
        self.name = name
        self.run_type = run_type
        self.execute_application_name = execute_application_name
        self.source=source
        self.params = params

    def __hash__(self):
        return (self.host, self.port, self.database, self.name).__hash__()

    def __str__(self):
        return f"LinkisHiveAddress(database={self.database}, name={self.name})"

    def __repr__(self):
        return self.__str__()


class LocalFSAddress(AddressABC):
    def __init__(self, path):
        self.path = path

    def __hash__(self):
        return (self.path).__hash__()

    def __str__(self):
        return f"LocalFSAddress(path={self.path})"

    def __repr__(self):
        return self.__str__()
