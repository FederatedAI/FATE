from fate_arch.abc import AddressABC
from fate_arch.metastore.db_utils import StorageConnector


class AddressBase(AddressABC):
    def __init__(self, connector_name=None):
        self.connector_name = connector_name
        if connector_name:
            connector = StorageConnector(connector_name=connector_name)
            if connector.get_info():
                for k, v in connector.get_info().items():
                    if hasattr(self, k) and v:
                        self.__setattr__(k, v)

    @property
    def connector(self):
        return {}

    @property
    def storage_engine(self):
        return


class StandaloneAddress(AddressBase):
    def __init__(self, home=None, name=None, namespace=None, storage_type=None, connector_name=None):
        self.home = home
        self.name = name
        self.namespace = namespace
        self.storage_type = storage_type
        super(StandaloneAddress, self).__init__(connector_name=connector_name)

    def __hash__(self):
        return (self.home, self.name, self.namespace, self.storage_type).__hash__()

    def __str__(self):
        return f"StandaloneAddress(name={self.name}, namespace={self.namespace})"

    def __repr__(self):
        return self.__str__()

    @property
    def connector(self):
        return {"home": self.home}


class EggRollAddress(AddressBase):
    def __init__(self, home=None, name=None, namespace=None, connector_name=None):
        self.name = name
        self.namespace = namespace
        self.home = home
        super(EggRollAddress, self).__init__(connector_name=connector_name)

    def __hash__(self):
        return (self.home, self.name, self.namespace).__hash__()

    def __str__(self):
        return f"EggRollAddress(name={self.name}, namespace={self.namespace})"

    def __repr__(self):
        return self.__str__()

    @property
    def connector(self):
        return {"home": self.home}


class HDFSAddress(AddressBase):
    def __init__(self, name_node=None, path=None, connector_name=None):
        self.name_node = name_node
        self.path = path
        super(HDFSAddress, self).__init__(connector_name=connector_name)

    def __hash__(self):
        return (self.name_node, self.path).__hash__()

    def __str__(self):
        return f"HDFSAddress(name_node={self.name_node}, path={self.path})"

    def __repr__(self):
        return self.__str__()

    @property
    def connector(self):
        return {"name_node": self.name_node}


class PathAddress(AddressBase):
    def __init__(self, path=None, connector_name=None):
        self.path = path
        super(PathAddress, self).__init__(connector_name=connector_name)

    def __hash__(self):
        return self.path.__hash__()

    def __str__(self):
        return f"PathAddress(path={self.path})"

    def __repr__(self):
        return self.__str__()


class MysqlAddress(AddressBase):
    def __init__(self, user=None, passwd=None, host=None, port=None, db=None, name=None, connector_name=None):
        self.user = user
        self.passwd = passwd
        self.host = host
        self.port = port
        self.db = db
        self.name = name
        self.connector_name = connector_name
        super(MysqlAddress, self).__init__(connector_name=connector_name)

    def __hash__(self):
        return (self.host, self.port, self.db, self.name).__hash__()

    def __str__(self):
        return f"MysqlAddress(db={self.db}, name={self.name})"

    def __repr__(self):
        return self.__str__()

    @property
    def connector(self):
        return {"user": self.user, "passwd": self.passwd, "host": self.host, "port": self.port, "db": self.db}


class HiveAddress(AddressBase):
    def __init__(self, host=None, name=None, port=10000, username=None, database='default', auth_mechanism='PLAIN',
                 password=None, connector_name=None):
        self.host = host
        self.username = username
        self.port = port
        self.database = database
        self.auth_mechanism = auth_mechanism
        self.password = password
        self.name = name
        super(HiveAddress, self).__init__(connector_name=connector_name)

    def __hash__(self):
        return (self.host, self.port, self.database, self.name).__hash__()

    def __str__(self):
        return f"HiveAddress(database={self.database}, name={self.name})"

    def __repr__(self):
        return self.__str__()

    @property
    def connector(self):
        return {
            "host": self.host,
            "port": self.port,
            "username": self.username,
            "password": self.password,
            "auth_mechanism": self.auth_mechanism,
            "database": self.database}


class LinkisHiveAddress(AddressBase):
    def __init__(self, host="127.0.0.1", port=9001, username='', database='', name='', run_type='hql',
                 execute_application_name='hive', source={}, params={}, connector_name=None):
        self.host = host
        self.port = port
        self.username = username
        self.database = database if database else f"{username}_ind"
        self.name = name
        self.run_type = run_type
        self.execute_application_name = execute_application_name
        self.source = source
        self.params = params
        super(LinkisHiveAddress, self).__init__(connector_name=connector_name)

    def __hash__(self):
        return (self.host, self.port, self.database, self.name).__hash__()

    def __str__(self):
        return f"LinkisHiveAddress(database={self.database}, name={self.name})"

    def __repr__(self):
        return self.__str__()


class LocalFSAddress(AddressBase):
    def __init__(self, path=None, connector_name=None):
        self.path = path
        super(LocalFSAddress, self).__init__(connector_name=connector_name)

    def __hash__(self):
        return (self.path).__hash__()

    def __str__(self):
        return f"LocalFSAddress(path={self.path})"

    def __repr__(self):
        return self.__str__()
