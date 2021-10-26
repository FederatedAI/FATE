

class EngineType(object):
    COMPUTING = "computing"
    STORAGE = "storage"
    FEDERATION = "federation"


class CoordinationProxyService(object):
    ROLLSITE = "rollsite"
    NGINX = "nginx"
    FATEFLOW = "fateflow"


class CoordinationCommunicationProtocol(object):
    HTTP = "http"
    GRPC = "grpc"


class FederatedMode(object):
    SINGLE = "SINGLE"
    MULTIPLE = "MULTIPLE"

    def is_single(self, value):
        return value == self.SINGLE

    def is_multiple(self, value):
        return value == self.MULTIPLE


class FederatedCommunicationType(object):
    PUSH = "PUSH"
    PULL = "PULL"


class BaseType:
    def to_dict(self):
        return dict([(k.lstrip("_"), v) for k, v in self.__dict__.items()])

    def to_dict_with_type(self):
        def _dict(obj):
            module = None
            if issubclass(obj.__class__, BaseType):
                data = {}
                for attr, v in obj.__dict__.items():
                    k = attr.lstrip("_")
                    data[k] = _dict(v)
                module = obj.__module__
            elif isinstance(obj, (list, tuple)):
                data = []
                for i, vv in enumerate(obj):
                    data.append(_dict(vv))
            elif isinstance(obj, dict):
                data = {}
                for _k, vv in obj.items():
                    data[_k] = _dict(vv)
            else:
                data = obj
            return {"type": obj.__class__.__name__, "data": data, "module": module}
        return _dict(self)


class Party(BaseType):
    """
    Uniquely identify
    """

    def __init__(self, role, party_id):
        self.role = str(role)
        self.party_id = str(party_id)

    def __hash__(self):
        return (self.role, self.party_id).__hash__()

    def __str__(self):
        return f"Party(role={self.role}, party_id={self.party_id})"

    def __repr__(self):
        return self.__str__()

    def __lt__(self, other):
        return (self.role, self.party_id) < (other.role, other.party_id)

    def __eq__(self, other):
        return self.party_id == other.party_id and self.role == other.role


class DTable(BaseType):
    def __init__(self, namespace, name, partitions=None):
        self._name = name
        self._namespace = namespace
        self._partitions = partitions

    def __str__(self):
        return f"DTable(namespace={self._namespace}, name={self._name})"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self._namespace == other.namespace and self._name == other.name

    @property
    def name(self):
        return self._name

    @property
    def namespace(self):
        return self._namespace

    @property
    def partitions(self):
        return self._partitions
