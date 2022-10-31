class UriTypes(object):
    LOCAL = "file"
    SQL = "sql"
    LMDB = "lmdb"


class SupportRole(object):
    LOCAL = "local"
    GUEST = "guest"
    HOST = "host"
    ARBITER = "arbiter"

    @classmethod
    def support_roles(cls):
        return [cls.LOCAL, cls.GUEST, cls.HOST, cls.ARBITER]


class Backend(object):
    STANDALONE = "standalone"
    EGGROLL = "EGGROLL"
    SPARK = "SPARK"


class LinkKey(object):
    DATA = "data"
    MODEL = "model"
    CACHE = "cache"

    @classmethod
    def get_all_link_keywords(cls):
        return [cls.DATA, cls.MODEL, cls.CACHE]


class InputKey(object):
    def __init__(self):
        self._keys = set()

    def register(self, keys):
        if isinstance(keys, str):
            keys = [keys]

        self._keys = set([key.lower() for key in keys])

        return self

    @property
    def keys(self):
        return self._keys


class InputDataKey(InputKey):
    def __init__(self):
        super(InputDataKey, self).__init__()

    @property
    def data(self):
        if "data" not in self._keys:
            raise ValueError("data input key does not register in this component")
        return "data"

    @property
    def train_data(self):
        if "train_data" not in self._keys:
            raise ValueError("train_data input key does not register in this component")
        return "train_data"

    @property
    def validate_data(self):
        if "validate_data" not in self._keys:
            raise ValueError("validate_data input key does not register in this component")
        return "validate_data"

    @property
    def test_data(self):
        if "test_data" not in self._keys:
            raise ValueError("test_data input key does not register in this component")

        return "test_data"


class InputModelKey(InputKey):
    def __init__(self):
        super(InputModelKey, self).__init__()

    @property
    def model(self):
        if "model" not in self._keys:
            raise ValueError("model input key does not register in this component")

        return "model"

    @property
    def isometric_model(self):
        if "isometric_model" not in self._keys:
            raise ValueError("isometric_model input key does not register in this component")

        return "isometric_model"


class InputCacheKey(InputKey):
    def __init__(self):
        super(InputCacheKey, self).__init__()

    @property
    def cache(self):
        if "cache" not in self._keys:
            raise ValueError("cache input key does not register in this component")

        return "cache"


LOCAL_INPUT = "Standalone::local::input"
