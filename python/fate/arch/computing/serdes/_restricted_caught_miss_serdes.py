import io
import pickle

from ruamel import yaml

from ._restricted_serdes import RestrictedUnpickler


def get_restricted_catch_miss_serdes():
    return WhitelistCatchRestrictedSerdes


class WhitelistCatchRestrictedSerdes:
    @classmethod
    def serialize(cls, obj) -> bytes:
        return pickle.dumps(obj)

    @classmethod
    def deserialize(cls, bytes) -> object:
        return RestrictedCatchUnpickler(io.BytesIO(bytes)).load()


class RestrictedCatchUnpickler(RestrictedUnpickler):
    caught_miss = {}

    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except pickle.UnpicklingError:
            if (module, name) not in self.caught_miss:
                if module not in self.caught_miss:
                    self.caught_miss[module] = set()
                self.caught_miss[module].add(name)
            return self._load(module, name)

    @classmethod
    def dump_miss(cls, path):
        with open(path, "w") as f:
            yaml.dump({module: list(names) for module, names in cls.caught_miss.items()}, f)


def dump_miss(path):
    RestrictedCatchUnpickler.dump_miss(path)

