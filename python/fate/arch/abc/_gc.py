import abc


class GarbageCollectionABC(metaclass=abc.ABCMeta):

    def add_gc_action(self, tag: str, obj, method, args_dict):
        ...
