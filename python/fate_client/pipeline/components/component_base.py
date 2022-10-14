import copy
from ..conf.types import SupportRole
from ..utils.id_gen import get_uuid


class Component(object):
    __instance = {}

    def __init__(self, *args, **kwargs):
        if "name" in kwargs:
            self._component_name = kwargs.pop("name")
        self._component_param = kwargs
        self.__party_instance = {}
        self._module_name = None
        self._role = None
        self._index = None

    def __new__(cls, *args, **kwargs):
        if cls.__name__.lower() not in cls.__instance:
            cls.__instance[cls.__name__.lower()] = 0

        new_cls = object.__new__(cls)
        new_cls.set_name(cls.__instance[cls.__name__.lower()])
        cls.__instance[cls.__name__.lower()] += 1

        return new_cls

    def set_name(self, idx):
        self._component_name = self.__class__.__name__.lower() + "_" + str(idx)

    def _set_role(self, role):
        self._role = role

    def _set_index(self, idx):
        self._index = idx

    def __getitem__(self, index) -> "Component":
        if not isinstance(index, (int, list, slice)):
            raise ValueError("Index should be int or list of integer")

        if isinstance(index, slice):
            if index.start is None or index.stop is None:
                raise ValueError(f"Slice {index} is not support, start and stop should be given")
            start = index.start
            stop = index.stop
            step = index.step if index.step else 1 if start < stop else -1
            index = [idx for idx in range(start, stop, step)]
            if len(index) == 1:
                index = index[0]

        index_key = str(index) if isinstance(index, int) else "|".join(map(str, index))

        del self.__party_instance[self._role]["party"][self._index]
        self._set_index(index_key)

        self.__party_instance[self._role]["party"][self._index] = self
        return self

    @property
    def guest(self) -> "Component":
        inst = self.get_party_instance(role=SupportRole.GUEST)
        return inst

    @property
    def host(self) -> "Component":
        inst = self.get_party_instance(role=SupportRole.HOST)
        return inst

    @property
    def arbiter(self) -> "Component":
        inst = self.get_party_instance(role=SupportRole.ARBITER)
        return inst

    def get_party_instance(self, role="guest") -> 'Component':
        if role not in SupportRole.support_roles():
            raise ValueError("Role should be one of guest/host/arbiter")

        if role not in self.__party_instance:
            self.__party_instance[role] = dict()
            self.__party_instance[role]["party"] = dict()

        index = get_uuid()

        inst = copy.deepcopy(self)
        self._decrease_instance_count()

        inst._set_role(role)
        inst._set_index(index)

        self.__party_instance[role]["party"][index] = inst

        return self.__party_instance[role]["party"][index]

    @classmethod
    def _decrease_instance_count(cls):
        cls.__instance[cls.__name__.lower()] -= 1

    @property
    def name(self):
        return self._component_name

    @property
    def module(self):
        return self._module_name

    def component_param(self, **kwargs):
        for attr, val in kwargs:
            self._component_param[attr] = val

    def get_component_param(self):
        return self._component_param

    def get_role_param_conf(self, roles=None):
        role_param_conf = dict()

        if not self.__party_instance:
            return role_param_conf

        for role in self.__party_instance:
            role_param_conf[role] = dict()

            valid_partyids = roles.get(role)
            for party_id in self.__party_instance[role]["party"]:
                if not party_id:
                    continue

                if isinstance(party_id, int):
                    party_key = str(valid_partyids.index(party_id))
                else:
                    party_list = list(map(int, party_id.split("|", -1)))
                    party_list.sort()
                    party_key = "|".join(map(str, [valid_partyids.index(party) for party in party_list]))

                party_inst = self.__party_instance[role]["party"][party_id]

                if party_key not in role_param_conf:
                    role_param_conf[role][party_key] = dict()

                role_param_conf[role][party_key][self._component_name] = party_inst.get_component_param()

        return role_param_conf
