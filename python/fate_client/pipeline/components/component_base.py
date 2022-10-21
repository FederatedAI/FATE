import copy
from ..conf.types import SupportRole
from ..utils.id_gen import get_uuid


class Component(object):
    __instance = {}

    def __init__(self, *args, **kwargs):
        if "name" in kwargs:
            self._component_name = kwargs.pop("name")
        self._component_param = kwargs
        self._support_roles = None
        self.__party_instance = {}
        self._module_name = None
        self._role = None
        self._index = None
        self._callable = True

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

    def _set_callable(self, status):
        self._callable = status

    def callable(self):
        return self._callable

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

        if isinstance(index, list):
            index.sort()
        index_key = str(index) if isinstance(index, int) else "|".join(map(str, index))

        # del self.__party_instance[self._role]["party"][self._index]
        self._set_index(index_key)
        self._set_callable(True)

        self.__party_instance[self._index] = copy.deepcopy(self)
        return self.__party_instance[self._index]

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

        index = get_uuid()

        inst = copy.deepcopy(self)
        inst.party_instance = {}
        self._decrease_instance_count()

        inst._set_role(role)
        inst._set_index(index)
        inst._set_callable(False)

        self.__party_instance[role][index] = inst
        return inst

    def get_support_roles(self):
        return self._support_roles

    @classmethod
    def _decrease_instance_count(cls):
        cls.__instance[cls.__name__.lower()] -= 1

    @property
    def party_instance(self):
        return self.__party_instance

    @party_instance.setter
    def party_instance(self, party_instance):
        self.__party_instance = party_instance

    @property
    def name(self):
        return self._component_name

    @property
    def module(self):
        return self._module_name

    @property
    def support_roles(self):
        return self._support_roles

    def component_param(self, **kwargs):
        for attr, val in kwargs:
            self._component_param[attr] = val

    def get_component_param(self):
        return self._component_param

    def get_role_param(self, role, index):
        component_param = self._component_param
        if role not in self.__party_instance:
            return component_param

        index = str(index)

        role_inst_dict = self.__party_instance[role]

        for _, inst in role_inst_dict.items():
            for party_index, party_inst in inst.party_instance.items():
                party_index = party_index.split("|")
                if index not in party_index:
                    continue

                component_param.update(party_inst.get_component_param())

        return component_param

    def validate_runtime_env(self, roles):
        runtime_roles = roles.get_runtime_roles()
        for role, role_inst in self.__party_instance.items():
            if role not in runtime_roles:
                raise ValueError(f"role {role} does not set in pipeline")
            runtime_role_parties = roles.get_party_list_by_role(role)

            mx_idx = 0
            for _, inst in role_inst.items():
                for party_key in inst.party_instance:
                    parties = map(int, party_key.split("|", -1))
                    mx_idx = max(mx_idx, max(parties))

            if mx_idx >= len(runtime_role_parties):
                raise ValueError(f"role {role}, index {mx_idx} out of bound")
