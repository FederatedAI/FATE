from typing import Protocol


class Component(Protocol):
    name: str

    @classmethod
    def params_validate(cls, params):
        ...

    @classmethod
    def get_role_cpn(cls, role):
        ...


class ComponentRoleModuleNotFoundError(Exception):
    ...


cpn_dict = dict()


def cpn_register(cls):

    if isinstance(cls, str):
        name = cls

        def _wrap(module):
            cpn_dict[name] = module
            return module

        return _wrap
    else:
        cpn_dict[cls.name] = cls
    return cls
