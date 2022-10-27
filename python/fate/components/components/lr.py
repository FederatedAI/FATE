from ._cpn import ComponentRoleModuleNotFoundError, cpn_register


@cpn_register("lr")
class LRComponent:
    @classmethod
    def params_validate(cls, params):
        return True

    @classmethod
    def get_role_cpn(cls, role):
        from fate.ml.lr.arbiter import LrModuleArbiter
        from fate.ml.lr.guest import LrModuleGuest
        from fate.ml.lr.host import LrModuleHost

        module = dict(
            guest=LrModuleGuest, host=LrModuleHost, arbiter=LrModuleArbiter
        ).get(role)
        if module is None:
            raise ComponentRoleModuleNotFoundError(f"cpn={cls}, role={role}")
        return module
