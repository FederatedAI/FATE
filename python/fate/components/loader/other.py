def load_role(role: str):
    from fate.components import Role

    return Role(role)


def load_stage(stage: str):
    from fate.components import Stage

    return Stage(stage)
