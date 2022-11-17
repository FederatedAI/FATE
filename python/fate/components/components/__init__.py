import logging

from .lr import hetero_lr

BUILDIN_COMPONENTS = {"lr": hetero_lr}

logger = logging.getLogger(__name__)


def load_component(cpn_name: str):

    # from buildin
    if cpn_name in BUILDIN_COMPONENTS:
        return BUILDIN_COMPONENTS[cpn_name]

    # from entrypoint
    import pkg_resources

    for cpn_ep in pkg_resources.iter_entry_points(group="fate.plugins.cpn"):
        try:
            cpn_register = cpn_ep.load()
            cpn_registered_name = cpn_register.registered_name()
        except Exception as e:
            logger.warning(
                f"register cpn from entrypoint(named={cpn_ep.name}, module={cpn_ep.module_name}) failed: {e}"
            )
            continue
        if cpn_registered_name == cpn_name:
            return cpn_register
    raise RuntimeError(f"could not find registerd cpn named `{cpn_name}`")
