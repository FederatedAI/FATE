import logging

logger = logging.getLogger(__name__)


def load_component(cpn_name: str):
    from fate.components.components import BUILDIN_COMPONENTS
    from fate.components.cpn import _Component

    # from buildin
    for cpn in BUILDIN_COMPONENTS:
        if cpn.name == cpn_name:
            return cpn

    # from entrypoint
    import pkg_resources

    for cpn_ep in pkg_resources.iter_entry_points(group="fate.ext.component"):
        try:
            candidate_cpn: _Component = cpn_ep.load()
            candidate_cpn_name = candidate_cpn.name
        except Exception as e:
            logger.warning(
                f"register cpn from entrypoint(named={cpn_ep.name}, module={cpn_ep.module_name}) failed: {e}"
            )
            continue
        if candidate_cpn_name == cpn_name:
            return candidate_cpn
    raise RuntimeError(f"could not find registerd cpn named `{cpn_name}`")


def list_components():
    import pkg_resources
    from fate.components.components import BUILDIN_COMPONENTS

    buildin_components = [c.name for c in BUILDIN_COMPONENTS]
    third_parties_components = []

    for cpn_ep in pkg_resources.iter_entry_points(group="fate.ext.component"):
        try:
            candidate_cpn = cpn_ep.load()
            candidate_cpn_name = candidate_cpn.name
            third_parties_components.append([candidate_cpn_name])
        except Exception as e:
            logger.warning(
                f"register cpn from entrypoint(named={cpn_ep.name}, module={cpn_ep.module_name}) failed: {e}"
            )
            continue
    return dict(buildin=buildin_components, thirdparty=third_parties_components)
