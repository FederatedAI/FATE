# def get_cpn(cpn_name: str):

#     # from buildin
#     from .components import cpn_dict

#     if cpn_name in cpn_dict:
#         return cpn_dict[cpn_name]

#     # from entrypoint
#     import pkg_resources

#     for cpn_ep in pkg_resources.iter_entry_points(group="fate.plugins.cpn"):
#         try:
#             cpn_register = cpn_ep.load()
#             cpn_registered_name = cpn_register.registered_name()
#         except Exception as e:
#             logger.warning(
#                 f"register cpn from entrypoint(named={cpn_ep.name}, module={cpn_ep.module_name}) failed: {e}"
#             )
#             continue
#         if cpn_registered_name == cpn_name:
#             return cpn_register
#     raise RuntimeError(f"could not find registerd cpn named `{cpn_name}`")

import logging

logger = logging.getLogger(__name__)


from io import StringIO
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar

from .spec import (
    ArtifactSpec,
    ArtifactType,
    ComponentSpec,
    ComponentSpecV1,
    InputAnnotated,
    InputDefinitionsSpec,
    OutputAnnotated,
    OutputDefinitionsSpec,
    ParameterSpec,
    roles,
)

T = TypeVar("T")


class Cpn:
    def __init__(
        self,
        name: str,
        roles: List[roles],
        provider: str,
        version: str,
        description: str = "",
    ) -> None:
        self.name = name
        self.provider = provider
        self.version = version
        self.description = description
        self.roles = roles

        self._params: Dict[str, ParameterSpec] = {}
        self._artifacts: Dict[str, Tuple[bool, ArtifactSpec]] = {}

    def parameter(
        self,
        name: str,
        type: type,
        default: Any,
        optional: bool,
    ) -> Callable[[T], T]:
        self._params[name] = ParameterSpec(type=type.__name__, default=default, optional=optional)

        def _wrap(func):
            return func

        return _wrap

    def artifact(
        self,
        name: str,
        type: Type[ArtifactType],
        optional=False,
        roles: Optional[List[roles]] = None,
        stages=None,
    ) -> Callable[[T], T]:
        annotated, type_name, *_ = getattr(type, "__metadata__", [None, {}])
        name = type_name if type_name else name
        if annotated == OutputAnnotated:
            self._artifacts[name] = (
                True,
                ArtifactSpec(type=type.type, optional=optional, roles=roles, stages=stages),
            )
        elif annotated == InputAnnotated:
            self._artifacts[name] = (
                False,
                ArtifactSpec(type=type.type, optional=optional, roles=roles, stages=stages),
            )
        else:
            raise ValueError(f"bad type: {type}")

        def _wrap(func):
            return func

        return _wrap

    def get_spec(self):
        input_artifacts = {}
        output_artifacts = {}
        for name, (is_output, artifact) in self._artifacts.items():
            if is_output:
                output_artifacts[name] = artifact
            else:
                input_artifacts[name] = artifact
        input_definition = InputDefinitionsSpec(parameters=self._params, artifacts=input_artifacts)
        output_definition = OutputDefinitionsSpec(artifacts=output_artifacts)
        component = ComponentSpec(
            name=self.name,
            description=self.description,
            provider=self.provider,
            version=self.version,
            labels=[],
            roles=self.roles,
            inputDefinitions=input_definition,
            outputDefinitions=output_definition,
        )
        return ComponentSpecV1(component=component)

    def dump_yaml(self, stream=None):
        import ruamel.yaml

        spec = self.get_spec()
        inefficient = False
        if stream is None:
            inefficient = True
            stream = StringIO()
        yaml = ruamel.yaml.YAML()
        yaml.indent(mapping=2, sequence=4, offset=2)
        yaml.dump(spec.dict(), stream=stream)
        if inefficient:
            return stream.getvalue()
