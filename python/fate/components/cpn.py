#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

# Copyright 2014 Pallets

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:

# 1.  Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.

# 2.  Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in the
#     documentation and/or other materials provided with the distribution.

# 3.  Neither the name of the copyright holder nor the names of its
#     contributors may be used to endorse or promote products derived from
#     this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

"""
use decorators to define component for FATE.
flowing codes modified from [click](https://github.com/pallets/click) project
"""

import inspect
import logging
from typing import Any, Dict, List, Optional

import pydantic
from fate.components import T_ROLE, T_STAGE, MetricArtifact, Role, Stage


class ComponentDeclarError(Exception):
    ...


class ComponentApplyError(Exception):
    ...


logger = logging.getLogger(__name__)


class _Component:
    def __init__(
        self,
        name: str,
        roles: List[T_ROLE],
        provider,
        version,
        description,
        callback,
        parameters: List["_ParameterDeclareClass"],
        artifacts: "_ComponentArtifacts",
        is_subcomponent: bool = False,
    ) -> None:
        self.is_subcomponent = is_subcomponent
        self.name = name
        self.roles = roles
        self.provider = provider
        self.version = version
        self.description = description
        self.callback = callback
        self.parameters = parameters
        if not self.description:
            self.description = ""
        self.artifacts = artifacts
        self.func_args = list(inspect.signature(self.callback).parameters.keys())
        self.stage_components: List[_Component] = []

    def validate_declare(self):
        # validate
        if self.func_args[0] != "ctx":
            raise ComponentDeclarError("bad component definition, first argument should be `ctx`")
        if self.func_args[1] != "role":
            raise ComponentDeclarError("bad component definition, second argument should be `role`")

        # assert parameters defined once
        _defined = set()
        for p in self.parameters:
            if p.name in _defined:
                raise ComponentDeclarError(f"parameter named `{p.name}` declared multiple times")
            _defined.add(p.name)

        # validate func arguments
        undeclared_func_parameters = set(self.func_args[2:])

        def _check_and_remove(name, arg_type):
            if name not in undeclared_func_parameters:
                raise ComponentDeclarError(
                    f"{arg_type} named `{name}` declar in decorator, but not found in function's argument"
                )
            undeclared_func_parameters.remove(name)

        for parameter in self.parameters:
            _check_and_remove(parameter.name, "parameter")
        for name in self.artifacts.get_artifacts():
            _check_and_remove(name, "artifact")
        if undeclared_func_parameters:
            raise ComponentDeclarError(
                f"function's arguments `{undeclared_func_parameters}` lack of corresponding parameter or artifact decorator"
            )

    def execute(self, ctx, role, **kwargs):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"execution arguments: {kwargs}")
        return self.callback(ctx, role, **kwargs)

    def dict(self):
        return self._flatten_stages()._dict()

    def _flatten_stages(self) -> "_Component":
        parameter_mapping = {parameter.name: parameter for parameter in self.parameters}
        merged_artifacts = self.artifacts
        for stage_cpn in self.stage_components:
            stage_cpn = stage_cpn._flatten_stages()
            # merge parameters
            for parameter in stage_cpn.parameters:
                # update or error
                if parameter.name not in parameter_mapping:
                    parameter_mapping[parameter.name] = parameter
                else:
                    parameter_mapping[parameter.name].merge(parameter)
            merged_artifacts = merged_artifacts.merge(stage_cpn.artifacts)

        return _Component(
            name=self.name,
            roles=self.roles,
            provider=self.provider,
            version=self.version,
            description=self.description,
            callback=self.callback,
            parameters=list(parameter_mapping.values()),
            artifacts=merged_artifacts,
            is_subcomponent=self.is_subcomponent,
        )

    def _dict(self):
        from fate.components import InputAnnotated, OutputAnnotated
        from fate.components.spec.component import (
            ArtifactSpec,
            ComponentSpec,
            ComponentSpecV1,
            InputDefinitionsSpec,
            OutputDefinitionsSpec,
            ParameterSpec,
        )

        input_artifacts = {}
        output_artifacts = {}
        for _, artifact in self.artifacts.get_artifacts().items():
            annotated = getattr(artifact.type, "__metadata__", [None])[0]
            roles = artifact.roles or self.roles
            if annotated == OutputAnnotated:
                output_artifacts[artifact.name] = ArtifactSpec(
                    type=artifact.type.type,
                    optional=artifact.optional,
                    roles=roles,
                    stages=artifact.stages,
                    description=artifact.desc,
                )
            elif annotated == InputAnnotated:
                input_artifacts[artifact.name] = ArtifactSpec(
                    type=artifact.type.type,
                    optional=artifact.optional,
                    roles=roles,
                    stages=artifact.stages,
                    description=artifact.desc,
                )
            else:
                raise ValueError(f"bad artifact: {artifact}")

        input_parameters = {}
        from fate.components.params import Parameter

        for parameter in self.parameters:
            if isinstance(parameter.type, Parameter):  # recomanded
                type_name = type(parameter.type).__name__
                type_meta = parameter.type.dict()
            else:
                type_name = parameter.type.__name__
                type_meta = {}

            input_parameters[parameter.name] = ParameterSpec(
                type=type_name,
                type_meta=type_meta,
                default=parameter.default,
                optional=parameter.optional,
                description=parameter.desc,
            )

        input_definition = InputDefinitionsSpec(parameters=input_parameters, artifacts=input_artifacts)
        output_definition = OutputDefinitionsSpec(artifacts=output_artifacts)
        component = ComponentSpec(
            name=self.name,
            description=self.description,
            provider=self.provider,
            version=self.version,
            labels=[],
            roles=self.roles,
            input_definitions=input_definition,
            output_definitions=output_definition,
        )
        return ComponentSpecV1(component=component)

    def dump_yaml(self, stream=None):
        from io import StringIO

        import ruamel.yaml

        spec = self.dict()
        inefficient = False
        if stream is None:
            inefficient = True
            stream = StringIO()
        yaml = ruamel.yaml.YAML()
        yaml.indent(mapping=2, sequence=4, offset=2)
        yaml.dump(spec.dict(), stream=stream)
        if inefficient:
            return stream.getvalue()

    def predict(self, roles=[], provider: Optional[str] = None, version: Optional[str] = None, description=None):
        from fate.components import PREDICT

        return self.stage(roles=roles, name=PREDICT.name, provider=provider, version=version, description=description)

    def train(self, roles=[], provider: Optional[str] = None, version: Optional[str] = None, description=None):
        from fate.components import TRAIN

        return self.stage(roles=roles, name=TRAIN.name, provider=provider, version=version, description=description)

    def stage(
        self, roles=[], name=None, provider: Optional[str] = None, version: Optional[str] = None, description=None
    ):
        r"""Creates a new stage component with :class:`_Component` and uses the decorated function as
        callback.  This will also automatically attach all decorated
        :func:`artifact`\s and :func:`parameter`\s as parameters to the component execution.

        The stage name of the component defaults to the name of the function.
        If you want to change that, you can
        pass the intended name as the first argument.

        Once decorated the function turns into a :class:`Component` instance
        that can be invoked as a component execution.

        :param name: the name of the component.  This defaults to the function
                    name.
        """

        def wrap(f):
            sub_cpn = _component(
                name, roles or self.roles, provider or self.provider, version or self.version, description, True
            )(f)
            self.stage_components.append(sub_cpn)
            return sub_cpn

        return wrap


def component(
    roles: List[Role],
    name: Optional[str] = None,
    provider: Optional[str] = None,
    version: Optional[str] = None,
    description: Optional[str] = None,
):
    r"""Creates a new :class:`_Component` and uses the decorated function as
    callback.  This will also automatically attach all decorated
    :func:`artifact`\s and :func:`parameter`\s as parameters to the component execution.

    The name of the component defaults to the name of the function.
    If you want to change that, you can
    pass the intended name as the first argument.

    Once decorated the function turns into a :class:`Component` instance
    that can be invoked as a component execution.

    :param name: the name of the component.  This defaults to the function
                 name.
    """
    from fate import __provider__, __version__

    if version is None:
        version = __version__
    if provider is None:
        provider = __provider__
    component_roles = [r.name for r in roles]
    return _component(
        name=name,
        roles=component_roles,
        provider=provider,
        version=version,
        description=description,
        is_subcomponent=False,
    )


def _component(name, roles, provider, version, description, is_subcomponent):
    from fate.components import DEFAULT

    def decorator(f):
        cpn_name = name or f.__name__.lower()
        if isinstance(f, _Component):
            raise TypeError("Attempted to convert a callback into a component twice.")
        try:
            parameters = f.__component_parameters__
            parameters.reverse()
            del f.__component_parameters__
        except AttributeError:
            parameters = []
        try:
            artifacts = f.__component_artifacts__
            del f.__component_artifacts__
        except AttributeError:
            artifacts = _ComponentArtifacts()

        if is_subcomponent:
            artifacts.set_stages([cpn_name])
        else:
            artifacts.set_stages([DEFAULT.name])
        desc = description
        if desc is None:
            desc = inspect.getdoc(f)
            if isinstance(desc, bytes):
                desc = desc.decode("utf-8")
        else:
            desc = inspect.cleandoc(desc)
        cpn = _Component(
            name=cpn_name,
            roles=roles,
            provider=provider,
            version=version,
            description=desc,
            callback=f,
            parameters=parameters,
            artifacts=artifacts,
            is_subcomponent=is_subcomponent,
        )
        cpn.__doc__ = f.__doc__
        cpn.validate_declare()
        return cpn

    return decorator


class _ArtifactDeclareClass(pydantic.BaseModel):
    name: str
    type: Any
    roles: List[T_ROLE]
    stages: List[T_STAGE]
    desc: str
    optional: bool

    def is_active_for(self, stage: Stage, role: Role):
        if self.stages is not None and stage.name not in self.stages:
            return False
        if self.roles and role.name not in self.roles:
            return False
        return True

    def __str__(self) -> str:
        return f"ArtifactDeclare<name={self.name}, type={self.type}, roles={self.roles}, stages={self.stages}, optional={self.optional}>"

    def merge(self, a: "_ArtifactDeclareClass"):
        if set(self.roles) != set(a.roles):
            raise ComponentDeclarError(
                f"artifact {self.name} declare multiple times with different roles: `{self.roles}` vs `{a.roles}`"
            )
        if self.optional != a.optional:
            raise ComponentDeclarError(
                f"artifact {self.name} declare multiple times with different optional: `{self.optional}` vs `{a.optional}`"
            )
        if self.type != a.type:
            raise ComponentDeclarError(
                f"artifact {self.name} declare multiple times with different optional: `{self.type}` vs `{a.type}`"
            )
        stages = set(self.stages)
        stages.update(a.stages)
        stages = list(stages)
        return _ArtifactDeclareClass(
            name=self.name, type=self.type, roles=self.roles, stages=stages, desc=self.desc, optional=self.optional
        )


class _ComponentArtifacts(pydantic.BaseModel):
    class Artifacts(pydantic.BaseModel):
        data_artifact: Dict[str, _ArtifactDeclareClass] = pydantic.Field(default_factory=dict)
        model_artifact: Dict[str, _ArtifactDeclareClass] = pydantic.Field(default_factory=dict)
        metric_artifact: Dict[str, _ArtifactDeclareClass] = pydantic.Field(default_factory=dict)

        def add_data(self, artifact):
            self.data_artifact[artifact.name] = artifact

        def add_model(self, artifact):
            self.model_artifact[artifact.name] = artifact

        def add_metric(self, artifact):
            self.metric_artifact[artifact.name] = artifact

        def get_artifact(self, name):
            return self.data_artifact.get(name) or self.model_artifact.get(name) or self.metric_artifact.get(name)

        def merge(self, stage_artifacts):
            def _merge(a, b):
                result = {}
                result.update(a)
                for k, v in b.items():
                    if k not in result:
                        result[k] = v
                    else:
                        result[k] = result[k].merge(v)
                return result

            data_artifact = _merge(self.data_artifact, stage_artifacts.data_artifact)
            model_artifact = _merge(self.model_artifact, stage_artifacts.model_artifact)
            metric_artifact = _merge(self.metric_artifact, stage_artifacts.metric_artifact)
            return _ComponentArtifacts.Artifacts(
                data_artifact=data_artifact, model_artifact=model_artifact, metric_artifact=metric_artifact
            )

    inputs: Artifacts = pydantic.Field(default_factory=Artifacts)
    outputs: Artifacts = pydantic.Field(default_factory=Artifacts)

    def set_stages(self, stages):
        def _set_all(artifacts: Dict[str, _ArtifactDeclareClass]):
            for _, artifact in artifacts.items():
                artifact.stages = stages

        _set_all(self.inputs.data_artifact)
        _set_all(self.inputs.model_artifact)
        _set_all(self.inputs.metric_artifact)
        _set_all(self.outputs.data_artifact)
        _set_all(self.outputs.model_artifact)
        _set_all(self.outputs.metric_artifact)

    def get_artifacts(self) -> Dict[str, _ArtifactDeclareClass]:
        artifacts = {}
        artifacts.update(self.inputs.data_artifact)
        artifacts.update(self.inputs.model_artifact)
        artifacts.update(self.inputs.metric_artifact)
        artifacts.update(self.outputs.data_artifact)
        artifacts.update(self.outputs.model_artifact)
        artifacts.update(self.outputs.metric_artifact)
        return artifacts

    def merge(self, stage_artifacts: "_ComponentArtifacts"):
        return _ComponentArtifacts(
            inputs=self.inputs.merge(stage_artifacts.inputs), outputs=self.outputs.merge(stage_artifacts.outputs)
        )


def artifact(name, type, roles: Optional[List[Role]] = None, desc="", optional=False):
    """attaches an artifact to the component."""
    if roles is None:
        artifact_roles = []
    else:
        artifact_roles = [r.name for r in roles]

    def decorator(f):
        description = desc
        if description:
            description = inspect.cleandoc(description)
        if not hasattr(f, "__component_artifacts__"):
            f.__component_artifacts__ = _ComponentArtifacts()

        from fate.components import (
            DatasetArtifact,
            InputAnnotated,
            ModelArtifact,
            OutputAnnotated,
        )

        annotates = getattr(type, "__metadata__", [None])
        origin_type = getattr(type, "__origin__")
        artifact_dec = _ArtifactDeclareClass(
            name=name, type=type, roles=artifact_roles, stages=[], desc=description, optional=optional
        )
        if InputAnnotated in annotates:
            if issubclass(origin_type, DatasetArtifact):
                f.__component_artifacts__.inputs.add_data(artifact_dec)
            elif issubclass(origin_type, ModelArtifact):
                f.__component_artifacts__.inputs.add_model(artifact_dec)
            elif issubclass(origin_type, MetricArtifact):
                f.__component_artifacts__.inputs.add_metric(artifact_dec)
            else:
                raise ValueError(f"bad artifact, name: `{name}`, type: `{type}`")

        elif OutputAnnotated in annotates:
            if issubclass(origin_type, DatasetArtifact):
                f.__component_artifacts__.outputs.add_data(artifact_dec)
            elif issubclass(origin_type, ModelArtifact):
                f.__component_artifacts__.outputs.add_model(artifact_dec)
            elif issubclass(origin_type, MetricArtifact):
                f.__component_artifacts__.outputs.add_metric(artifact_dec)
            else:
                raise ValueError(f"bad artifact, name: `{name}`, type: `{type}`")
        else:
            raise ValueError(f"bad artifact, name: `{name}`, type: `{type}`")
        return f

    return decorator


class _ParameterDeclareClass:
    def __init__(self, name, type, default, optional, desc) -> None:
        self.name = name
        self.type = type
        self.default = default
        self.optional = optional
        self.desc = desc

    def __str__(self) -> str:
        return f"Parameter<name={self.name}, type={self.type}, default={self.default}, optional={self.optional}>"

    def merge(self, p: "_ParameterDeclareClass"):
        if self.default != p.default:
            raise ComponentDeclarError(
                f"parameter {p.name} declare multiple times with different default: `{self.default}` vs `{p.default}`"
            )
        if self.optional != p.optional:
            raise ComponentDeclarError(
                f"parameter {parameter.name} declare multiple times with different optional: `{self.optional}` vs `{p.optional}`"
            )
        if self.type != p.type:
            raise ComponentDeclarError(
                f"parameter {parameter.name} declare multiple times with different type: `{self.type}` vs `{self.type}`"
            )
        return self


def parameter(name, type, default=None, optional=True, desc=""):
    """attaches an parameter to the component."""

    def decorator(f):
        description = desc
        if description is not None:
            description = inspect.cleandoc(description)
        if not hasattr(f, "__component_parameters__"):
            f.__component_parameters__ = []
        f.__component_parameters__.append(_ParameterDeclareClass(name, type, default, optional, desc))
        return f

    return decorator
