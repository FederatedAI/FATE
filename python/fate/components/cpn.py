#
#  Copyright 2022 The FATE Authors. All Rights Reserved.
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
import pprint
from typing import List, Optional, OrderedDict

from fate.components import Role


class ComponentDeclarError(Exception):
    ...


class ComponentApplyError(Exception):
    ...


logger = logging.getLogger(__name__)


class _Component:
    def __init__(
        self,
        name: str,
        roles,
        provider,
        version,
        description,
        callback,
        parameters: List["_ParameterDeclareClass"],
        artifacts: List["_ArtifactDeclareClass"],
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
        # assert parameters defined once
        _defined = set()
        for p in self.parameters:
            if p.name in _defined:
                raise ComponentDeclarError(f"parameter named `{p.name}` declared multiple times")
            _defined.add(p.name)
        self.artifacts = artifacts

        self.func_args = list(inspect.signature(self.callback).parameters.keys())

        # validate
        if self.func_args[0] != "ctx":
            raise ComponentDeclarError("bad component definition, first argument should be `ctx`")
        if self.func_args[1] != "role":
            raise ComponentDeclarError("bad component definition, second argument should be `role`")
        undeclared_func_parameters = set(self.func_args[2:])
        for parameter in self.parameters:
            if parameter.name not in undeclared_func_parameters:
                raise ComponentDeclarError(
                    f"parameter named `{parameter.name}` declar in decorator, but not found in function's argument"
                )
            undeclared_func_parameters.remove(parameter.name)
        for artifact in self.artifacts:
            if artifact.name not in undeclared_func_parameters:
                raise ComponentDeclarError(
                    f"artifact named `{artifact.name}` declar in decorator, but not found in function's argument"
                )
            undeclared_func_parameters.remove(artifact.name)
        if undeclared_func_parameters:
            raise ComponentDeclarError(
                f"function's arguments `{undeclared_func_parameters}` lack of corresponding decorator"
            )

        self.stage_components: List[_Component] = []

    def validate_and_extract_execute_args(self, role, stage, inputs_artifacts, outputs_artifacts, inputs_parameters):
        from fate.components.loader.artifact import load_artifact

        name_artifact_mapping = {artifact.name: artifact for artifact in self.artifacts}
        name_parameter_mapping = {parameter.name: parameter for parameter in self.parameters}
        execute_args = [role]
        for arg in self.func_args[2:]:
            # arg support to be artifact
            if arti := name_artifact_mapping.get(arg):
                if (arti.stages is None or stage in arti.stages) and ((not (arti.roles)) or role.name in arti.roles):
                    # get corresponding applying config
                    if isinstance(arti, _InputArtifactDeclareClass):
                        artifact_apply = inputs_artifacts.get(arg)
                    elif isinstance(arti, _OutputArtifactDeclareClass):
                        artifact_apply = outputs_artifacts.get(arg)
                    else:
                        artifact_apply = None

                    if artifact_apply is None:
                        if arti.optional:
                            execute_args.append(None)
                        # not found, and not optional
                        else:
                            raise ComponentApplyError(f"artifact `{arg}` required, declare: `{arti}`")
                    # try apply
                    else:
                        try:
                            # annotated metadata drop in inherite, so pass type as argument here
                            # maybe we could find more elegant way some day
                            execute_args.append(load_artifact(artifact_apply, arti.type))
                        except Exception as e:
                            raise ComponentApplyError(
                                f"artifact `{arg}` with applying config `{artifact_apply}` can't apply to `{arti}`"
                            ) from e
                else:
                    execute_args.append(None)

            # arg support to be parameter
            elif parameter := name_parameter_mapping.get(arg):
                parameter_apply = inputs_parameters.get(arg)
                if parameter_apply is None:
                    if not parameter.optional:
                        raise ComponentApplyError(f"parameter `{arg}` required, declare: `{parameter}`")
                    else:
                        execute_args.append(parameter.default)
                else:
                    if type(parameter_apply) != parameter.type:
                        raise ComponentApplyError(
                            f"parameter `{arg}` with applying config `{parameter_apply}` can't apply to `{parameter}`"
                            f": {type(parameter_apply)} != {parameter.type}"
                        )
                    else:
                        execute_args.append(parameter_apply)
            else:
                raise ComponentApplyError(f"should not go here")

        return execute_args

    def execute(self, ctx, *args):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"execution arguments: {pprint.pformat(OrderedDict(zip(self.func_args, [ctx, *args])))}")
        return self.callback(ctx, *args)

    def get_artifacts(self):
        mapping = {artifact.name: artifact for artifact in self.artifacts}
        for stage_cpn in self.stage_components:
            for artifact_name, artifact in stage_cpn.get_artifacts().items():
                # update or merge
                if artifact_name not in mapping:
                    mapping[artifact_name] = artifact
                else:
                    old = mapping[artifact_name]
                    if set(old.roles) != set(artifact.roles):
                        raise ComponentDeclarError(
                            f"artifact {artifact_name} declare multiple times with different roles: `{old.roles}` vs `{artifact.roles}`"
                        )
                    if old.optional != artifact.optional:
                        raise ComponentDeclarError(
                            f"artifact {artifact_name} declare multiple times with different optional: `{old.optional}` vs `{artifact.optional}`"
                        )
                    if old.type != artifact.type:
                        raise ComponentDeclarError(
                            f"artifact {artifact_name} declare multiple times with different optional: `{old.type}` vs `{artifact.type}`"
                        )
                    stages = set(old.stages)
                    stages.update(artifact.stages)
                    old.stages = list(stages)
        return mapping

    def get_parameters(self):
        mapping = {parameter.name: parameter for parameter in self.parameters}
        for stage_cpn in self.stage_components:
            for parameter_name, parameter in stage_cpn.get_parameters().items():
                # update or error
                if parameter_name not in mapping:
                    mapping[parameter_name] = parameter
                else:
                    old = mapping[parameter_name]
                    if set(old.default) != set(artifact.default):
                        raise ComponentDeclarError(
                            f"artifact {parameter_name} declare multiple times with different roles: `{old.default}` vs `{artifact.default}`"
                        )
                    if old.optional != artifact.optional:
                        raise ComponentDeclarError(
                            f"artifact {parameter_name} declare multiple times with different optional: `{old.optional}` vs `{artifact.optional}`"
                        )
                    if old.type != artifact.type:
                        raise ComponentDeclarError(
                            f"artifact {parameter_name} declare multiple times with different optional: `{old.type}` vs `{artifact.type}`"
                        )
        return mapping

    def dict(self):
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
        for artifact_name, artifact in self.get_artifacts().items():
            annotated = getattr(artifact.type, "__metadata__", [None])[0]
            roles = artifact.roles or self.roles
            if annotated == OutputAnnotated:
                output_artifacts[artifact_name] = ArtifactSpec(
                    type=artifact.type.type, optional=artifact.optional, roles=roles, stages=artifact.stages
                )
            elif annotated == InputAnnotated:
                input_artifacts[artifact_name] = ArtifactSpec(
                    type=artifact.type.type, optional=artifact.optional, roles=roles, stages=artifact.stages
                )
            else:
                raise ValueError(f"bad artifact: {artifact}")

        input_parameters = {}
        for parameter_name, parameter in self.get_parameters().items():
            input_parameters[parameter_name] = ParameterSpec(
                type=parameter.type.__name__,
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
    roles = [r.name for r in roles]
    return _component(
        name=name, roles=roles, provider=provider, version=version, description=description, is_subcomponent=False
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
            artifacts.reverse()
            del f.__component_artifacts__
        except AttributeError:
            artifacts = []
        for artifact in artifacts:
            if is_subcomponent:
                artifact.stages = [cpn_name]
            else:
                artifact.stages = [DEFAULT.name]
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
        return cpn

    return decorator


class _ArtifactDeclareClass:
    def __init__(self, name, type, roles, stages, desc, optional) -> None:
        self.name = name
        self.type = type
        self.roles = roles
        self.stages = stages
        self.desc = desc
        self.optional = optional


class _OutputArtifactDeclareClass(_ArtifactDeclareClass):
    def __str__(self) -> str:
        return f"OutputArtifact<name={self.name}, type={self.type}, roles={self.roles}, stages={self.stages}, optional={self.optional}>"


class _InputArtifactDeclareClass(_ArtifactDeclareClass):
    def __str__(self) -> str:
        return f"InputArtifact<name={self.name}, type={self.type}, roles={self.roles}, stages={self.stages}, optional={self.optional}>"


def _create_artifact_declare_class(name, type, roles, desc, optional):
    from fate.components import InputAnnotated, OutputAnnotated

    annotates = getattr(type, "__metadata__", [None])
    if OutputAnnotated in annotates:
        return _OutputArtifactDeclareClass(name, type, roles, [], desc, optional)
    elif InputAnnotated in annotates:
        return _InputArtifactDeclareClass(name, type, roles, [], desc, optional)
    else:
        raise ValueError(f"bad artifact: {name}")


def artifact(name, type, roles: Optional[List[Role]] = None, desc=None, optional=False):
    """attaches an artifact to the component."""
    if roles is None:
        roles = []
    roles = [r.name for r in roles]

    def decorator(f):
        description = desc
        if description is not None:
            description = inspect.cleandoc(description)
        if not hasattr(f, "__component_artifacts__"):
            f.__component_artifacts__ = []
        f.__component_artifacts__.append(_create_artifact_declare_class(name, type, roles, description, optional))
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
