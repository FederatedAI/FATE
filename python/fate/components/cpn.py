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
from typing import List, OrderedDict


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
    ) -> None:
        import inspect

        self.name = name
        self.roles = roles
        self.provider = provider
        self.version = version
        self.description = description
        self.callback = callback
        self.parameters = parameters
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
        if self.func_args[2] != "stage":
            raise ComponentDeclarError("bad component definition, third argument should be `stage`")
        undeclared_func_parameters = set(self.func_args[3:])
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

    def validate_and_extract_execute_args(self, config):
        role = config.role
        stage = config.stage
        name_artifact_mapping = {artifact.name: artifact for artifact in self.artifacts}
        name_parameter_mapping = {parameter.name: parameter for parameter in self.parameters}
        execute_args = [role, stage]
        for arg in self.func_args[3:]:
            # arg support to be artifact
            if arti := name_artifact_mapping.get(arg):
                if (arti.stages is None or stage in arti.stages) and (arti.roles is None or role in arti.roles):
                    # get corresponding applying config
                    if isinstance(arti, _InputArtifactDeclareClass):
                        artifact_apply = config.inputs.artifacts.get(arg)
                    elif isinstance(arti, _OutputArtifactDeclareClass):
                        artifact_apply = config.outputs.artifacts.get(arg)
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
                            execute_args.append(arti.type.parse_desc(artifact_apply))
                        except Exception as e:
                            raise ComponentApplyError(
                                f"artifact `{arg}` with applying config `{artifact_apply}` can't apply to `{arti}`"
                            ) from e
                else:
                    execute_args.append(None)

            # arg support to be parameter
            elif parameter := name_parameter_mapping.get(arg):
                if not parameter.optional:
                    parameter_apply = config.inputs.parameters.get(arg)
                    if parameter_apply is None:
                        raise ComponentApplyError(f"parameter `{arg}` required, declare: `{parameter}`")
                    else:
                        if type(parameter_apply) != parameter.type:
                            raise ComponentApplyError(
                                f"parameter `{arg}` with applying config `{parameter_apply}` can't apply to `{parameter}`"
                                f": {type(parameter_apply)} != {parameter.type}"
                            )
                        else:
                            execute_args.append(parameter_apply)
                else:
                    execute_args.append(parameter.default)
            else:
                raise ComponentApplyError(f"should not go here")

        return execute_args

    def execute(self, ctx, *args):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"execution arguments: {pprint.pformat(OrderedDict(zip(self.func_args, [ctx, *args])))}")
        return self.callback(ctx, *args)

    def dict(self):
        from fate.components.spec.component import (
            ArtifactSpec,
            ComponentSpec,
            ComponentSpecV1,
            InputDefinitionsSpec,
            OutputDefinitionsSpec,
            ParameterSpec,
        )
        from fate.components.spec.types import InputAnnotated, OutputAnnotated

        input_artifacts = {}
        output_artifacts = {}
        for artifact in self.artifacts:
            annotated = getattr(artifact.type, "__metadata__", [None])[0]
            if annotated == OutputAnnotated:
                output_artifacts[artifact.name] = ArtifactSpec(
                    type=artifact.type.type, optional=artifact.optional, roles=artifact.roles, stages=artifact.stages
                )
            elif annotated == InputAnnotated:
                input_artifacts[artifact.name] = ArtifactSpec(
                    type=artifact.type.type, optional=artifact.optional, roles=artifact.roles, stages=artifact.stages
                )
            else:
                raise ValueError(f"bad artifact: {artifact}")

        input_parameters = {}
        for parameter in self.parameters:
            input_parameters[parameter.name] = ParameterSpec(
                type=parameter.type.__name__, default=parameter.default, optional=parameter.optional
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


def component(name=None, roles=[], provider="fate", version="2.0.0.alpha", description=None):
    r"""Creates a new :class:`Component` and uses the decorated function as
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

    def decorator(f):
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
        desc = description
        if desc is None:
            desc = inspect.getdoc(f)
            if isinstance(desc, bytes):
                desc = desc.decode("utf-8")
        else:
            desc = inspect.cleandoc(desc)
        cpn = _Component(
            name=name or f.__name__.lower(),
            roles=roles,
            provider=provider,
            version=version,
            description=desc,
            callback=f,
            parameters=parameters,
            artifacts=artifacts,
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


def _create_artifact_declare_class(name, type, roles, stages, desc, optional):
    from fate.components.spec.types import InputAnnotated, OutputAnnotated

    annotates = getattr(type, "__metadata__", [None])
    if OutputAnnotated in annotates:
        return _OutputArtifactDeclareClass(name, type, roles, stages, desc, optional)
    elif InputAnnotated in annotates:
        return _InputArtifactDeclareClass(name, type, roles, stages, desc, optional)
    else:
        raise ValueError(f"bad artifact: {name}")


def artifact(name, type, roles=None, stages=None, desc=None, optional=False):
    """attaches an artifact to the component."""

    def decorator(f):
        description = desc
        if description is not None:
            description = inspect.cleandoc(description)
        if not hasattr(f, "__component_artifacts__"):
            f.__component_artifacts__ = []
        f.__component_artifacts__.append(
            _create_artifact_declare_class(name, type, roles, stages, description, optional)
        )
        return f

    return decorator


class _ParameterDeclareClass:
    def __init__(self, name, type, default, optional, desc) -> None:
        self.name = name
        self.type = type
        self.default = default
        self.optional = optional

    def __str__(self) -> str:
        return f"Parameter<name={self.name}, type={self.type}, default={self.default}, optional={self.optional}>"


def parameter(name, type, default=None, optional=False, desc=None):
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
