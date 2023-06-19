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
use decorators to define component_desc for FATE.
flowing codes modified from [click](https://github.com/pallets/click) project
"""

import inspect
import logging
from typing import List, Optional

from fate.components.core.essential import (
    CROSS_VALIDATION,
    DEFAULT,
    PREDICT,
    TRAIN,
    Role,
    Stage,
)

from ._component_artifact import ComponentArtifactDescribes
from ._parameter import ComponentParameterDescribes

logger = logging.getLogger(__name__)


class Component:
    def __init__(
        self,
        name: str,
        roles: List[Role],
        provider,
        version,
        description,
        callback,
        parameters: ComponentParameterDescribes,
        artifacts: ComponentArtifactDescribes,
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
        self.stage_components: List[Component] = []

    def validate_declare(self):
        # validate
        if self.func_args[0] != "ctx":
            raise ComponentDeclareError("bad component_desc definition, first argument should be `ctx`")
        if self.func_args[1] != "role":
            raise ComponentDeclareError("bad component_desc definition, second argument should be `role`")

        if set(self.func_args[2:]) != {*self.parameters.mapping.keys(), *self.artifacts.keys()}:
            raise ComponentDeclareError(
                f"bad component_desc definition, function arguments `{self.func_args[2:]}` should be same as {self.parameters.mapping.keys()} and {self.artifacts.keys()}"
            )

    def execute(self, ctx, role, **kwargs):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"execution arguments: {kwargs}")
        return self.callback(ctx, role, **kwargs)

    def dict(self):
        return self._flatten_stages()._dict()

    def _flatten_stages(self) -> "Component":
        merged_parameters = self.parameters
        merged_artifacts = self.artifacts
        for stage_cpn in self.stage_components:
            stage_cpn = stage_cpn._flatten_stages()
            merged_parameters = merged_parameters.merge(stage_cpn.parameters)
            merged_artifacts = merged_artifacts.merge(stage_cpn.artifacts)

        return Component(
            name=self.name,
            roles=self.roles,
            provider=self.provider,
            version=self.version,
            description=self.description,
            callback=self.callback,
            parameters=merged_parameters,
            artifacts=merged_artifacts,
            is_subcomponent=self.is_subcomponent,
        )

    def _dict(self):
        from fate.components.core.spec.component import ComponentSpec, ComponentSpecV1

        return ComponentSpecV1(
            component=ComponentSpec(
                name=self.name,
                description=self.description,
                provider=self.provider,
                version=self.version,
                labels=[],
                roles=self.roles,
                parameters=self.parameters.get_parameters_spec(),
                input_artifacts=self.artifacts.get_inputs_spec(self.roles),
                output_artifacts=self.artifacts.get_outputs_spec(self.roles),
            )
        )

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

    def dump_simplified_yaml(self, stream=None):
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

    def predict(
        self, roles: List = None, provider: Optional[str] = None, version: Optional[str] = None, description=None
    ):

        if roles is None:
            roles = []

        return self.stage(roles=roles, name=PREDICT.name, provider=provider, version=version, description=description)

    def train(
        self, roles: List = None, provider: Optional[str] = None, version: Optional[str] = None, description=None
    ):

        if roles is None:
            roles = []

        return self.stage(roles=roles, name=TRAIN.name, provider=provider, version=version, description=description)

    def cross_validation(
        self, roles: List = None, provider: Optional[str] = None, version: Optional[str] = None, description=None
    ):

        if roles is None:
            roles = []

        return self.stage(
            roles=roles, name=CROSS_VALIDATION.name, provider=provider, version=version, description=description
        )

    def stage(
        self,
        roles: List = None,
        name=None,
        provider: Optional[str] = None,
        version: Optional[str] = None,
        description=None,
    ):
        r"""Creates a new stage component_desc with :class:`_Component` and uses the decorated function as
        callback.  This will also automatically attach all decorated
        :func:`artifact`\s and :func:`parameter`\s as parameters to the component_desc execution.

        The stage name of the component_desc defaults to the name of the function.
        If you want to change that, you can
        pass the intended name as the first argument.

        Once decorated the function turns into a :class:`Component` instance
        that can be invoked as a component_desc execution.
        """
        if roles is None:
            roles = []

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
    :func:`artifact`\s and :func:`parameter`\s as parameters to the component_desc execution.

    The name of the component_desc defaults to the name of the function.
    If you want to change that, you can
    pass the intended name as the first argument.

    Once decorated the function turns into a :class:`Component` instance
    that can be invoked as a component_desc execution.
    """
    from fate import __provider__, __version__

    if version is None:
        version = __version__
    if provider is None:
        provider = __provider__
    return _component(
        name=name,
        roles=roles,
        provider=provider,
        version=version,
        description=description,
        is_subcomponent=False,
    )


def _component(name, roles, provider, version, description, is_subcomponent):
    def decorator(f):
        cpn_name = name or f.__name__.lower()
        if isinstance(f, Component):
            raise TypeError("Attempted to convert a callback into a component_desc twice.")
        try:
            parameters = f.__component_parameters__
            del f.__component_parameters__
        except AttributeError:
            parameters = ComponentParameterDescribes()
        try:
            artifacts = f.__component_artifacts__
            del f.__component_artifacts__
        except AttributeError:
            artifacts = ComponentArtifactDescribes()

        if is_subcomponent:
            artifacts.set_stages([Stage.from_str(cpn_name)])
        else:
            artifacts.set_stages([DEFAULT])
        desc = description
        if desc is None:
            desc = inspect.getdoc(f)
            if isinstance(desc, bytes):
                desc = desc.decode("utf-8")
        else:
            desc = inspect.cleandoc(desc)
        cpn = Component(
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


class ComponentDeclareError(Exception):
    ...
