import logging
import typing
from typing import Dict, List, Union

from fate.components.core.essential import Role, Stage

from ._component import Component
from .artifacts._base_type import AT, ArtifactDescribe, M, _ArtifactsType

if typing.TYPE_CHECKING:
    from fate.arch import Context

    from ..spec.artifact import ArtifactInputApplySpec, ArtifactOutputApplySpec
    from ..spec.task import TaskConfigSpec

logger = logging.getLogger(__name__)


class ComponentExecutionIO:
    class InputPair:
        def __init__(self, artifact, reader):
            self.artifact = artifact
            self.reader = reader

    class OutputPair:
        def __init__(self, artifact, writer):
            self.artifact = artifact
            self.writer = writer

    def __init__(self, ctx: "Context", component: Component, role: Role, stage: Stage, config):
        self.parameter_artifacts_desc = {}
        self.parameter_artifacts_apply = {}
        self.input_data: Dict[str, ComponentExecutionIO.InputPair] = {}
        self.input_model: Dict[str, ComponentExecutionIO.InputPair] = {}
        self.output_data: Dict[str, ComponentExecutionIO.OutputPair] = {}
        self.output_model: Dict[str, ComponentExecutionIO.OutputPair] = {}
        self.output_metric: Dict[str, ComponentExecutionIO.OutputPair] = {}

        logging.debug(f"parse and apply component artifacts")

        for arg in component.func_args[2:]:
            if not (
                self._handle_parameter(component, arg, config)
                or self._handle_input(ctx, component, arg, stage, role, config)
                or self._handle_output(ctx, component, arg, stage, role, config)
            ):
                raise ValueError(f"args `{arg}` not provided")

        self._handle_output(ctx, component, "metric", stage, role, config)

    def _handle_parameter(self, component, arg, config):
        if parameter := component.parameters.mapping.get(arg):
            apply_spec: ArtifactInputApplySpec = config.parameters.get(arg)
            applied_parameter = parameter.apply(apply_spec)
            logging.debug(f"apply parameter `{parameter.name}`: {parameter} -> {applied_parameter}")
            self.parameter_artifacts_apply[parameter.name] = applied_parameter
            return True
        return False

    def _handle_input(self, ctx, component, arg, stage, role, config):
        from fate.arch import URI

        for input_pair_dict, artifacts in [
            (self.input_data, component.artifacts.data_inputs),
            (self.input_model, component.artifacts.model_inputs),
        ]:
            if allow_artifacts := artifacts.get(arg):
                if allow_artifacts.is_active_for(stage, role):
                    apply_spec: Union[
                        ArtifactInputApplySpec, List[ArtifactInputApplySpec]
                    ] = config.input_artifacts.get(arg)
                    if apply_spec is not None:
                        try:
                            if allow_artifacts.is_multi:
                                if not isinstance(apply_spec, list):
                                    raise ComponentArtifactApplyError(
                                        f"`{arg}` expected list of artifact, but single artifact get"
                                    )
                                readers = []
                                for c in apply_spec:
                                    uri = URI.from_string(c.uri)
                                    arti = allow_artifacts.get_correct_arti(c)
                                    readers.append(arti.get_reader(ctx, uri, c.metadata, arti.get_type().type_name))
                                input_pair_dict[arg] = ComponentExecutionIO.InputPair(
                                    artifact=_ArtifactsType([r.artifact for r in readers]), reader=readers
                                )
                            else:
                                uri = URI.from_string(apply_spec.uri)
                                arti = allow_artifacts.get_correct_arti(apply_spec)
                                reader = arti.get_reader(ctx, uri, apply_spec.metadata, arti.get_type().type_name)
                                input_pair_dict[arg] = ComponentExecutionIO.InputPair(
                                    artifact=reader.artifact, reader=reader
                                )
                        except Exception as e:
                            raise ComponentArtifactApplyError(
                                f"load as input artifact({allow_artifacts}) error: {e}"
                            ) from e
                    elif allow_artifacts.optional:
                        input_pair_dict[arg] = ComponentExecutionIO.InputPair(artifact=None, reader=None)
                    else:
                        raise ComponentArtifactApplyError(
                            f"load as input artifact({allow_artifacts}) error: `{arg}` is not optional but None got"
                        )
                    logger.debug(
                        f"apply artifact `{allow_artifacts.name}`: {apply_spec} -> {input_pair_dict[arg].reader}"
                    )
                    return True
                else:
                    logger.debug(f"skip artifact `{allow_artifacts.name}` for stage `{stage}` and role `{role}`")
                    input_pair_dict[arg] = ComponentExecutionIO.InputPair(artifact=None, reader=None)
                    return True
        return False

    def _handle_output(self, ctx, component, arg, stage, role, config):
        from fate.arch import URI

        for output_pair_dict, artifacts in [
            (self.output_data, component.artifacts.data_outputs),
            (self.output_model, component.artifacts.model_outputs),
            (self.output_metric, component.artifacts.metric_outputs),
        ]:

            if allowed_artifacts := artifacts.get(arg):
                if allowed_artifacts.is_active_for(stage, role):
                    apply_spec: ArtifactOutputApplySpec = config.output_artifacts.get(arg)
                    if apply_spec is not None:
                        try:
                            if allowed_artifacts.is_multi:
                                if not apply_spec.is_template():
                                    raise ComponentArtifactApplyError(
                                        "template uri required for multiple output artifact"
                                    )
                                arti = allowed_artifacts.get_correct_arti(apply_spec)
                                writers = WriterGenerator(ctx, arti, apply_spec)
                                output_pair_dict[arg] = ComponentExecutionIO.OutputPair(
                                    artifact=writers.recorder, writer=writers
                                )

                            else:
                                if apply_spec.is_template():
                                    raise ComponentArtifactApplyError(
                                        "template uri is not supported for non-multiple output artifact"
                                    )
                                arti = allowed_artifacts.get_correct_arti(apply_spec)
                                writer = arti.get_writer(
                                    ctx, URI.from_string(apply_spec.uri), arti.get_type().type_name
                                )
                                output_pair_dict[arg] = ComponentExecutionIO.OutputPair(
                                    artifact=writer.artifact, writer=writer
                                )
                        except Exception as e:
                            raise ComponentArtifactApplyError(
                                f"load as output artifact({allowed_artifacts}) error: {e}"
                            ) from e
                    elif allowed_artifacts.optional:
                        output_pair_dict[arg] = ComponentExecutionIO.OutputPair(artifact=None, writer=None)
                    else:
                        raise ComponentArtifactApplyError(
                            f"load as output artifact({allowed_artifacts}) error: apply_config is None but not optional"
                        )
                    logger.debug(
                        f"apply artifact `{allowed_artifacts.name}`: {apply_spec} -> {output_pair_dict[arg].writer}"
                    )
                    return True
                else:
                    logger.debug(f"skip artifact `{allowed_artifacts.name}` for stage `{stage}` and role `{role}`")
                    output_pair_dict[arg] = ComponentExecutionIO.OutputPair(artifact=None, writer=None)
                    return True
        return False

    def get_kwargs(self, with_metrics=False):
        kwargs = {**self.parameter_artifacts_apply}
        kwargs.update({k: v.reader for k, v in self.input_data.items()})
        kwargs.update({k: v.reader for k, v in self.input_model.items()})
        kwargs.update({k: v.writer for k, v in self.output_data.items()})
        kwargs.update({k: v.writer for k, v in self.output_model.items()})
        if with_metrics:
            kwargs.update({k: v.writer for k, v in self.output_metric.items()})
        return kwargs

    def get_metric_writer(self):
        return self.output_metric["metric"].writer

    def dump_io_meta(self, config: "TaskConfigSpec") -> dict:
        from fate.components.core.spec.artifact import IOArtifactMeta

        def _get_meta(d, with_source=False):
            result = {}
            for k, arti_type in d.items():
                if arti_type is not None:
                    if with_source:
                        arti_type.update_source_metadata(config, k)
                    result[k] = arti_type.dict()
            return result

        io_meta = IOArtifactMeta(
            inputs=IOArtifactMeta.InputMeta(
                data=_get_meta({k: v.artifact for k, v in self.input_data.items()}),
                model=_get_meta({k: v.artifact for k, v in self.input_model.items()}),
            ),
            outputs=IOArtifactMeta.OutputMeta(
                data=_get_meta({k: v.artifact for k, v in self.output_data.items()}, with_source=True),
                model=_get_meta({k: v.artifact for k, v in self.output_model.items()}, with_source=True),
                metric=_get_meta({k: v.artifact for k, v in self.output_metric.items()}),
            ),
        )
        return io_meta.dict(exclude_none=True)


class WriterGenerator:
    def __init__(
        self, ctx: "Context", artifact_describe: "ArtifactDescribe[AT, M]", apply_config: "ArtifactOutputApplySpec"
    ):
        self.ctx = ctx
        self.artifact_describe = artifact_describe
        self.apply_config = apply_config

        self.recorder = _ArtifactsType([])
        self.current = 0

    def get_recorder(self):
        return self.recorder

    def __iter__(self):
        return self

    def __next__(self):
        from fate.arch import URI

        uri = URI.from_string(self.apply_config.uri.format(index=self.current))
        writer = self.artifact_describe.get_writer(self.ctx, uri, self.artifact_describe.get_type().type_name)
        self.recorder.artifacts.append(writer.artifact)
        self.current += 1
        return writer

    def __str__(self):
        return f"{self.__class__.__name__}({self.artifact_describe}, index={self.current}>"

    def __repr__(self):
        return str(self)


class ComponentArtifactApplyError(RuntimeError):
    ...
