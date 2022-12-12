import logging
import time
import traceback
from typing import Any, Dict

from fate.arch.context import Context
from fate.components import params
from fate.components.cpn import (
    ComponentApplyError,
    _Component,
    _InputArtifactDeclareClass,
    _OutputArtifactDeclareClass,
)
from fate.components.loader.artifact import load_artifact
from fate.components.loader.component import load_component
from fate.components.loader.computing import load_computing
from fate.components.loader.device import load_device
from fate.components.loader.federation import load_federation
from fate.components.loader.mlmd import MLMD, load_mlmd
from fate.components.loader.other import load_role, load_stage
from fate.components.loader.output import OutputPool, load_pool
from fate.components.spec.task import TaskConfigSpec

logger = logging.getLogger(__name__)


def execute_component(config: TaskConfigSpec):
    taskid = config.taskid
    mlmd = load_mlmd(config.conf.mlmd, taskid)
    output_pool = load_pool(config.conf.output)
    computing = load_computing(config.conf.computing)
    federation = load_federation(config.conf.federation, computing)
    device = load_device(config.conf.device)
    role = load_role(config.role)
    stage = load_stage(config.stage)
    ctx = Context(
        context_name=taskid,
        device=device,
        computing=computing,
        federation=federation,
    )
    logger.debug(f"component={config.component}, context={ctx}")
    try:
        logger.debug("running...")
        mlmd.execution_status.log_excution_start()
        component = load_component(config.component)
        try:
            if not stage.is_default:
                # use sub component to handle stage
                for stage_component in component.stage_components:
                    if stage_component.name == stage.name:
                        component = stage_component
                        break
                else:
                    raise ValueError(f"stage `{stage.name}` for component `{component.name}` not supported")

            # get execute key-word arguments
            execute_kwargs = {}
            # parse and validate parameters
            execute_kwargs.update(parse_input_parameters(mlmd, component, config.inputs.parameters))
            # parse and validate inputs
            execute_kwargs.update(parse_input_artifacts(mlmd, component, stage, role, config.inputs.artifacts))
            # fill in outputs
            execute_kwargs.update(parse_output_artifacts(mlmd, component, stage, role, output_pool))

            # execute
            component.execute(ctx, role, **execute_kwargs)
        except Exception as e:
            tb = traceback.format_exc()
            mlmd.execution_status.log_excution_exception(dict(exception=str(e.args), traceback=tb))
            raise e
        else:
            mlmd.execution_status.log_excution_end()
    except Exception as e:
        logger.error(e, exc_info=True)
        raise e
    else:
        logger.debug("done without error, waiting signal to terminate")
        while not mlmd.execution_status.safe_terminate():
            time.sleep(0.5)
        logger.debug("terminating, bye~")
    finally:
        logger.debug("cleaning...")
        # context.clean()
        logger.debug("clean finished")


def parse_input_parameters(mlmd: MLMD, cpn: _Component, input_parameters: Dict[str, Any]) -> dict:
    execute_parameters = {}
    name_parameter_mapping = {parameter.name: parameter for parameter in cpn.parameters}
    for arg in cpn.func_args[2:]:
        if parameter := name_parameter_mapping.get(arg):
            parameter_apply = input_parameters.get(arg)
            if parameter_apply is None:
                if not parameter.optional:
                    raise ComponentApplyError(f"parameter `{arg}` required, declare: `{parameter}`")
                else:
                    execute_parameters[parameter.name] = parameter.default
                    mlmd.io.log_input_parameter(parameter.name, parameter.default)
            else:
                # TODO: enhance type validate
                try:
                    value = params.parse(parameter.type, parameter_apply)
                except Exception as e:
                    raise ComponentApplyError(f"apply value `{parameter_apply}` to parameter `{arg}` failed:\n{e}")
                execute_parameters[parameter.name] = value
                mlmd.io.log_input_parameter(parameter.name, parameter_apply)
    return execute_parameters


def parse_input_artifacts(mlmd: MLMD, cpn: _Component, stage, role, input_artifacts) -> dict:

    execute_input_artifacts = {}
    name_input_artifacts_mapping = {
        artifact.name: artifact for artifact in cpn.artifacts if isinstance(artifact, _InputArtifactDeclareClass)
    }
    for arg in cpn.func_args[2:]:
        if arti := name_input_artifacts_mapping.get(arg):
            execute_input_artifacts[arg] = None
            if arti.is_active_for(stage, role):
                artifact_apply = input_artifacts.get(arg)
                if artifact_apply is not None:
                    # try apply
                    try:
                        execute_input_artifacts[arg] = load_artifact(artifact_apply, arti.type)
                    except Exception as e:
                        raise ComponentApplyError(
                            f"artifact `{arg}` with applying config `{artifact_apply}` can't apply to `{arti}`"
                        ) from e
                    mlmd.io.log_input_artifact(arg, execute_input_artifacts[arg])
                    continue
                else:
                    if not arti.optional:
                        raise ComponentApplyError(f"artifact `{arg}` required, declare: `{arti}`")
            mlmd.io.log_input_artifact(arg, execute_input_artifacts[arg])
    return execute_input_artifacts


def parse_output_artifacts(mlmd: MLMD, cpn: _Component, stage, role, output_pool: OutputPool) -> dict:

    execute_output_artifacts = {}
    name_output_artifacts_mapping = {
        artifact.name: artifact for artifact in cpn.artifacts if isinstance(artifact, _OutputArtifactDeclareClass)
    }
    for arg in cpn.func_args[2:]:
        if arti := name_output_artifacts_mapping.get(arg):
            execute_output_artifacts[arg] = None
            if arti.is_active_for(stage, role):
                execute_output_artifacts[arg] = output_pool.create_artifact(arti.name, arti.type)
                mlmd.io.log_output_artifact(arg, execute_output_artifacts[arg])
    return execute_output_artifacts
