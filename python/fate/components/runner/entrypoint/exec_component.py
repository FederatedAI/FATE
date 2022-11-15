import logging
import time
import traceback
from typing import Optional

from fate.arch.context import Context
from fate.components.entrypoint.uri.uri import URI

from .parser import FATEComponentTaskConfig, TaskExecuteStatus

logger = logging.getLogger(__name__)


class StatusTracker:
    def __init__(self, out_uri, in_uri: Optional[str]) -> None:
        self.out_uri = URI.from_string(out_uri)
        self.in_uri = None if in_uri is None else URI.from_string(in_uri)

    def update_status(self, status: TaskExecuteStatus, extra={}):
        self.out_uri.write_kv("status", dict(status=status.name, extra=extra))

    def safe_to_terminate(self):
        if self.in_uri is not None:
            return self.in_uri.read_kv("status")
        # no input status provided, we have no choice but terminate
        return True


class ComponentExecException(Exception):
    ...


class ParamsValidateFailed(ComponentExecException):
    ...


def task_execute(config: FATEComponentTaskConfig):
    # status tracker
    StatusTracker(config.status_output, config.status_input)

    # init context
    context_name = config.task_id
    computing = get_computing(config)
    federation = get_federation(config, computing)
    context = Context(
        context_name=context_name,
        device=config.extra.device,
        computing=computing,
        federation=federation,
    )

    cpn = get_cpn(config.cpn)
    role_module = cpn.get_role_cpn(config.role)(**config.params)
    logger.info(f"cpn={cpn}, context={context}")
    try:
        logger.info("running...")
        status_tracker.update_status(TaskExecuteStatus.RUNNING)
        try:
            if not cpn.params_validate(config.params):
                raise ParamsValidateFailed()
            run(
                role_module,
                context,
                config.data_inputs,
                config.data_outputs,
                config.model_inputs,
                config.model_outputs,
                config.metrics_output,
            )
        except Exception as e:
            tb = traceback.format_exc()
            status_tracker.update_status(TaskExecuteStatus.FAILED, dict(exception=e.args, traceback=tb))
            raise e
        else:
            status_tracker.update_status(TaskExecuteStatus.SUCCESS)
    except Exception as e:
        raise e
    else:
        logger.info("done without error, waiting signal to terminate")
        while not status_tracker.safe_to_terminate():
            time.sleep(0.5)
        logger.info("terminating, bye~")
    finally:
        logger.info("cleaning...")
        # context.clean()
        logger.info("clean finished")


def run(
    role_cpn,
    ctx,
    data_inputs,
    data_outputs,
    model_inputs,
    model_outputs,
    metrics_output,
):
    # TODO: `hyper routine` such as warmstart/cv/stepwise may wrap here
    # for baisc demo, simply fixed to `train and predict`

    # TODO: extra imformation needed: how many inputs and outputs?
    data_input = URI.from_string(data_inputs[0]).read_df(ctx)
    role_cpn.fit(ctx, data_input)


def get_computing(config: FATEComponentTaskConfig):
    if (engine := config.extra.distributed_computing_backend.engine) == "standalone":
        from fate.arch.computing.standalone import CSession

        return CSession(config.extra.distributed_computing_backend.computing_id)
    elif engine == "eggroll":
        from fate.arch.computing.eggroll import CSession

        return CSession(config.extra.distributed_computing_backend.computing_id)
    elif engine == "spark":
        from fate.arch.computing.spark import CSession

        return CSession(config.extra.distributed_computing_backend.computing_id)

    else:
        raise ParamsValidateFailed(f"extra.distributed_computing_backend.engine={engine} not support")


def get_federation(config: FATEComponentTaskConfig, computing):
    if (engine := config.extra.federation_backend.engine) == "standalone":
        from fate.arch.federation.standalone import StandaloneFederation

        federation_config = config.extra.federation_backend
        return StandaloneFederation(
            computing,
            federation_config.federation_id,
            federation_config.parties.local,
            federation_config.parties.parties,
        )

    else:
        raise ParamsValidateFailed(f"extra.distributed_computing_backend.engine={engine} not support")
