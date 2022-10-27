import logging
import time
import traceback
from enum import Enum
from typing import Dict, List, Literal, Optional, Tuple

import pydantic
from fate.arch.context import Context
from fate.arch.unify import device
from fate.components.cpn import get_cpn
from fate.components.entrypoint.uri.uri import URI

logger = logging.getLogger(__name__)


class TaskExecuteStatus(Enum):
    RUNNING = "running"
    FAILED = "failed"
    SUCCESS = "success"


class FATECpnExtra(pydantic.BaseModel):
    class DistributedComputingBackend(pydantic.BaseModel):
        engine: str
        computing_id: str

    class FederationBackend(pydantic.BaseModel):
        class Parties(pydantic.BaseModel):
            local: Tuple[Literal["guest", "host", "arbiter"], str]
            parties: List[Tuple[Literal["guest", "host", "arbiter"], str]]

        engine: str
        federation_id: str
        parties: Parties

    device: device
    distributed_computing_backend: DistributedComputingBackend
    federation_backend: FederationBackend

    @pydantic.validator("device", pre=True)
    def _device_validate(cls, v):
        if not isinstance(v, str):
            raise ValueError("must be str")
        for dev in device:
            if dev.name == v.strip().upper():
                return dev
        raise ValueError(f"should be one of {[dev.name for dev in device]}")


class FATEComponentTaskConfig(pydantic.BaseModel):
    task_id: str
    cpn: str
    role: str
    params: Dict
    data_inputs: List[str]
    data_outputs: List[str]
    model_inputs: List[str]
    model_outputs: List[str]
    status_output: str
    extra: FATECpnExtra
    metrics_output: Optional[str] = None
    status_input: Optional[str] = None


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
    status_tracker = StatusTracker(config.status_output, config.status_input)
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
            status_tracker.update_status(
                TaskExecuteStatus.FAILED, dict(exception=e.args, traceback=tb)
            )
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
    from fate.arch.computing.standalone import CSession

    return CSession(config.extra.distributed_computing_backend.computing_id)


def get_federation(config: FATEComponentTaskConfig, computing):
    from fate.arch.federation.standalone import StandaloneFederation

    federation_config = config.extra.federation_backend
    return StandaloneFederation(
        computing,
        federation_config.federation_id,
        federation_config.parties.local,
        federation_config.parties.parties,
    )
