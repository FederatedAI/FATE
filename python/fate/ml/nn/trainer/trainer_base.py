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

import os
import re
import torch
import torch.distributed as dist
import math
import sys
from torch import nn
from typing import Any, Dict, List, Tuple, Union, Callable
from enum import Enum
from fate.arch import Context
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from transformers import TrainingArguments as _hf_TrainingArguments, PreTrainedTokenizer
from transformers import Trainer, EvalPrediction
from transformers.trainer_utils import has_length
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import _utils
from fate.ml.aggregator.base import Aggregator
import logging
from transformers import logging as transformers_logging
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from typing import Optional
from dataclasses import dataclass, field, fields
from fate.ml.aggregator import AggregatorType
from fate.ml.nn.model_zoo.hetero_nn_model import HeteroNNModelGuest, HeteroNNModelHost
from transformers.trainer import logger as logger_
from fate.ml.evaluation.metric_base import MetricEnsemble
from transformers import IntervalStrategy, DefaultFlowCallback


# Reset the logger to redirect logs output
transformers_logging.disable_default_handler()
transformers_logging.enable_propagation()
logger = logging.getLogger(__name__)


def get_ith_checkpoint(directory, i):
    # List all files in the directory
    files = os.listdir(directory)

    # Filter for checkpoint directories
    checkpoint_dirs = [f for f in files if f.startswith("checkpoint-")]

    # Extract the numbers from the checkpoint directory names
    checkpoint_numbers = [int(re.search(r"\d+", dir).group()) for dir in checkpoint_dirs]

    # Pair the checkpoint directories with their numbers and sort by the
    # numbers
    sorted_checkpoints = sorted(zip(checkpoint_dirs, checkpoint_numbers), key=lambda x: x[1])

    if i < 0:
        raise ValueError(f"checkpoint idx i must be greater than or equal to 0, got {i}")
    if i > len(sorted_checkpoints) - 1:
        raise ValueError(f"checkpoint number is {len(sorted_checkpoints)}, but got {i}")
    # Return the name of the ith checkpoint directory
    return sorted_checkpoints[i][0]


"""
Fed Arguments
"""


class AggregateStrategy(Enum):
    EPOCH = "epoch"
    STEP = "steps"


@dataclass
class FedArguments(object):
    """
    The argument for Fed algorithm
    """

    aggregate_strategy: str = field(default=AggregateStrategy.EPOCH.value)
    aggregate_freq: int = field(default=1)
    aggregator: str = field(default=AggregatorType.SECURE_AGGREGATE.value)

    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support). It obfuscates
        the token values by removing their value.
        """
        # filter out fields that are defined as field(init=False)
        d = dict((field.name, getattr(self, field.name)) for field in fields(self) if field.init)

        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
            if k.endswith("_token"):
                d[k] = f"<{k.upper()}>"
        return d


@dataclass
class _TrainingArguments(_hf_TrainingArguments):
    # in fate-2.0, we will control the output dir when using pipeline
    output_dir: str = field(default="./")
    disable_tqdm: bool = field(default=True)
    save_strategy: str = field(default="no")
    logging_strategy: str = field(default="epoch")
    logging_steps: int = field(default=1)
    evaluation_strategy: str = field(default="no")
    logging_dir: str = field(default=None)
    checkpoint_idx: int = field(default=None)
    # by default, we use constant learning rate, the same as FATE-1.X
    lr_scheduler_type: str = field(default="constant")
    log_level: str = field(default="info")
    deepspeed: Optional[str] = field(default=None)
    save_safetensors: bool = field(default=False)
    use_cpu: bool = field(default=True)

    def __post_init__(self):
        self.push_to_hub = False
        self.hub_model_id = None
        self.hub_strategy = "every_save"
        self.hub_token = None
        self.hub_private_repo = False
        self.push_to_hub_model_id = None
        self.push_to_hub_organization = None
        self.push_to_hub_token = None

        super().__post_init__()


@dataclass
class TrainingArguments(_TrainingArguments):
    # To simplify the to dict result(to_dict only return non-default args)

    def to_dict(self):
        # Call the superclass's to_dict method
        all_args = super().to_dict()
        # Get a dict with default values for all fields
        default_args = _TrainingArguments().to_dict()
        # Filter out args that are equal to their default values
        set_args = {name: value for name, value in all_args.items() if value != default_args.get(name)}
        return set_args


"""
Fed Callback Related Classes
"""


class ShortcutCallBackInterFace(object):
    def __init__(self) -> None:
        pass

    def on_init_end(
        self,
        ctx: Context,
        aggregator: Aggregator,
        fed_args: FedArguments,
        args: TrainingArguments,
        model: Optional[nn.Module] = None,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        dataloader: Optional[Tuple[DataLoader]] = None,
        control: Optional[TrainerControl] = None,
        state: Optional[TrainerState] = None,
        **kwargs,
    ):
        pass

    def on_train_begin(
        self,
        ctx: Context,
        aggregator: Aggregator,
        fed_args: FedArguments,
        args: TrainingArguments,
        model: Optional[nn.Module] = None,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        dataloader: Optional[Tuple[DataLoader]] = None,
        control: Optional[TrainerControl] = None,
        state: Optional[TrainerState] = None,
        **kwargs,
    ):
        pass

    def on_train_end(
        self,
        ctx: Context,
        aggregator: Aggregator,
        fed_args: FedArguments,
        args: TrainingArguments,
        model: Optional[nn.Module] = None,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        dataloader: Optional[Tuple[DataLoader]] = None,
        control: Optional[TrainerControl] = None,
        state: Optional[TrainerState] = None,
        **kwargs,
    ):
        pass

    def on_epoch_begin(
        self,
        ctx: Context,
        aggregator: Aggregator,
        fed_args: FedArguments,
        args: TrainingArguments,
        model: Optional[nn.Module] = None,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        dataloader: Optional[Tuple[DataLoader]] = None,
        control: Optional[TrainerControl] = None,
        state: Optional[TrainerState] = None,
        **kwargs,
    ):
        pass

    def on_epoch_end(
        self,
        ctx: Context,
        aggregator: Aggregator,
        fed_args: FedArguments,
        args: TrainingArguments,
        model: Optional[nn.Module] = None,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        dataloader: Optional[Tuple[DataLoader]] = None,
        control: Optional[TrainerControl] = None,
        state: Optional[TrainerState] = None,
        **kwargs,
    ):
        pass

    def on_step_begin(
        self,
        ctx: Context,
        aggregator: Aggregator,
        fed_args: FedArguments,
        args: TrainingArguments,
        model: Optional[nn.Module] = None,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        dataloader: Optional[Tuple[DataLoader]] = None,
        control: Optional[TrainerControl] = None,
        state: Optional[TrainerState] = None,
        **kwargs,
    ):
        pass

    def on_step_end(
        self,
        ctx: Context,
        aggregator: Aggregator,
        fed_args: FedArguments,
        args: TrainingArguments,
        model: Optional[nn.Module] = None,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        dataloader: Optional[Tuple[DataLoader]] = None,
        control: Optional[TrainerControl] = None,
        state: Optional[TrainerState] = None,
        **kwargs,
    ):
        pass


class FedCallbackInterface(object):
    def __init__(self):
        pass

    def on_federation(
        self,
        ctx: Context,
        aggregator: Aggregator,
        fed_args: FedArguments,
        args: TrainingArguments,
        model: Optional[nn.Module] = None,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        dataloader: Optional[Tuple[DataLoader]] = None,
        control: Optional[TrainerControl] = None,
        state: Optional[TrainerState] = None,
        **kwargs,
    ):
        pass

    def init_aggregator(self, fed_arg: FedArguments):
        raise NotImplementedError("init_aggregator() must be implemented in subclass, init aggregator here")


# I dont like huggingface logging
class LogSuppressFilter(logging.Filter):
    def filter(self, record):
        suppress_list = {"\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n"}
        if record.getMessage() in suppress_list:
            return False
        return True


logger_.addFilter(LogSuppressFilter())


def compute_max_aggregation(
    fed_args: FedArguments, max_epoch: int, max_steps: int, epochs_trained: int, steps_trained: int
) -> int:
    assert (
        max_epoch > epochs_trained and max_epoch > 0
    ), "max_epoch must be greater than epochs_trained: {} and greater than 0".format(epochs_trained)
    assert (
        max_steps > steps_trained and max_steps > 0
    ), "max_steps must be greater than steps_trained: {} and greater than 0".format(steps_trained)

    if isinstance(fed_args.aggregate_freq, float) and fed_args.aggregate_freq < 1 and fed_args.aggregate_freq > 0:
        if fed_args.aggregate_strategy == AggregateStrategy.EPOCH.value:
            aggregate_freq = int(max_epoch / int(1 / fed_args.aggregate_freq))
        elif fed_args.aggregate_strategy == AggregateStrategy.STEP.value:
            aggregate_freq = int(max_steps / int(1 / fed_args.aggregate_freq))

    elif isinstance(fed_args.aggregate_freq, int) and fed_args.aggregate_freq > 0:
        aggregate_freq = fed_args.aggregate_freq
    else:
        raise ValueError("aggregate_freq must be a positive integer or a float between 0 and 1")

    if fed_args.aggregate_strategy == AggregateStrategy.EPOCH.value:
        max_aggregation = int((max_epoch - epochs_trained) / aggregate_freq)
    elif fed_args.aggregate_strategy == AggregateStrategy.STEP.value:
        max_aggregation = int((max_steps - steps_trained) / aggregate_freq)
    else:
        raise ValueError('aggregate_strategy must be either "epoch" or "steps"')

    return max_aggregation, aggregate_freq


def can_aggregate_loss(args: TrainingArguments, fed_args: FedArguments):
    # check if logging strategy is aligned with aggregation strategy
    # make sure that the aggregated loss is correct
    can_aggregate = False
    if args.logging_strategy == fed_args.aggregate_strategy:
        if fed_args.aggregate_strategy == "epoch":
            if args.logging_steps == fed_args.aggregate_freq:
                can_aggregate = True

    if not can_aggregate:
        logger.warning(
            "Not able to aggregate loss, "
            'Trainer is only able to aggregate loss on server when aggregate strategy and logging strategy are "epoch",'
            " and aggregate_freq and logging_freq are the same"
        )

    return can_aggregate


class AggregationChecker:
    def __init__(
        self,
        fed_args,
        max_aggregation,
        aggregate_freq,
        max_epoch: int,
        max_steps: int,
        epochs_trained: int,
        steps_trained: int,
        can_aggregate_loss: bool,
    ):
        self.fed_args = fed_args
        self.max_epoch = max_epoch
        self.max_steps = max_steps
        self.epochs_trained = epochs_trained
        self.steps_trained = steps_trained
        self.model_aggregation_count = 0
        self.loss_aggregation_count = 0
        self.aggregate_freq = aggregate_freq
        self.max_aggregation = max_aggregation
        self.can_aggregate_loss = can_aggregate_loss

    def report(self):
        logger.info(f"Aggregation count: {self.model_aggregation_count} / {self.max_aggregation}")

    def should_aggregate(self, state: TrainerState) -> bool:
        cur_epoch = int(state.epoch)
        cur_step = int(state.global_step)

        if self.model_aggregation_count >= self.max_aggregation:
            return False

        if cur_epoch > self.max_epoch:
            return False

        strategy = self.fed_args.aggregate_strategy

        if strategy == AggregateStrategy.EPOCH.value:
            if cur_epoch > self.epochs_trained and (cur_epoch - self.epochs_trained) % self.aggregate_freq == 0:
                return True
        elif strategy == AggregateStrategy.STEP.value:
            if cur_step > self.steps_trained and (cur_step - self.steps_trained) % self.aggregate_freq == 0:
                return True

        return False

    def inc_model_agg_count(self):
        self.model_aggregation_count += 1
        self.report()

    def inc_loss_agg_count(self):
        self.loss_aggregation_count += 1


class LossLoggingCallback(TrainerCallback):
    def __init__(self, ctx: Context):
        self.ctx = ctx
        self.sub_ctx = self.ctx.sub_ctx("nn_loss")

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        latest_log = state.log_history[-1]
        logger.info(latest_log)
        if "loss" in latest_log:
            loss = latest_log["loss"]
            if args.logging_strategy == IntervalStrategy.EPOCH:
                idx = int(state.log_history[-1]["epoch"])
            elif args.logging_strategy == IntervalStrategy.STEPS:
                idx = int(state.log_history[-1]["step"])
            else:
                raise RuntimeError("unknown logging strategy")
            self.sub_ctx.indexed_ctx(idx).metrics.log_loss("loss", loss)


class FedParameterAlignCallback(TrainerCallback):
    def __init__(
        self,
        trainer_class,
        ctx: Context,
        training_args: TrainingArguments,
        fed_args: FedArguments,
        is_server: bool = False,
    ) -> None:
        super().__init__()
        self.trainer_class = trainer_class
        self.ctx = ctx
        self.is_server = is_server
        self.training_args = training_args
        self.can_aggregate_loss = False
        self.fed_args = fed_args
        self._suffix = "fed_para"
        self._send_count = 0
        self._parameters = None
        self._aggregation_checker = None

    def get_aggregation_checker(self):
        return self._aggregation_checker

    def _client_send_parameters(self, state: TrainerState, args, train_dataloader):
        # client need to compute: epochs, max_steps, num_step_per_epoch, trained_epoch, trained_steps
        # and sync with server

        # compute num_train_epochs, max_steps
        len_dataloader = None

        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)

        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as
            # necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps

        # warm start related variables
        epochs_trained = state.global_step // num_update_steps_per_epoch
        if not args.ignore_data_skip:
            steps_trained_in_current_epoch = state.global_step % (num_update_steps_per_epoch)
            steps_trained_in_current_epoch *= args.gradient_accumulation_steps
        else:
            steps_trained_in_current_epoch = 0

        max_aggregation, aggregate_freq = compute_max_aggregation(
            self.fed_args, num_train_epochs, max_steps, epochs_trained, state.global_step
        )
        logger.info("computed max_aggregation is {}".format(max_aggregation))

        # if able to aggregate loss
        if can_aggregate_loss(args, self.fed_args):
            self.can_aggregate_loss = True

        # send parameters
        parameters = {
            "num_train_epochs": num_train_epochs,
            "max_steps": max_steps,
            "num_update_steps_per_epoch": num_update_steps_per_epoch,
            "epochs_trained": epochs_trained,
            "steps_trained_in_current_epoch": steps_trained_in_current_epoch,
            "max_aggregation": max_aggregation,
            "aggregate_freq": aggregate_freq,
            "aggregation_strategy": self.fed_args.aggregate_strategy,
            "can_aggregate_loss": self.can_aggregate_loss,
        }

        logger.info("parameters is {}".format(parameters))
        self._parameters = parameters

        if args.world_size <= 1 or args.local_rank == 0:
            self.ctx.arbiter.put(self._suffix + "_" + str(self._send_count), parameters)
            self.can_aggregate_loss = self.ctx.arbiter.get("agg_loss_" + str(self._send_count))

            if args.world_size > 1:
                can_agg_loss_t = torch.tensor([self.can_aggregate_loss], dtype=torch.bool).cuda(args.device)
                can_agg_loss_array = [can_agg_loss_t for _ in range(args.world_size)]
                dist.scatter(can_agg_loss_t, can_agg_loss_array, async_op=False)
        else:
            can_agg_loss_t = torch.tensor([self.can_aggregate_loss], dtype=torch.bool).cuda(args.device)
            dist.scatter(can_agg_loss_t, src=0, async_op=False)
            self.can_aggregate_loss = can_agg_loss_t.item()

        self._send_count += 1
        self.trainer_class.aggregation_checker = AggregationChecker(
            self.fed_args,
            max_aggregation,
            aggregate_freq,
            num_train_epochs,
            max_steps,
            epochs_trained,
            state.global_step,
            self.can_aggregate_loss,
        )

    def get_parameters(self):
        return self._parameters

    def _startegy_type(self, strategy):
        # by step or by epoch
        by_step = set([AggregateStrategy.STEP.value])
        by_epoch = set([AggregateStrategy.EPOCH.value])
        if strategy in by_step:
            return "by_step"
        elif strategy in by_epoch:
            return "by_epoch"
        else:
            raise ValueError("strategy {} not supported".format(strategy))

    def _check_fed_strategy(self, parameters):
        # check the fed strategy, assert all clients has the same startegy
        all_cilent_strategy = [p["aggregation_strategy"] for p in parameters]
        logger.info("all client strategies are {}".format(all_cilent_strategy))
        strategy_flag = self._startegy_type(all_cilent_strategy[0])
        for p in all_cilent_strategy[1:]:
            if self._startegy_type(p) != strategy_flag:
                raise ValueError(
                    "fed strategy not match, all clients has to have the same strategy: by epoch(epoch) or by step(step, progress_percentage),\n \
                                 please check: {}".format(
                        all_cilent_strategy
                    )
                )

        return strategy_flag

    def _check_federation_round(self, parameters):
        agg_round = [p["max_aggregation"] for p in parameters]
        if len(set(agg_round)) != 1:
            raise ValueError(
                "federation round not match, all clients has to have the same aggregation round,\n \
                              please check: {}".format(
                    agg_round
                )
            )
        return agg_round[0]

    def _check_aggregate_loss(self, parameters):
        aggregate_loss = True
        for p in parameters:
            flag = p["can_aggregate_loss"]
            if not flag:
                return False
        return aggregate_loss

    def _server_check_parameters(self):
        # check if all clients parameters of aggregation match
        para_1 = self.ctx.hosts.get(self._suffix + "_" + str(self._send_count))
        para_2 = self.ctx.guest.get(self._suffix + "_" + str(self._send_count))
        para_1.append(para_2)
        para = para_1
        # strategy = self._check_fed_strategy(para)
        agg_round = self._check_federation_round(para)
        self.can_aggregate_loss = self._check_aggregate_loss(para)
        self._parameters = {"max_aggregation": agg_round}
        self._parameters["can_aggregate_loss"] = self.can_aggregate_loss
        self.ctx.guest.put("agg_loss_" + str(self._send_count), self.can_aggregate_loss)
        self.ctx.hosts.put("agg_loss_" + str(self._send_count), self.can_aggregate_loss)
        self._send_count += 1

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self.trainer_class.local_mode:
            logger.info("FedParameterAlignCallback: local model, skipping federated parameter checking")
            return
        else:
            if self.is_server:
                self._server_check_parameters()
            else:
                train_dataloader = kwargs["train_dataloader"]
                self._client_send_parameters(state, args, train_dataloader)


class CallbackWrapper(TrainerCallback):
    def __init__(self, ctx: Context, wrapped_trainer: "HomoTrainerMixin"):
        self.ctx = ctx
        self.wrapped_trainer = wrapped_trainer
        self.fed_arg = self.wrapped_trainer._fed_args

    def _call_wrapped(self, ctx, aggregator, fed_arg, event_name: str, **kwargs):
        event = getattr(self.wrapped_trainer, event_name)
        kwargs["scheduler"] = kwargs.pop("lr_scheduler", None)
        train_dataloader = kwargs.pop("train_dataloader", None)
        eval_dataloader = kwargs.pop("eval_dataloader", None)
        dataloaders = tuple(filter(None, (train_dataloader, eval_dataloader)))
        kwargs["dataloader"] = dataloaders
        return event(ctx, aggregator, fed_arg, **kwargs)


class WrappedFedCallback(CallbackWrapper):
    def __init__(self, ctx: Context, wrapped_trainer: "HomoTrainerMixin"):
        super().__init__(ctx, wrapped_trainer)

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # initialize aggregator
        # doesnot call wrapper here, make sure aggregator is not called before
        # it is initialized
        if self.wrapped_trainer.local_mode:
            logger.info("local mode, skip federation aggregator initialization, aggregator will be None")
        else:
            self.wrapped_trainer.aggregator = self.wrapped_trainer.init_aggregator(self.ctx, self.fed_arg)

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self.wrapped_trainer.local_mode:
            return
        # aggregate loss
        if self.fed_arg.aggregate_strategy == AggregateStrategy.EPOCH.value:
            if self.wrapped_trainer.aggregation_checker.can_aggregate_loss:
                if "loss" not in state.log_history[-1]:  # only process train loss
                    return
                loss = state.log_history[-1]["loss"]
                agg_round = self.wrapped_trainer.aggregation_checker.loss_aggregation_count
                aggregator = self.wrapped_trainer.aggregator
                sub_ctx = self.ctx.sub_ctx("loss_aggregation").indexed_ctx(agg_round)
                aggregator.loss_aggregation(sub_ctx, loss)
                self.wrapped_trainer.aggregation_checker.inc_loss_agg_count()

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self.wrapped_trainer.local_mode:
            return
        if self.fed_arg.aggregate_strategy == AggregateStrategy.EPOCH.value:
            if self.wrapped_trainer.aggregation_checker.should_aggregate(state):
                logger.info("aggregation on epoch end")
                agg_round = self.wrapped_trainer.aggregation_checker.model_aggregation_count
                sub_ctx = self.ctx.sub_ctx("aggregation").indexed_ctx(agg_round)
                ret = self._call_wrapped(
                    sub_ctx,
                    self.wrapped_trainer.aggregator,
                    self.fed_arg,
                    "on_federation",
                    args=args,
                    state=state,
                    control=control,
                    **kwargs,
                )
                self.wrapped_trainer.aggregation_checker.inc_model_agg_count()
                return ret

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self.wrapped_trainer.local_mode:
            return
        if self.fed_arg.aggregate_strategy == AggregateStrategy.STEP.value:
            if self.wrapped_trainer.aggregation_checker.should_aggregate(state):
                logger.info("aggregation on step end")
                agg_round = self.wrapped_trainer.aggregation_checker.model_aggregation_count
                sub_ctx = self.ctx.sub_ctx("aggregation").indexed_ctx(agg_round)
                ret = self._call_wrapped(
                    sub_ctx,
                    self.wrapped_trainer.aggregator,
                    self.fed_arg,
                    "on_federation",
                    args=args,
                    state=state,
                    control=control,
                    **kwargs,
                )
                self.wrapped_trainer.aggregation_checker.inc_model_agg_count()
                return ret


class WrappedShortcutCallback(CallbackWrapper):
    def __init__(self, ctx: Context, wrapped_trainer: "HomoTrainerMixin"):
        super().__init__(ctx, wrapped_trainer)

    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        return self._call_wrapped(
            self.ctx,
            self.wrapped_trainer.aggregator,
            self.fed_arg,
            "on_init_end",
            args=args,
            state=state,
            control=control,
            **kwargs,
        )

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        return self._call_wrapped(
            self.ctx,
            self.wrapped_trainer.aggregator,
            self.fed_arg,
            "on_train_begin",
            args=args,
            state=state,
            control=control,
            **kwargs,
        )

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        return self._call_wrapped(
            self.ctx,
            self.wrapped_trainer.aggregator,
            self.fed_arg,
            "on_train_end",
            args=args,
            state=state,
            control=control,
            **kwargs,
        )

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        return self._call_wrapped(
            self.ctx,
            self.wrapped_trainer.aggregator,
            self.fed_arg,
            "on_epoch_begin",
            args=args,
            state=state,
            control=control,
            **kwargs,
        )

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        return self._call_wrapped(
            self.ctx,
            self.wrapped_trainer.aggregator,
            self.fed_arg,
            "on_epoch_end",
            args=args,
            state=state,
            control=control,
            **kwargs,
        )

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        return self._call_wrapped(
            self.ctx,
            self.wrapped_trainer.aggregator,
            self.fed_arg,
            "on_step_begin",
            args=args,
            state=state,
            control=control,
            **kwargs,
        )

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        return self._call_wrapped(
            self.ctx,
            self.wrapped_trainer.aggregator,
            self.fed_arg,
            "on_step_end",
            args=args,
            state=state,
            control=control,
            **kwargs,
        )


"""
Mixin Class For Federation Trainer
"""


def _parse_metrics_ensemble_result(metrics_ensemble_result):
    rs_dict = {}
    for i in metrics_ensemble_result:
        if isinstance(i, dict):
            rs_dict[i["metric"]] = i["val"]
        elif isinstance(i, list):
            for k in i:
                if isinstance(k, dict):
                    rs_dict[k["metric"]] = k["val"]

    return rs_dict


class HeteroTrainerMixin(ShortcutCallBackInterFace):
    def __init__(
        self,
        ctx: Context,
        model: nn.Module,
        training_args: TrainingArguments,
        train_set: Dataset,
        val_set: Dataset = None,
        loss_fn: nn.Module = None,
        optimizer: torch.optim.Optimizer = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        callbacks: Optional[List[TrainerCallback]] = [],
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
    ):
        super().__init__()

        self.ctx: Context = ctx
        self._callbacks = callbacks
        self._args = training_args
        self._user_compute_metric_func = compute_metrics
        self.train_dataset = train_set
        self.eval_dataset = val_set
        self.loss_func = loss_fn

    def _compute_metrics_warp_func(self, *args, **kwargs):
        if self._user_compute_metric_func is None:
            return {}
        else:
            eval_result = self._user_compute_metric_func(*args, **kwargs)
            if isinstance(self._user_compute_metric_func, MetricEnsemble):
                return _parse_metrics_ensemble_result(eval_result)
            return eval_result

    def _set_ctx_to_model(self, model: Union[HeteroNNModelGuest, HeteroNNModelHost]):
        model.set_context(self.ctx)


class HomoTrainerMixin(FedCallbackInterface, ShortcutCallBackInterFace):
    def __init__(
        self,
        ctx: Context,
        model: nn.Module,
        training_args: TrainingArguments,
        fed_args: FedArguments,
        train_set: Dataset,
        val_set: Dataset = None,
        loss_fn: nn.Module = None,
        optimizer: torch.optim.Optimizer = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        callbacks: Optional[List[TrainerCallback]] = [],
        use_hf_default_behavior: bool = False,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        local_mode: bool = False,
        save_trainable_weights_only: bool = False,
    ):
        super().__init__()

        assert isinstance(callbacks, list), "callback must be a list containing Callback objects, but got {}".format(
            callbacks
        )

        self.ctx: Context = ctx
        self.local_mode = local_mode
        self._callbacks = callbacks
        self._args = training_args
        self._fed_args = fed_args
        self._user_compute_metric_func = compute_metrics
        self.train_dataset = train_set
        self.eval_dataset = val_set
        self.loss_func = loss_fn
        self._use_hf_default_behavior = use_hf_default_behavior
        self._aggregator = None

        # for callback class to check if aggregation is needed
        self.aggregation_checker: AggregationChecker = None

        self._save_trainable_weights_only = save_trainable_weights_only

    def _compute_metrics_warp_func(self, *args, **kwargs):
        if self._user_compute_metric_func is None:
            return {}
        else:
            eval_result = self._user_compute_metric_func(*args, **kwargs)
            if isinstance(self._user_compute_metric_func, MetricEnsemble):
                return _parse_metrics_ensemble_result(eval_result)
            return eval_result

    def _handle_callback(self, callback_handler, new_callbacks):
        # remove default logger.info callback, need to use our logging
        # strategy
        new_callback_list = []
        for i in callback_handler.callbacks:
            new_callback_list.append(i)
        new_callback_list += new_callbacks
        callback_handler.callbacks = new_callback_list

    def _add_fate_callback(self, callback_handler):
        # the callback handler is Trainer.callback_handler
        # call order:
        # fed callback aggregator init(once), parameter check(once),
        # on federation of fedcallback
        # callbacks of shortcutcallback
        callback_handler.callbacks.append(LossLoggingCallback(self.ctx))
        callback_handler.callbacks.append(WrappedFedCallback(self.ctx, self))
        callback_handler.callbacks.append(
            FedParameterAlignCallback(
                self, self.ctx, fed_args=self._fed_args, training_args=self._args, is_server=False
            )
        )
        callback_handler.callbacks.append(WrappedShortcutCallback(self.ctx, self))

    def _remove_fed_callback(self, callback_class):
        self.callback_handler.callbacks = [
            c for c in self.callback_handler.callbacks if not isinstance(c, callback_class)
        ]

    def set_local_mode(self):
        self.local_mode = True
        logger.info("trainer set to local mode")

    def set_fed_mode(self):
        self.local_mode = False
        logger.info("trainer set to federated mode")

    @property
    def aggregator(self):
        return self._aggregator

    @aggregator.setter
    def aggregator(self, value):
        self._aggregator = value


"""
Base Classes of NN Trainer
"""


class HeteroTrainerBase(Trainer, HeteroTrainerMixin):
    def __init__(
        self,
        ctx: Context,
        model: Union[HeteroNNModelGuest, HeteroNNModelHost],
        training_args: TrainingArguments,
        train_set: Dataset,
        val_set: Dataset = None,
        loss_fn: nn.Module = None,
        optimizer: torch.optim.Optimizer = None,
        data_collator: Callable = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        callbacks: Optional[List[TrainerCallback]] = [],
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
    ):
        HeteroTrainerMixin.__init__(
            self,
            ctx=ctx,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            training_args=training_args,
            train_set=train_set,
            val_set=val_set,
            scheduler=scheduler,
            callbacks=callbacks,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
        )

        if data_collator is None:
            data_collator = _utils.collate.default_collate

        # concat checkpoint path if checkpoint idx is set
        if self._args.checkpoint_idx is not None:
            checkpoint_path = self._args.resume_from_checkpoint
            if checkpoint_path is not None and os.path.exists(checkpoint_path):
                checkpoint_folder = get_ith_checkpoint(checkpoint_path, self._args.checkpoint_idx)
                self._args.resume_from_checkpoint = os.path.join(checkpoint_path, checkpoint_folder)

        Trainer.__init__(
            self,
            model=model,
            args=self._args,
            train_dataset=train_set,
            eval_dataset=val_set,
            data_collator=data_collator,
            optimizers=(optimizer, scheduler),
            tokenizer=tokenizer,
            compute_metrics=self._compute_metrics_warp_func,
        )

        # update callbacks
        self.callback_handler.callbacks.append(LossLoggingCallback(ctx))


class HomoTrainerClient(Trainer, HomoTrainerMixin):

    """
    FedTrainerClient is designed to handle diverse federated training tasks.

    By extending the transformers.Trainer class, this class allows customization of the federated training,
    evaluation, and prediction processes to meet the needs of specific federateion training tasks. Users can
    override relevant methods to implement custom functionality.
    """

    def __init__(
        self,
        ctx: Context,
        model: nn.Module,
        training_args: TrainingArguments,
        fed_args: FedArguments,
        train_set: Dataset,
        val_set: Dataset = None,
        loss_fn: nn.Module = None,
        optimizer: torch.optim.Optimizer = None,
        data_collator: Callable = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        callbacks: Optional[List[TrainerCallback]] = [],
        use_hf_default_behavior: bool = False,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        local_mode: bool = False,
    ):
        # in case you forget to set evaluation_strategy
        if val_set is not None and training_args.evaluation_strategy == "no":
            training_args.evaluation_strategy = "epoch"

        HomoTrainerMixin.__init__(
            self,
            ctx=ctx,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            training_args=training_args,
            fed_args=fed_args,
            train_set=train_set,
            val_set=val_set,
            scheduler=scheduler,
            callbacks=callbacks,
            use_hf_default_behavior=use_hf_default_behavior,
            compute_metrics=compute_metrics,
            local_mode=local_mode,
        )

        if data_collator is None:
            data_collator = _utils.collate.default_collate

        # concat checkpoint path if checkpoint idx is set
        if self._args.checkpoint_idx is not None:
            checkpoint_path = self._args.resume_from_checkpoint
            if checkpoint_path is not None and os.path.exists(checkpoint_path):
                checkpoint_folder = get_ith_checkpoint(checkpoint_path, self._args.checkpoint_idx)
                self._args.resume_from_checkpoint = os.path.join(checkpoint_path, checkpoint_folder)

        Trainer.__init__(
            self,
            model=model,
            args=self._args,
            train_dataset=train_set,
            eval_dataset=val_set,
            data_collator=data_collator,
            optimizers=(optimizer, scheduler),
            tokenizer=tokenizer,
            compute_metrics=self._compute_metrics_warp_func,
        )

        self._add_fate_callback(self.callback_handler)

    def init_aggregator(self, ctx: Context, fed_arg: FedArguments):
        return None

    def compute_loss(self, model, inputs, **kwargs):
        if self._use_hf_default_behavior:
            return super().compute_loss(model, inputs, **kwargs)
        else:
            # (features, labels), this format is used in FATE-1.x
            if isinstance(inputs, tuple) or isinstance(inputs, list) and len(inputs) == 2:
                feats, labels = inputs
                output = model(feats)
                loss = self.loss_func(output, labels)
                return loss
            else:
                return super().compute_loss(model, inputs, **kwargs)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        if self._use_hf_default_behavior:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
        else:
            # (features, labels), this format is used in FATE-1.x
            # now the model is in eval status
            inputs = self._prepare_inputs(inputs)
            if isinstance(inputs, tuple) or isinstance(inputs, list) and len(inputs) == 2:
                with torch.no_grad():
                    feats, labels = inputs
                    logits = model(feats)
                    return None, logits, labels
            else:
                return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)


class HomoTrainerServer(object):
    def __init__(self, ctx: Context, local_mode: bool = False) -> None:
        self.ctx = ctx
        self.local_mode = local_mode
        self._max_steps = None
        self._parameter_check_callback = FedParameterAlignCallback(self, self.ctx, None, None, is_server=True)
        self._max_aggregation = None
        self.aggregator = None
        self.can_aggregate_loss = True

    def set_fed_context(self, ctx: Context):
        assert isinstance(ctx, Context), "ctx must be a Context object, but got {}".format(ctx)
        self.ctx = ctx

    def set_local_mode(self):
        self.local_mode = True
        logger.info("trainer set to local mode")

    def set_fed_mode(self):
        self.local_mode = False
        logger.info("trainer set to federated mode")

    def on_train_end(self, ctx: Context, aggregator: Aggregator):
        pass

    def on_train_begin(self, ctx: Context, aggregator: Aggregator):
        pass

    def on_init_end(self, ctx: Context, aggregator: Aggregator):
        pass

    def on_federation(self, ctx: Context, aggregator: Aggregator, agg_iter_idx: int):
        pass

    def init_aggregator(self, ctx: Context) -> Aggregator:
        pass

    def _aggregate_loss(self, ctx):
        pass

    def train(self):
        if self.local_mode:
            logger.info("Local model is set, skip initializing fed setting & aggregator")
            return

        self.aggregator: Aggregator = self.init_aggregator(self.ctx)
        logger.info("Initialized aggregator Done: {}".format(self.aggregator))
        self._parameter_check_callback.on_train_begin(None, None, None)  # only get parameters from clients and align
        parameters = self._parameter_check_callback.get_parameters()
        self._max_aggregation = parameters["max_aggregation"]
        self.can_aggregate_loss = parameters["can_aggregate_loss"]
        logger.info("checked parameters are {}".format(parameters))

        self.on_init_end(self.ctx, aggregator=self.aggregator)
        self.on_train_begin(self.ctx, aggregator=self.aggregator)

        ctx = self.ctx
        for i in range(self._max_aggregation):
            sub_ctx = ctx.sub_ctx("aggregation").indexed_ctx(i)
            self.on_federation(sub_ctx, aggregator=self.aggregator, agg_iter_idx=i)
            if self.can_aggregate_loss:
                loss_sub_ctx = ctx.sub_ctx("loss_aggregation").indexed_ctx(i)
                loss = self.aggregator.loss_aggregation(loss_sub_ctx)
                sub_ctx.metrics.log_loss("loss", loss)

        self.on_train_end(self.ctx, aggregator=self.aggregator)

    def predict(self):
        pass
