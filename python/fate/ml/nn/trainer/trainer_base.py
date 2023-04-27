import torch
from torch import nn
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from fate.interface import Context
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from transformers import TrainingArguments as hf_TrainingArguments
from transformers import Trainer, TrainerState, TrainerControl, EvalPrediction
from torch.optim.lr_scheduler import _LRScheduler, LambdaLR
from torch.utils.data import _utils
from fate.ml.aggregator.base import Aggregator
from transformers.trainer import logger
from transformers.trainer_callback import TrainerCallback, PrinterCallback
from typing import Optional
import time


def time_decorator(descr=""):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            logger.info(f"{descr} takes {end_time - start_time:.2f} seconds.")
            return result
        return wrapper
    return decorator


@dataclass
class FedArguments(object):
    ...


@dataclass
class TrainingArguments(hf_TrainingArguments):
    
    # By default, we disable tqdm progress bar for logging conerns.
    output_dir: str = field(default="./")
    disable_tqdm: bool = field(default=True)
    save_strategy: str = field(default="no")
    logging_strategy: str = field(default="epoch")
    evaluation_strategy: str = field(default="no")


class FedCallBackInterFace(object):

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
            **kwargs):
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
            **kwargs):
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
            **kwargs):
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
            **kwargs):
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
            **kwargs):
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
            **kwargs):
        pass

    def on_substep_end(
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
            **kwargs):
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
        **kwargs):
        pass

    def on_log(
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
        **kwargs):
        pass

    def init_aggregator(self):
        raise NotImplementedError('init_aggregator() must be implemented in subclass, init aggregator here')
    


# I dont like huggingface logging
class LogSuppressFilter(logging.Filter):
    def filter(self, record):
        suppress_list = set(
            ["\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n"]
        )
        if record.getMessage() in suppress_list:
            return False
        return True
    

class FedTrainerCallbackWrapper(TrainerCallback):

    def __init__(self, ctx: Context, wrapped_trainer: 'FedTrainerClient'):
        self.ctx = ctx
        self.wrapped_trainer = wrapped_trainer
        self.aggregator = wrapped_trainer.aggregator
        self.fed_arg = wrapped_trainer._fed_args

    def _call_wrapped(self, event_name: str, **kwargs):
        event = getattr(self.wrapped_trainer, event_name)
        kwargs['scheduler'] = kwargs.pop('lr_scheduler', None)

        train_dataloader = kwargs.pop('train_dataloader', None)
        eval_dataloader = kwargs.pop('eval_dataloader', None)
        dataloaders = tuple(filter(None, (train_dataloader, eval_dataloader)))
        kwargs['dataloader'] = dataloaders

        return event(self.ctx, self.aggregator, self.fed_arg, **kwargs)

    def on_init_end(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs):
        return self._call_wrapped(
            'on_init_end',
            args=args,
            state=state,
            control=control,
            **kwargs)

    def on_train_begin(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs):
        return self._call_wrapped(
            'on_train_begin',
            args=args,
            state=state,
            control=control,
            **kwargs)

    def on_train_end(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs):
        return self._call_wrapped(
            'on_train_end',
            args=args,
            state=state,
            control=control,
            **kwargs)

    def on_epoch_begin(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs):
        return self._call_wrapped(
            'on_epoch_begin',
            args=args,
            state=state,
            control=control,
            **kwargs)

    def on_epoch_end(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs):
        return self._call_wrapped(
            'on_epoch_end',
            args=args,
            state=state,
            control=control,
            **kwargs)

    def on_step_begin(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs):
        return self._call_wrapped(
            'on_step_begin',
            args=args,
            state=state,
            control=control,
            **kwargs)

    def on_substep_end(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs):
        return self._call_wrapped(
            'on_substep_end',
            args=args,
            state=state,
            control=control,
            **kwargs)

    def on_step_end(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs):
        return self._call_wrapped(
            'on_step_end',
            args=args,
            state=state,
            control=control,
            **kwargs)
    
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        return self._call_wrapped(
            'on_log',
            args=args,
            state=state,
            control=control,
            **kwargs)


logger.addFilter(LogSuppressFilter())


class StdFedTrainerMixin(FedCallBackInterFace):

    def __init__(self,
                 ctx: Context,
                 model: nn.Module,
                 loss_fn: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 training_args: TrainingArguments,
                 fed_args: FedArguments,
                 train_set: Dataset,
                 val_set: Dataset = None,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 callbacks: Optional[List[TrainerCallback]] = [],
                 use_hf_default_behavior: bool = False,
                 compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None
                 ):
        
        assert isinstance(
            callbacks, list), 'callback must be a list containing Callback objects, but got {}'.format(
            callbacks)
        
        self.ctx = ctx
        self._callbacks = callbacks
        self._args = training_args
        self._fed_args = fed_args
        self._user_compute_metric_func = compute_metrics
        self.train_dataset = train_set
        self.eval_dataset = val_set
        self._aggregator: Aggregator = self.init_aggregator()
        self.loss_func = loss_fn
        self._use_hf_default_behavior = use_hf_default_behavior

    @property
    def aggregator(self):
        if self._aggregator is None:
            raise RuntimeError('Aggregator is not initialized')
        return self._aggregator
        
    def _compute_metrics_warp_func(self, *args, **kwargs):
        
        if self._user_compute_metric_func is None:
            return {}
        else:
            eval_result = self._user_compute_metric_func(*args, **kwargs)
            # Do some FATEBoard Callback here
            return eval_result
    
    def _handle_callback(self, callback_handler, new_callbacks):

        # remove default printer callback, need to user our logging strategy
        new_callback_list = []
        for i in callback_handler.callbacks:
            if not isinstance(i, PrinterCallback):
                new_callback_list.append(i)
        new_callback_list += new_callbacks
        callback_handler.callbacks = new_callback_list

    def _add_fed_callback(self, callback_handler, fed_callback_inst):
        has_fed_callback = False

        for c in callback_handler.callbacks:
            if isinstance(c, type(fed_callback_inst)):
                has_fed_callback = True
                break

        if not has_fed_callback:
            callback_handler.callbacks.append(fed_callback_inst)
            
    def _remove_fed_callback(self, callback_class):
        self.callback_handler.callbacks = [
            c for c in self.callback_handler.callbacks if not isinstance(
                c, callback_class)]



class FedTrainerClient(Trainer, StdFedTrainerMixin):

    """
    FedTrainerClient is designed to handle diverse federated training tasks.

    By extending the transformers.Trainer class, this class allows customization of the federated training,
    evaluation, and prediction processes to meet the needs of specific federateion training tasks. Users can
    override relevant methods to implement custom functionality.
    """

    def __init__(self,
                 ctx: Context,
                 model: nn.Module,
                 loss_fn: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 training_args: TrainingArguments,
                 fed_args: FedArguments,
                 train_set: Dataset,
                 val_set: Dataset = None,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 callbacks: Optional[List[TrainerCallback]] = [],
                 use_hf_default_behavior: bool = False,
                 compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None
                 ):

        # default use no lr decay
        if scheduler is None:
            scheduler = LambdaLR(optimizer, lambda x: 1)
    
        # in case you forget to set evaluation_strategy
        if val_set is not None and training_args.evaluation_strategy == 'no':
            training_args.evaluation_strategy = 'epoch'

        
        StdFedTrainerMixin.__init__(self,
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
                                    compute_metrics=compute_metrics
                                    )

        Trainer.__init__(self,
                        model=model,
                        args=self._args,
                        train_dataset=train_set,
                        eval_dataset=val_set,
                        data_collator=_utils.collate.default_collate,
                        optimizers=[optimizer, scheduler],
                        compute_metrics=self._compute_metrics_warp_func
                        )
        
        print(self._fed_args)
        fed_trainer_callback = FedTrainerCallbackWrapper(self.ctx, self)
        self._handle_callback(self.callback_handler, self._callbacks)
        self._add_fed_callback(self.callback_handler, fed_trainer_callback)

    def init_aggregator(self):
        return None

    def compute_loss(self, model, inputs, **kwargs):

        if self._use_hf_default_behavior:
            return super().compute_loss(model, inputs, **kwargs)
        else:
            feats, labels = inputs
            logits = model(feats)
            loss = self.loss_func(logits.flatten(), labels.flatten())
            return loss

    def prediction_step(self,
                        model: nn.Module,
                        inputs: Dict[str,
                                     Union[torch.Tensor,
                                           Any]],
                        prediction_loss_only: bool,
                        ignore_keys: Optional[List[str]] = None):
        
        if self._use_hf_default_behavior:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
        else:
            with torch.no_grad():
                feats, labels = inputs
                logits = model(feats)
            return (None, logits, labels)
        

class FedTrainerServer(FedCallBackInterFace):

    def __init__(self, ctx: Context, training_args: TrainingArguments, fed_args: FedArguments) -> None:
        self.ctx = ctx
        self.training_args = training_args
        self.aggregator: Aggregator = self.init_aggregator()
        self._args = training_args
        self._fed_args = fed_args

    def init_aggregator(self):
        return None

    def train(self):
        self.on_init_end(self.ctx, aggregator=self.aggregator, args=self._args, fed_args=self._fed_args)
        self.on_train_begin(self.ctx, aggregator=self.aggregator, args=self._args, fed_args=self._fed_args)
        for epoch in range(self.training_args.num_train_epochs):
            self.on_epoch_begin(self.ctx, aggregator=self.aggregator, args=self._args, fed_args=self._fed_args)
            # for step in range(self.training_args.num_steps_per_epoch):
            #     self.on_step_begin(self.ctx, aggregator=self.aggregator, args=self._args, fed_args=self._fed_args, step=step)
            #     self.on_substep_end(self.ctx, aggregator=self.aggregator, args=self._args, fed_args=self._fed_args)
            #     self.on_step_end(self.ctx, aggregator=self.aggregator, args=self._args, fed_args=self._fed_args, step=step)
            self.on_epoch_end(self.ctx, aggregator=self.aggregator, args=self._args, fed_args=self._fed_args)
        self.on_train_end(self.ctx, aggregator=self.aggregator, args=self._args, fed_args=self._fed_args)

