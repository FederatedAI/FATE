import torch
from torch import nn
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from fate.interface import Context
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from transformers import TrainingArguments, Trainer, TrainerState, TrainerControl
from torch.optim.lr_scheduler import _LRScheduler, LambdaLR
from torch.utils.data import _utils
from fate.ml.aggregator.base import Aggregator
from transformers.trainer import logger
from transformers.trainer_callback import TrainerCallback



@dataclass
class FedArguments(object):
    ...


from typing import Optional

class FedCallBackInterFace(object):

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

    def __init__(self, ctx: Context, wrap_class: 'FedTrainerClient'):
        self.ctx = ctx
        self.wrap_class = wrap_class
        self.aggregator = wrap_class.aggregator
        self.fed_arg = wrap_class._fed_args

    def _call_wrapped(self, event_name: str, **kwargs):
        event = getattr(self.wrap_class, event_name)
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


logger.addFilter(LogSuppressFilter())


class FedTrainerClient(Trainer, FedCallBackInterFace):

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
                 use_hf_default_behavior: bool = False):

        assert isinstance(
            callbacks, list), 'callback must be a list containing Callback objects, but got {}'.format(
            callbacks)
        self._callbacks = callbacks

        self._args = training_args
        self._fed_args = fed_args

        super().__init__(model=model,
                    args=self._args,
                    train_dataset=train_set,
                    eval_dataset=val_set,
                    data_collator=_utils.collate.default_collate,
                    optimizers=[optimizer, scheduler],
                    callbacks=callbacks
                    )
        
        self.ctx = ctx
        self.train_dataset = train_set
        self.eval_dataset = val_set
        logger.info('initializing aggregator')
        self.aggregator: Aggregator = self.init_aggregator()
        self.loss_func = loss_fn
        self._loss = []
        self._use_hf_default_behavior = use_hf_default_behavior

        assert isinstance(
            callbacks, list), 'callback must be a list containing Callback objects, but got {}'.format(callbacks)
        # default use no lr decay
        if scheduler is None:
            scheduler = LambdaLR(optimizer, lambda x: 1)

    def _add_fed_callback(self):
        has_fed_callback = False
        for c in self.callback_handler.callbacks:
            if isinstance(c, FedTrainerCallbackWrapper):
                has_fed_callback = True
                break
        if not has_fed_callback:
            self.callback_handler.callbacks.append(
                FedTrainerCallbackWrapper(self.ctx, self))

    def _remove_fed_callback(self):
        self.callback_handler.callbacks = [
            c for c in self.callback_handler.callbacks if not isinstance(
                c, FedTrainerCallbackWrapper)]
        
    def init_aggregator(self):
        return None

    def compute_loss(self, model, inputs, **kwargs):

        if self._use_hf_default_behavior:
            return super().compute_loss(model, inputs, **kwargs)
        else:
            feats, labels = inputs
            logits = model(feats)
            loss = self.loss_func(logits.flatten(), labels.flatten())
            self._loss.append(loss)
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

    def train(
        self,
        local_training: bool = False,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
        **kwargs,
    ):
        # if local training is True, then the training process will be executed locally
        # federateion callback will be removed
        if local_training:
            self._remove_fed_callback()
        else:
            self._add_fed_callback()

        return super().train(resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)
    
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

